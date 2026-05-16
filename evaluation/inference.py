"""
inference.py is used to load the model from the config,
run the test set through the model in order to make a trasncription.
That transcription is then stored, and used by metrics.py to cerate
the appropriate metrics (e.g. word error rate).
"""

import json
import logging
import os
import sys
import time
import types

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess import preprocess_batch
from evaluation.metrics import compute_cer, compute_rtf, compute_wer


# loads the model based on 'model_id', specified in the config file
def load_model(config):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    processor = AutoProcessor.from_pretrained(config.model_id)

    # GPU = float16, CPU = float32
    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    if device == "cuda":
        # device_map="auto" avoids the .to(device) CUDA allocator bug present on MIG instances
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_id, dtype=torch_dtype, device_map="auto")
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_id, dtype=torch_dtype)
        model.to(device)

    # if the model is set to lora, load the adapter on top
    if config.model_type == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.lora_adapter_path)

    model.eval()
    return model, processor, device


# the main loop running inference / test set
def run_inference(config):

    # dir configurations and loggers, so we store everything correctly
    out_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(out_dir, "inference.log"))
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    logger.info("run_name=%s model_type=%s", config.run_name, config.model_type)

    # loads the model
    model, processor, device = load_model(config)
    logger.info("device=%s", device)

    # loads the test set
    ds = load_dataset(config.dataset, config.dataset_config, split="test", trust_remote_code=True)

    # we could use max_samples in the config, if inference is too slow to run with all samples
    if config.max_samples is not None:
        ds = ds.select(range(config.max_samples))
    logger.info("Loaded %d samples", len(ds))

    results = []
    n_batches = (len(ds) + config.batch_size - 1) // config.batch_size

    # loops through each batch
    for batch_idx, start in enumerate(tqdm(range(0, len(ds), config.batch_size), total=n_batches, desc=config.run_name)):
        batch = ds[start : start + config.batch_size]

        # extracts relevant info from the raw data / audio files
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        references = batch[config.text_column]
        audio_durations = [len(arr) / sr for arr, sr in zip(audio_arrays, sampling_rates)]

        # preprocessing of the extracted audio data from the dataset
        input_features = preprocess_batch(audio_arrays, processor)
        input_features = input_features.to(device=device, dtype=next(model.parameters()).dtype)

        t0 = time.perf_counter()
        with torch.no_grad():
            # inference; model predicts tokens based on audio inputs
            predicted_ids = model.generate(
                input_features,
                language=config.language,
                task=config.task,
            )
        inference_time = time.perf_counter() - t0
        per_sample_time = inference_time / len(audio_arrays)

        # translates the token outputs from the model into readable text
        hypotheses = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # creates a nice object / dict with the results, for storage / documentation
        for i, (ref, hyp, dur) in enumerate(zip(references, hypotheses, audio_durations)):
            results.append({
                "id": start + i,
                "reference": ref,
                "hypothesis": hyp,
                "audio_duration_s": dur,
                "inference_time_s": per_sample_time,
                "rtf": per_sample_time / dur,
            })
        if batch_idx % 10 == 0:
            logger.info("batch=%d samples_processed=%d", batch_idx, start + len(audio_arrays))

    # stores the results, paths and everything needed to navigate back to them
    # NOTE: transcriptions are saved first so metrics can be re-run later without re-running inference
    df = pd.DataFrame(results)
    transcriptions_path = os.path.join(out_dir, "transcriptions.csv")
    df.to_csv(transcriptions_path, index=False)
    logger.info("Transcriptions saved to %s", transcriptions_path)

    # loads the saved csv file with the results, and calcs the evaluation metrics
    df_saved = pd.read_csv(transcriptions_path)
    wer = compute_wer(df_saved["reference"].tolist(), df_saved["hypothesis"].tolist())
    cer = compute_cer(df_saved["reference"].tolist(), df_saved["hypothesis"].tolist())
    rtf = compute_rtf(df_saved["inference_time_s"].tolist(), df_saved["audio_duration_s"].tolist())

    # stores the final results and writes to a file + logs everything
    summary = {
        "run_name": config.run_name,
        "num_samples": len(df_saved),
        "wer": wer,
        "cer": cer,
        "mean_rtf": rtf,
        "config": vars(config),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)
    logger.info("WER=%.4f CER=%.4f mean_RTF=%.4f", wer, cer, rtf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ASR inference")
    parser.add_argument("--config", required=True, help="Path to inference YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = types.SimpleNamespace(**yaml.safe_load(f))

    run_inference(cfg)
