"""
inference.py is used to load the model from the config,
run the test set through the model in order to make a trasncription.
That transcription is then stored, and used by metrics.py to cerate
the appropriate metrics (e.g. word error rate).
"""

import json
import logging
import os
import signal
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import logging as hf_logging
import warnings

# silence the noisy (but harmless) transformers warnings around Whisper generation:
# duplicate logits processors, max_length vs max_new_tokens, missing attention mask.
# Some come via transformers' logger, others via Python warnings.warn() — silence both.
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers.*")

# transformers tries to pre-allocate the full model size in one CUDA chunk before loading
# weights. On MIG instances with cudaMallocAsync this single big allocation OOMs even when
# layer-by-layer loading would fit. Disable the warmup (it is only a perf optimization).
import transformers.modeling_utils as _modeling_utils
_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


# safety net so a single pathological batch can't stall the whole run
class BatchTimeout(Exception):
    pass


def _batch_timeout_handler(signum, frame):
    raise BatchTimeout()


signal.signal(signal.SIGALRM, _batch_timeout_handler)

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
        # device_map={"": "cuda:0"} forces all layers onto the GPU without memory estimation.
        # device_map="auto" triggers accelerate's NVML-based memory check, which fails on MIG
        # instances (misreports available VRAM), causing CPU offloading and a CUDA allocator crash.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.model_id,
            dtype=torch_dtype,
            device_map={"": "cuda:0"},
        )
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
    batch_timeout_s = getattr(config, "batch_timeout_s", 30)
    n_timed_out_batches = 0
    n_timed_out_samples = 0

    # loops through each batch
    for batch_idx, start in enumerate(tqdm(range(0, len(ds), config.batch_size), total=n_batches, desc=config.run_name)):
        batch = ds[start : start + config.batch_size]

        # extracts relevant info from the raw data / audio files
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        references = batch[config.text_column]
        audio_durations = [len(arr) / sr for arr, sr in zip(audio_arrays, sampling_rates)]

        # NST contains silent takes (audio with no speech but with a meta-instruction
        # like "be quiet during this recording" in the reference text). Whisper enters
        # a runaway hallucination loop on silence and stalls inference for many minutes,
        # so we detect silent samples and assign them an empty hypothesis directly.
        silent_threshold = getattr(config, "silent_rms_threshold", 0.001)
        silent_flags = [float(np.sqrt(np.mean(np.square(arr)))) < silent_threshold for arr in audio_arrays]
        active_idx = [i for i, s in enumerate(silent_flags) if not s]

        hypotheses = [""] * len(audio_arrays)
        inference_time = 0.0
        if active_idx:
            active_arrays = [audio_arrays[i] for i in active_idx]
            input_features = preprocess_batch(active_arrays, processor)
            input_features = input_features.to(device=device, dtype=next(model.parameters()).dtype)

            t0 = time.perf_counter()
            signal.alarm(batch_timeout_s)
            try:
                with torch.no_grad():
                    # inference; model predicts tokens based on audio inputs.
                    # max_new_tokens caps runaway hallucinations; no_repeat_ngram_size=3
                    # breaks the repetition loops Whisper falls into on repetitive audio.
                    predicted_ids = model.generate(
                        input_features,
                        language=config.language,
                        task=config.task,
                        max_new_tokens=getattr(config, "max_new_tokens", 200),
                        no_repeat_ngram_size=3,
                    )
                inference_time = time.perf_counter() - t0
                active_hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                for i, hyp in zip(active_idx, active_hyps):
                    hypotheses[i] = hyp
            except BatchTimeout:
                inference_time = time.perf_counter() - t0
                n_timed_out_batches += 1
                n_timed_out_samples += len(active_idx)
                logger.warning(
                    "batch=%d timed out after %ds — leaving %d samples with empty hypothesis",
                    batch_idx, batch_timeout_s, len(active_idx),
                )
            finally:
                signal.alarm(0)

        per_sample_time = inference_time / max(1, len(active_idx))

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

    # loads the saved csv file with the results, and calcs the evaluation metrics.
    # keep_default_na=False prevents pandas turning empty transcriptions into NaN (float),
    # which would crash jiwer downstream.
    df_saved = pd.read_csv(transcriptions_path, keep_default_na=False)
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
        "timed_out_batches": n_timed_out_batches,
        "timed_out_samples": n_timed_out_samples,
        "config": vars(config),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)
    logger.info("WER=%.4f CER=%.4f mean_RTF=%.4f", wer, cer, rtf)
    if n_timed_out_batches:
        logger.warning(
            "timed_out_batches=%d timed_out_samples=%d (hypotheses left empty for those samples)",
            n_timed_out_batches, n_timed_out_samples,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ASR inference")
    parser.add_argument("--config", required=True, help="Path to inference YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = types.SimpleNamespace(**yaml.safe_load(f))

    run_inference(cfg)
