"""
One-off diagnostic: process samples in a fixed range one at a time with timing,
to find the specific sample that stalls full-batch inference.
"""

import os
import signal
import sys
import time
import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.preprocess import preprocess_batch


class StallTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise StallTimeout()


signal.signal(signal.SIGALRM, _alarm)
TIMEOUT_S = 30

MODEL_ID = "KBLab/kb-whisper-large"
LO, HI = 255, 275

ds = load_dataset("NbAiLab/NST", "no-close", split="test", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map={"": "cuda:0"}
)
model.eval()

print(f"{'i':>4}  {'dur_s':>7}  {'gen_s':>7}  {'tokens':>6}  hyp", flush=True)
for i in range(LO, HI):
    a = ds[i]["audio"]
    dur = len(a["array"]) / a["sampling_rate"]

    print(f"  -> processing i={i} dur={dur:.2f}s ...", flush=True)
    feats = preprocess_batch([a["array"]], processor).to(device="cuda:0", dtype=torch.float16)
    t_gen0 = time.perf_counter()
    signal.alarm(TIMEOUT_S)
    try:
        with torch.no_grad():
            ids = model.generate(
                feats,
                language="no",
                task="transcribe",
                max_new_tokens=100,
                no_repeat_ngram_size=3,
            )
    except StallTimeout:
        print(f"{i:>4}  {dur:>7.2f}     >{TIMEOUT_S}s  STALLED — likely culprit", flush=True)
        continue
    finally:
        signal.alarm(0)
    t_gen1 = time.perf_counter()
    hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
    n_tok = ids.shape[-1]
    print(f"{i:>4}  {dur:>7.2f}  {t_gen1 - t_gen0:>7.2f}  {n_tok:>6}  {hyp[:80]!r}", flush=True)
