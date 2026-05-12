# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KTH course project for DT2119 (Speech and Speaker Recognition). The goal is to compare two Whisper-based ASR models on a Scandinavian speech dataset:

- **`KBLab/kb-whisper-large`** — Swedish/Scandinavian fine-tuned Whisper
- **`openai/whisper-large`** — original OpenAI Whisper
- **Dataset**: `NbAiLab/NST` (Norwegian Speech Technology corpus), `no-both` config (close + distant mic)

## Environment

Uses a local virtualenv at `.venv/`. Activate before running:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

First run downloads models and dataset to HuggingFace cache (`~/.cache/huggingface/`). Models are large (~3 GB each); the NST dataset is also large.

## Key Technical Notes

- `datasets` is pinned to `<3.0` — the API changed in v3 and existing code may break on newer versions.
- The dataset split options for `NbAiLab/NST` `no-both` config are `"train"` and `"test"`.
- Both models share the same `AutoProcessor` class; `model_kb` and `model_whisp` are both `AutoModelForSpeechSeq2Seq` instances.
- Audio in NST is typically 16 kHz; pass it through the processor before feeding to the model.
