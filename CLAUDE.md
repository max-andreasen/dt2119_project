# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KTH course project for DT2119 (Speech and Speaker Recognition). The goal is to fine-tune `KBLab/kb-whisper-large` on a Norwegian dataset using LoRA (PEFT), then compare the fine-tuned model against the base (untuned) `KBLab/kb-whisper-large` to measure any improvement on Norwegian speech.

- **Model**: `KBLab/kb-whisper-large` — Swedish/Scandinavian fine-tuned Whisper (baseline + LoRA fine-tuned variant)
- **Dataset**: `NbAiLab/NST` (Norwegian Speech Technology corpus), `no-both` config (close + distant mic)
- **Fine-tuning method**: LoRA via PEFT — keeps VRAM usage manageable for large Whisper models
- **Hyperparameter search**: Optuna may be used for pilot studies / LoRA hyperparameter search (rank, alpha, learning rate), depending on compute availability

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
- LoRA is applied via `peft` (`get_peft_model` / `LoraConfig`); target modules for Whisper are typically the attention projection layers (`q_proj`, `v_proj`).
- Optuna trials can be expensive — scope pilot studies to a small subset of training data or few epochs to keep search tractable.
