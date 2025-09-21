# Multi-Scale Optimization — Starter (macOS Apple Silicon)

This starter is tuned for a Mac M1/M2 on **CPU** (and can optionally use Metal GPU).
It sets up a baseline CIFAR-10 model and exports TFLite models for edge experiments.

## Quickstart

```bash
# 0) Create & activate a clean env (choose one)
# With conda/mamba:
conda create -n mso python=3.10 -y && conda activate mso
# OR with venv:
python3 -m venv .venv && source .venv/bin/activate

# 1) Install deps
pip install -U pip
pip install -r requirements.txt

# 2) Run baseline (CPU by default)
python part1_baseline.py

# Artifacts will appear under:
# - models/baseline_model.keras
# - exports/*.tflite
# - reports/baseline_metrics.json
# - charts/history_accuracy_loss.png
```

> If you **don't** want to use the Apple GPU, do nothing (CPU is default).
> If you **do** want GPU acceleration, keep `tensorflow-metal` installed—no extra code changes needed.

## Files
- `part1_baseline.py`: trains CIFAR-10 baseline with callbacks, evaluates, exports TFLite, logs metrics.
- `requirements.txt`: minimal dependencies for Apple Silicon.
- `reports/`, `models/`, `exports/`, `charts/`: output folders.