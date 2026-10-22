# Phase 7.4 Runbook — Regression Test Evaluation

## Run from the repository root

```bash
python scripts/evaluate_test.py \
  --manifest-dir data/processed/neojaundice \
  --image-root data/external/neojaundice/extracted/NeoJaundice \
  --checkpoint /path/to/selected/best_model.pt \
  --output-dir artifacts/evaluation/regression \
  --bootstrap-replicates 1000
```

PowerShell:

```powershell
python scripts/evaluate_test.py `
  --manifest-dir data/processed/neojaundice `
  --image-root data/external/neojaundice/extracted/NeoJaundice `
  --checkpoint "D:\path\to\best_model.pt" `
  --output-dir artifacts/evaluation/regression `
  --bootstrap-replicates 1000
```

Use paths that exist in the environment where the command runs. A Google Drive path from Colab is not automatically available on Windows.

## Expected completion
The command prints image-level metrics, patient-level metrics, the training-mean baseline, checkpoint hash, and output directory. It creates output files automatically.

## If checkpoint loading fails
Do not alter model weights or silently ignore missing/unexpected state-dictionary keys. Verify that `build_model()` is the same architecture used during training and inspect the checkpoint format.

## If the target column is absent
Inspect `train.csv` columns and pass the correct `--target-column`. The default is `blood(mg/dL)`, but its meaning must be confirmed from source documentation.
