# Phase 7.5 — Classification training runbook

## Purpose
Train an EfficientNet-B0 model against `treatment_label` using existing `scripts/train_tracked.py` and track results in MLflow. This label's clinical meaning is **not yet verified**. This phase does not evaluate on the held-out test set.

## Prerequisites
- Existing trainer at `scripts/train_tracked.py`, dataset module at `ml/training/dataset.py`.
- Existing train/val/test manifests. Pass their directory using `--manifest-dir` if different from default.
- Dataset images accessible via `--image-root`.
- PyTorch, torchvision, MLflow and existing training dependencies installed.
- Confirm manually that `NeoJaundiceDataset(task='classification')` returns the numeric `treatment_label` as its target. The runner only checks that the source contains the column name, not full semantics.

## Colab command
```bash
cd /content/neojaundice/neojaundice
python scripts/run_classification_phase75.py \
  --project-root /content/neojaundice/neojaundice \
  --manifest-dir data/processed/manifests \
  --image-root /content/neojaundice_data/NeoJaundice \
  --output-dir artifacts/checkpoints/classification_treatment_seed42 \
  --epochs 15 --patience 4 --batch-size 16 \
  --learning-rate 0.0001 --seed 42 --workers 2 \
  --mlflow-uri sqlite:////content/neojaundice_mlflow/mlflow.db
```
If manifests live elsewhere, change `--manifest-dir` accordingly. Use `--dry-run` to check the generated command without training.

## Validation
```bash
python -m pytest tests/unit/test_phase75_classification_runner.py -v
ls -lh artifacts/checkpoints/classification_treatment_seed42
```
Expected outputs: `best_model.pt`, `last_state.pt`, `history.json`, `phase75_run_record.json`.

## Safety and reproducibility
- Do not interpret treatment label as clinically validated jaundice status or treatment recommendation.
- Preserve test data until the checkpoint and threshold are frozen.
- Do not use test metrics to choose epochs, augmentations, or thresholds.
- The run record captures checkpoint SHA-256. Dataset version/hash must be captured separately with DVC before release.
- If interruption occurs, check existing trainer's `--resume` behavior before restarting. It may create a new MLflow run.

## Next phase
Freeze the classification checkpoint and decision threshold based on validation only. Evaluate image-level and patient-level performance on test, including confusion matrix, sensitivity, specificity, precision, F1, ROC-AUC and confidence intervals when valid. Then start FastAPI inference API.
