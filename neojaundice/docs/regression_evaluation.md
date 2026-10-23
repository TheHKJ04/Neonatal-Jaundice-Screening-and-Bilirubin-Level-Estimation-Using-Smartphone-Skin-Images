# Regression Test Evaluation Protocol

## Purpose
Evaluate the selected NeoJaundice regression checkpoint once on the held-out test split. This script does not train or tune the model.

## Inputs
- `data/processed/neojaundice/train.csv`
- `data/processed/neojaundice/test.csv`
- Extracted image root used by the training pipeline
- Selected `best_model.pt` checkpoint

## Outputs
By default, `artifacts/evaluation/regression/` is created automatically and contains:
- `test_predictions.csv`: per-image predictions and errors
- `patient_predictions.csv`: per-patient aggregated predictions
- `metrics.json`: metrics, baseline, checkpoint hash, and evaluation metadata
- `predicted_vs_actual.png`
- `residual_plot.png`
- `evaluation_report.md`

## Metrics
Image-level and patient-level MAE/RMSE are reported. The baseline predicts the mean target from `train.csv` only. Confidence intervals are estimated by resampling patients, not individual images.

## Reproducibility
The selected checkpoint SHA-256 is written to `metrics.json`. Record the command, software environment, dataset version, and MLflow run ID alongside the report.

## Research safeguards
- Do not tune model parameters using test results.
- Keep patient groups disjoint across train/validation/test.
- Confirm target measurement semantics and units from the original dataset documentation.
- This is dataset evaluation, not clinical validation or authorization for treatment decisions.
