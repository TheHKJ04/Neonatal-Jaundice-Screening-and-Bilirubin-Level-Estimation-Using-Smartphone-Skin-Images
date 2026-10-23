# Phase 7.6 — Frozen classification test evaluation

## What this phase does
Evaluates the saved best classification checkpoint against the held-out test manifest, with image-level and patient-level metrics and patient-level bootstrap confidence intervals. It does not train or tune a model.

## Critical label limitation
The classifier is trained against `treatment_label`. Until source documentation confirms what that field means, results describe agreement with that dataset label only—not clinical jaundice diagnosis, severity, or treatment recommendations.

## Run in Colab
```bash
cd /content/neojaundice/neojaundice
python -m pytest tests/unit/test_evaluate_classification_test.py -v
python scripts/evaluate_classification_test.py \
  --project-root /content/neojaundice/neojaundice \
  --manifest-dir data/processed/manifests \
  --image-root /content/neojaundice_data/NeoJaundice \
  --checkpoint artifacts/checkpoints/classification_treatment_seed42/best_model.pt \
  --output-dir artifacts/evaluation/classification \
  --batch-size 32 --workers 2 --threshold 0.5 --bootstrap-replicates 2000
```

If test CSV is not named `test.csv`, `test_manifest.csv`, or `test_split.csv`, add `--test-manifest /absolute/path/to/test.csv`.

## Outputs
- `test_predictions.csv`: image-level true label, predicted probability, thresholded prediction.
- `patient_predictions.csv`: per-patient mean probability and label.
- `metrics.json`: image and patient metrics, confusion counts, checkpoint hash, bootstrap intervals.
- `evaluation_report.md`: human-readable report.

## Metrics
Accuracy, precision, sensitivity/recall, specificity, F1, ROC-AUC (when both classes exist), average precision, Brier score, log loss, confusion matrix. Patient-level bootstrap intervals resample patients, not image rows.

## Rules
- Evaluate the held-out test set only after freezing the checkpoint and threshold.
- Do not adjust threshold or model based on test outcomes.
- If the dataset target's clinical semantics are unknown, report it as dataset-label classification.
- For a clinical application, external validation, calibration analysis, subgroup evaluation, prospective validation, and clinical governance are still required.
