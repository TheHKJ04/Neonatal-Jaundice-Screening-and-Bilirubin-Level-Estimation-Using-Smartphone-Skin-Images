# Machine-learning pipeline

## Dataset and target

The currently supplied classification scripts validate a binary `treatment_label` target with values 0 and 1. Their own run/evaluation records mark target semantics as unverified. Therefore, the labels must not be silently renamed to clinical “jaundice/no jaundice” ground truth.

The regression dataset manifest uses `bilirubin_value` as the known target column in the supplied project context; confirm units and measurement provenance for each split before training/evaluation.

## Reproducible pipeline

1. Validate manifests, paths, image integrity, patient identifiers, labels, units, and missingness.
2. Split at patient level to prevent images from the same newborn leaking across train/validation/test.
3. Fit preprocessing only on training data where applicable; record all preprocessing options.
4. Train and select checkpoints using training/validation data only.
5. Freeze checkpoint and thresholds before held-out test evaluation.
6. Record random seed, package versions, manifest hashes, checkpoint hashes, parameters, metrics, and run ID.
7. Report patient-level as well as image-level metrics where appropriate.
8. Evaluate subgroup performance and calibration only when sample sizes support meaningful analysis.
9. Do not claim clinical validity from performance against an unverified dataset label.

## Metrics

- Regression: MAE, RMSE, bias, and agreement plots; report the bilirubin unit and target distribution.
- Classification: sensitivity/recall, specificity, precision, F1, ROC-AUC, confusion matrix, calibration, and confidence intervals.
- For screening, false negatives and the patient-level unit of analysis need explicit discussion.

## Experiment and dataset versioning

MLflow should track run parameters, metrics, and artifacts; DVC or an equivalent immutable manifest/hash scheme should version datasets and splits. Never store private newborn images in a public repository. A tool being listed in the blueprint does not prove integration is complete.
