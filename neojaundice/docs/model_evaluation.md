# NeoJaundice Model Evaluation

## Objective

Evaluate neonatal jaundice classification and
bilirubin regression models.

## Dataset

Dataset: NeoJaundice

Split: Patient-independent

Training: 1563 images
Validation: 336 images
Testing: 336 images

## Classification

Target: treatment_label

Metrics:
- Accuracy
- Precision
- Recall
- Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

## Regression

Target: bilirubin_value

Metrics:
- MAE
- MSE
- RMSE
- R2
- Mean Bias
- Pearson Correlation
- Spearman Correlation
- Bland-Altman Analysis

## Statistical Analysis

Patient-cluster bootstrap confidence intervals.

## Reproducibility

Random seed: 42

Model checkpoint: To be recorded
Dataset version: To be recorded
Evaluation date: To be recorded

## Limitations

Research evaluation does not establish
clinical diagnostic validity.
