# NeoJaundice Machine Learning Pipeline

## Status

**Planned — dataset and model pipelines are not yet implemented.**

## 1. Research Objective

Investigate whether smartphone skin images contain
information useful for neonatal jaundice screening
and bilirubin level estimation.

## 2. Dataset Requirements

Each eligible training sample should have:

- Image identifier
- Pseudonymized patient identifier
- Smartphone image
- Acquisition timestamp
- Reference bilirubin measurement
- Reference measurement timestamp
- Measurement units
- Relevant acquisition metadata
- Applicable consent and data-use authorization

Additional clinically relevant metadata will be
defined during research protocol development.

## 3. Data Validation

Planned checks include:

- Duplicate detection
- Missing metadata
- Invalid images
- Unsupported formats
- Inconsistent units
- Invalid timestamps
- Implausible reference values
- Patient identifier consistency

## 4. Dataset Splitting

Use patient-level separation between training,
validation, and test sets.

The final splitting strategy will consider:

- Patient overlap
- Acquisition site
- Smartphone model
- Temporal effects
- Relevant population characteristics

## 5. Image Quality Assessment

Candidate checks include:

- Blur
- Underexposure
- Overexposure
- Color distortion
- Invalid image dimensions
- ROI visibility

Quality thresholds must be experimentally evaluated.

## 6. Preprocessing

Potential steps include:

1. Decode image
2. Validate dimensions
3. Assess image quality
4. Apply justified color processing
5. Extract skin ROI
6. Resize
7. Normalize
8. Convert to model tensor

The final preprocessing pipeline must be identical
between training and inference.

## 7. Feature Engineering

Candidate features include:

- Color statistics
- Channel distributions
- Region-level features
- Learned image embeddings

Features will be evaluated experimentally.

## 8. Baseline Models

Potential baselines:

- Linear regression
- Ridge regression
- Random forest
- Gradient boosting

These will provide comparison points for deep learning.

## 9. Deep Learning

Proposed backbone:

EfficientNet-B0

Potential tasks:

- Research-defined screening classification
- Bilirubin regression

The final output definitions depend on the
available labels and research protocol.

## 10. Evaluation

### Regression

Candidate metrics:

- MAE
- RMSE
- Bias
- Calibration
- Bland-Altman analysis

### Classification

Candidate metrics:

- Sensitivity
- Specificity
- Precision
- Recall
- ROC-AUC

Classification thresholds must be clinically justified.

## 11. Subgroup Analysis

Evaluate performance across available, ethically
collected and relevant subgroups.

Potential sources of performance variation include:

- Skin pigmentation
- Smartphone device
- Lighting conditions
- Gestational age
- Postnatal age
- Acquisition site

## 12. Explainability

GradCAM is planned for model interpretation.

Heatmaps will not be treated as proof that a model
uses clinically meaningful features.

## 13. Experiment Tracking

MLflow is planned for:

- Hyperparameters
- Metrics
- Artifacts
- Model versions
- Experiment comparison

## 14. Dataset Versioning

DVC is planned for dataset and pipeline versioning.

Patient-identifiable data must not be uploaded to
unapproved storage locations.

## 15. Deployment

Validated research models may later be packaged
for controlled inference experiments.

## 16. Current Limitations

No trained or clinically validated prediction model
is available at this stage.
