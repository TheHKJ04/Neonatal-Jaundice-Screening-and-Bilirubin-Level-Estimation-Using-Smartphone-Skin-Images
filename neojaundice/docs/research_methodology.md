# NeoJaundice Research Methodology

## Working Research Title

Neonatal Jaundice Screening and Bilirubin Level
Estimation Using Smartphone Skin Images

## 1. Research Problem

Neonatal hyperbilirubinemia requires timely assessment.

This research investigates the feasibility and limitations
of using smartphone skin images as an input to machine
learning methods for jaundice-related prediction.

## 2. Research Questions

RQ1: Can image-derived features predict reference
bilirubin measurements with useful accuracy?

RQ2: How do traditional machine learning models
compare with deep learning approaches?

RQ3: How sensitive are predictions to lighting,
camera characteristics, and image quality?

RQ4: How does performance vary across relevant
patient and acquisition subgroups?

RQ5: What are the limitations of image-only
bilirubin estimation?

## 3. Study Design

The final study design will depend on:

- Dataset availability
- Ethics approval
- Data-use permissions
- Reference measurement quality
- Sample size
- Acquisition protocol

The protocol is not yet finalized.

## 4. Ground Truth

Reference bilirubin measurements must be clearly
defined and documented.

The dataset must record the type and timing of
reference measurements.

Measurements taken at different times may not be
directly interchangeable.

## 5. Data Partitioning

Prevent patient-level leakage.

Where data permits, investigate external validation
across devices, sites, or populations.

## 6. Experimental Comparisons

Candidate experiments:

E1: Color-feature regression baseline

E2: Traditional machine learning baseline

E3: EfficientNet image model

E4: ROI-based image model

E5: Image and metadata fusion model

These are proposed experiments.

## 7. Evaluation

Report:

- Sample size
- Patient count
- Data partitioning
- Missing data
- Reference measurement distribution
- MAE and RMSE
- Prediction bias
- Confidence intervals
- Subgroup performance
- Failure cases

## 8. Statistical Analysis

Statistical procedures and uncertainty estimation
will be selected before final evaluation.

The held-out test set must not be repeatedly used
for hyperparameter tuning.

## 9. Reproducibility

Record:

- Dataset version
- Git commit
- Model version
- Configuration
- Random seed
- Package versions
- Hardware
- Training duration

## 10. Ethical Considerations

The research must address:

- Appropriate consent
- Data-use permissions
- Patient privacy
- Data minimization
- Secure storage
- Bias and representativeness
- Clinical risk
- Responsible reporting

## 11. Limitations

Potential limitations include:

- Device-specific color responses
- Lighting variability
- Skin pigmentation effects
- Limited dataset diversity
- Label timing mismatch
- Selection bias
- Dataset shift
- Poor external generalization

## 12. Publication Plan

Potential paper sections:

1. Abstract
2. Introduction
3. Related Work
4. Materials and Methods
5. Experimental Setup
6. Results
7. Discussion
8. Limitations
9. Conclusion
10. References

## 13. Current Status

This document is a preliminary methodology plan.

No experimental results or clinical performance
claims are available yet.
