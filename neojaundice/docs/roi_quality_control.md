# NeoJaundice — ROI Quality Control

## Phase

Phase 5 — Task 5.2

## Objective

Validate geometric ROI extraction integrity and prepare
a reproducible human visual review.

## Input

- Raw neonatal forehead and sternum images
- Preprocessing configuration
- Geometric ROI configuration
- ROI QC configuration

## Technical Validation

The pipeline validates:

1. ROI mask dimensions
2. Valid-pixel mask dimensions
3. Mask data types
4. Finite numerical values
5. Binary mask values
6. Padding exclusion
7. ROI pixel counts
8. Valid-pixel counts
9. Bounding-box geometry
10. ROI coverage consistency

## Review Sampling

Method: Region-stratified random sampling

Random seed: 42

Requested sample size:
- Forehead: 20 images
- Sternum: 20 images

Total requested: 40 images

This is an initial pilot review sample.

It is not a statistically justified clinical validation sample.

## Review Categories

- Suitable
- Hair or eyebrow contamination
- Clothing contamination
- Background contamination
- Strong shadow or reflection
- Uncertain

## Outputs

- qc_results.csv
- qc_summary.json
- review_manifest.csv
- Region-stratified ROI overlays

## Technical QC vs Visual Review

Technical QC verifies software and geometric consistency.

Visual review assesses whether the selected area
appears suitable for subsequent skin-color analysis.

Passing technical QC does not imply visual suitability.

## Research Limitations

- No expert-annotated ROI ground truth
- No verified numerical bilirubin labels
- No verified patient identifiers
- No verified forehead/sternum image pairing
- No clinical validation
- No demonstrated diagnostic performance

## Future Evaluation

Potential future metrics include:

- ROI suitability rate
- Non-skin contamination rate
- Reviewer agreement
- ROI coverage distribution
- Segmentation overlap against expert annotations

If multiple reviewers are available, inter-rater
agreement can be assessed.

## Reproducibility

Record:
- Dataset version
- ROI configuration version
- Preprocessing configuration version
- Random seed
- Source code commit
- Software environment

## Next Step

Phase 5 — Task 5.3:
OpenCV-based skin segmentation experiments.
