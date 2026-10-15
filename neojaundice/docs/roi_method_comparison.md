# NeoJaundice — ROI Method Comparison

## Phase

Phase 5 — Task 5.4

## Objective

Quantitatively compare experimental YCrCb and HSV
segmentation coverage within geometric ROIs.

## Dataset

478 neonatal skin images:
- 239 forehead images
- 239 sternum images

## Methods

1. Geometric ROI baseline
2. YCrCb thresholding
3. HSV thresholding

The geometric ROI defines the denominator for coverage.

## Primary Metric

Coverage:

Selected segmentation pixels / geometric ROI pixels

## Statistical Analysis

For each region and method:

- Mean coverage
- Median coverage
- Standard deviation
- Minimum and maximum
- First and third quartiles
- Image-level bootstrap confidence interval

## Paired Comparison

For each image:

Difference = HSV coverage - YCrCb coverage

Additional metrics:

- Mean paired difference
- Mean absolute difference
- Number of large disagreements
- Near-full coverage counts

## Bootstrap Configuration

- Iterations: 2000
- Confidence level: 95%
- Random seed: 42

## Interpretation

Coverage describes the proportion of the geometric ROI
retained by a segmentation method.

Coverage does not measure segmentation accuracy.

Similar coverage values do not imply pixel-level mask
agreement.

## Research Limitations

- No expert-annotated skin masks
- No verified patient identifiers
- No confirmed patient-level independence
- No verified numerical bilirubin labels
- No established clinical validity

Bootstrap intervals are descriptive at the image level.

They must not be interpreted as patient-level clinical
confidence intervals.

## Reproducibility

Record:

- Dataset version
- Segmentation configuration
- Evaluation configuration
- Source code commit
- Library versions
- Random seed

## Next Research Tasks

1. Review images with large method disagreements.
2. Compare actual pixel-level mask overlap.
3. Perform controlled threshold-sensitivity experiments.
4. Evaluate segmentation suitability using reviewed images.
5. Investigate robustness to lighting and skin pigmentation.

## Clinical Scope

Research use only.

No diagnostic performance or bilirubin estimation
performance has been established.
