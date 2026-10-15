# NeoJaundice — Experimental Skin Segmentation

## Phase

Phase 5 — Task 5.3

## Objective

Compare two color-based candidate skin segmentation
methods within geometric forehead and sternum ROIs.

## Dataset

478 neonatal skin images:
- 239 forehead images
- 239 sternum images

## Methods

### Geometric Baseline

A region-specific rectangular ROI is defined using
normalized coordinates.

### YCrCb Thresholding

RGB images are converted into OpenCV YCrCb space.

Candidate pixels are selected using configurable
channel thresholds.

### HSV Thresholding

RGB images are converted into OpenCV HSV space.

Candidate pixels are selected using configurable
channel thresholds.

## Final Mask

The final candidate mask is the intersection of:

1. Geometric ROI mask
2. Color-threshold mask

## Technical Validation

Checks include:

- Mask dimensions
- Binary values
- Finite values
- Geometric ROI containment
- Pixel-count consistency
- Coverage consistency

## Coverage Metric

Coverage is defined as:

selected candidate pixels / geometric ROI pixels

Coverage is not segmentation accuracy.

## Outputs

- segmentation_manifest.csv
- segmentation_summary.json
- Region-specific comparison previews

## Important Limitations

The methods use fixed color thresholds.

Their performance may vary with:
- Skin pigmentation
- Illumination
- Smartphone camera characteristics
- White balance
- Shadows
- Reflections

No annotated skin segmentation masks are available.

Therefore, Dice score and IoU against ground truth
cannot currently be established.

No verified numerical bilirubin labels are available.

The pipeline is for research only and is not a
validated medical device.

## Planned Experiments

1. Compare YCrCb and HSV coverage.
2. Review representative segmentation overlays.
3. Evaluate failure patterns by anatomical region.
4. Examine sensitivity to threshold changes.
5. Compare candidate masks with geometric ROI.
6. Investigate skin-tone and lighting robustness.

## Reproducibility

Record:
- Dataset version
- Configuration files
- Source code commit
- Random seeds where applicable
- Python environment
- Library versions

## Next Step

Phase 5.4 — ROI Method Comparison and Evaluation.
