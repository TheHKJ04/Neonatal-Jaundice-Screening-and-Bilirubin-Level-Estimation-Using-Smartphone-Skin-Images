# NeoJaundice — Geometric ROI Extraction

## Phase

Phase 5 — Task 5.1

## Objective

Develop a reproducible geometric ROI baseline for
forehead and sternum neonatal skin images.

## Input

- Preprocessed RGB image
- Valid-pixel mask
- Preprocessing geometry metadata
- Anatomical region identifier

## Method

Region-specific rectangular ROIs are defined using
normalized coordinates relative to resized image
content.

Coordinates are mapped into the padded image canvas.

A binary rectangle mask is intersected with the
preprocessing valid-pixel mask.

## Outputs

- ROI binary mask
- ROI pixel coordinates
- ROI pixel count
- Valid image pixel count
- ROI coverage
- ROI visualization

## Baseline Configurations

### Forehead

x_min: 0.30
y_min: 0.25
x_max: 0.70
y_max: 0.60

### Sternum

x_min: 0.30
y_min: 0.30
x_max: 0.70
y_max: 0.70

## Important Limitations

The ROI is geometric and does not guarantee skin-only
selection.

The current coordinates are experimental defaults.

No clinical validation of ROI correctness has been
performed.

ROI coverage is a geometric measurement, not a
clinical quality metric.

The dataset does not contain verified numerical
bilirubin reference labels.

## Planned Evaluation

1. Review representative ROI overlays.
2. Measure geometric coverage.
3. Evaluate non-skin contamination.
4. Compare region-specific ROI configurations.
5. Experiment with skin segmentation.
6. Compare downstream color features.

## Next Step

Phase 5 — Task 5.2: ROI Quality Validation.
