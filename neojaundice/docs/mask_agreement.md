# NeoJaundice — Pixel-Level Mask Agreement

## Phase

Phase 5 — Task 5.5

## Objective

Evaluate pixel-level agreement between YCrCb and HSV
candidate segmentation masks and investigate sensitivity
to configurable color thresholds.

## Dataset

478 neonatal skin images:

- 239 forehead
- 239 sternum

## Experiment A: Pixel-Level Agreement

Metrics:

- Intersection over Union
- Dice coefficient
- Pixel disagreement rate
- Intersection pixel count
- Union pixel count
- YCrCb-only pixels
- HSV-only pixels
- Pixels rejected by both methods

These metrics measure agreement between algorithms.

They do not measure segmentation accuracy.

## Experiment B: Threshold Sensitivity

Configurations:

1. Baseline
2. Moderate
3. Restrictive

The baseline reuses the original Phase 5.3 thresholds.

The alternative configurations are exploratory.

They are not clinically validated.

## Empty-Mask Handling

When both candidate masks are empty:

- IoU is undefined
- Dice is undefined
- Both-empty status is recorded

Undefined values are represented as null in JSON and
missing values in CSV.

## Evaluation Design

Every image is evaluated using every configuration.

Expected:

478 images x 3 configurations = 1434 evaluations.

Each evaluation generates two candidate masks.

## Visualization

Agreement overlays use:

- Green: selected by both methods
- Red: YCrCb only
- Blue: HSV only
- Original RGB: rejected by both

## Limitations

No expert-annotated skin masks are available.

No verified numerical bilirubin labels are available.

Patient identifiers are not verified.

Fixed thresholds may behave differently across:

- Skin pigmentation
- Lighting conditions
- Smartphone cameras
- White balance
- Shadows
- Reflections

Higher agreement does not imply higher accuracy.

Lower coverage does not imply better segmentation.

## Reproducibility

Record:

- Dataset version
- Preprocessing configuration
- Geometric ROI configuration
- Threshold configuration
- Git commit
- Python version
- Library versions

## Next Research Step

Evaluate segmentation suitability and decide whether
to retain color-based segmentation before feature
engineering.

## Clinical Scope

Research use only.

No diagnostic performance has been established.
