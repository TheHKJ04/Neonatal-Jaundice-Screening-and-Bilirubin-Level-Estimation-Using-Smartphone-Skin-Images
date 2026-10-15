# NeoJaundice Image Exploratory Data Analysis

## Phase

Phase 3 — Task 3.1

## Objective

Characterize the technical properties of neonatal skin images
before preprocessing and machine learning experiments.

## Input

An approved dataset directory containing JPEG or PNG images.

## Outputs

- image_statistics.csv
- dataset_summary.json
- Six descriptive distribution plots

## Image-Level Metrics

- Width
- Height
- Aspect ratio
- Brightness
- Contrast
- Laplacian variance
- RGB channel means
- Mean HSV saturation
- Overexposed pixel fraction
- Underexposed pixel fraction

## Reproducibility

The pipeline uses deterministic image discovery and
fixed descriptive calculations.

No random sampling is performed.

## Clinical Limitations

This dataset has no verified numerical bilirubin labels.

Image EDA must not be interpreted as bilirubin estimation,
clinical screening performance, or model validation.

Image brightness, RGB means, and saturation are influenced
by illumination, device processing, and skin pigmentation.

## Validation

Run:

python -m pytest tests/unit/test_image_eda.py -v

## Future Work

- Body-region-specific EDA
- Image quality review
- Color calibration experiments
- ROI extraction
- Preprocessing comparison
- External dataset evaluation

## EDA Results — 2026-10-14

Dataset:
forehead_sternum_mendeley

Images analyzed: 478
Failures: 0

Image geometry:

- Width: 640 pixels
- Height: 480 pixels
- Aspect ratio: 4:3
- Dimension variation: None

Brightness:

- Mean: 206.30
- Standard deviation: 8.15
- Range: 178.04–236.10

Contrast:

- Mean: 11.95
- Range: 5.54–26.06

Laplacian variance:

- Mean: 87.65
- Median: 45.93
- Range: 2.96–553.99

RGB means:

- Red: 239.25
- Green: 196.65
- Blue: 169.45

Exposure:

- Mean grayscale overexposed fraction: 0.00363
- Maximum: 0.13346
- Mean underexposed fraction: 0

Identified engineering issue:

The original EDA grouped all images under source_archive,
because it assumed the first directory was the body region.

Action:

Introduce explicit anatomical region detection from
directory names.

Preprocessing recommendation:

Preserve RGB information and aspect ratio.
Do not apply aggressive color enhancement by default.

Clinical limitation:

No numerical TSB/TcB labels are available.
