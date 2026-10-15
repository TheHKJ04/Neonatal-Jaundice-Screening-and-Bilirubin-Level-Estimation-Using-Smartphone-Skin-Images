# Region-Specific Image Exploratory Data Analysis

## Project

NeoJaundice — Neonatal Jaundice Screening and
Bilirubin Level Estimation Using Smartphone Skin Images.

## Phase

Phase 3 — Task 3.3

## Objective

Compare technical image characteristics between
forehead and sternum images.

## Dataset

Dataset ID: forehead_sternum_mendeley

Verified technical inventory:

- Total images: 478
- Forehead images: 239
- Sternum images: 239
- Invalid images: 0
- Exact duplicates: 0

## Input

data/interim/forehead_sternum_mendeley/eda/image_statistics.csv

## Metrics

- Brightness
- Contrast
- Laplacian variance
- Mean red intensity
- Mean green intensity
- Mean blue intensity
- Mean HSV saturation
- Grayscale overexposure fraction
- Grayscale underexposure fraction

## Descriptive Statistics

For each region and metric:

- Mean
- Median
- Population standard deviation
- Minimum
- Maximum
- 25th percentile
- 75th percentile
- Interquartile range

## Comparison Definition

Mean difference = Forehead mean - Sternum mean

A positive difference indicates a higher forehead mean.

A negative difference indicates a higher sternum mean.

## Outputs

- region_comparison.csv
- region_comparison.json
- Six comparison plots

## Reproducibility

The pipeline consumes an existing validated image-statistics
manifest and performs deterministic calculations.

No random sampling is used.

## Research Limitations

No numerical TSB/TcB measurements are available.

No verified patient-level pairing is available.

The analysis is descriptive and does not establish
clinical predictive performance.

Observed differences may reflect lighting, camera
processing, skin appearance, or acquisition conditions.

## Next Steps

- Interpret actual region-specific statistics
- Investigate color-channel clipping
- Design preprocessing experiments
- Develop ROI extraction
- Obtain verified labeled bilirubin data
