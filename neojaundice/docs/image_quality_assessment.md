# NeoJaundice — ROI Image Quality Assessment

## Phase

5.6B — Image Quality Assessment

## Objective

Evaluate brightness, contrast, sharpness, exposure,
and RGB statistics within geometric ROIs.

## Dataset

478 neonatal skin images.

Regions:
- Forehead: 239
- Sternum: 239

## ROI Method

Geometric ROI baseline.

## Metrics

- Mean grayscale brightness
- Grayscale brightness standard deviation
- Grayscale contrast
- Laplacian variance
- Underexposed pixel fraction
- Overexposed pixel fraction
- RGB channel means
- RGB channel standard deviations
- ROI pixel count

## Warning Policy

Thresholds are exploratory.

Warnings prioritize images for visual review.

No image is automatically rejected.

## Limitations

Laplacian variance is not a definitive blur measure.

Brightness depends on image acquisition conditions.

RGB statistics are affected by camera processing,
white balance, illumination, and skin appearance.

No clinical image-quality validation is claimed.

## Expected Outputs

- image_quality_manifest.csv
- image_quality_summary.json
- quality_review_candidates.csv

## Next Step

Inspect flagged images and evaluate ROI suitability
before Phase 6 color feature engineering.
