# NeoJaundice — Color Feature Engineering

## Phase
6.1

## Dataset
478 neonatal skin images.

## ROI
Geometric ROI baseline.

## Color Spaces
- RGB
- HSV
- YCrCb
- OpenCV Lab

## Statistics
- Mean
- Standard deviation
- Median
- 10th percentile
- 90th percentile

## Feature Count
60 color statistics plus ROI pixel count.

## Outputs
- color_features.csv
- color_features_summary.json

## Research Limitations
- No verified bilirubin labels.
- No supervised regression model trained.
- Color features are affected by illumination and camera processing.
- HSV hue is circular and should not be interpreted as an ordinary linear variable.
- OpenCV uint8 Lab values are encoded, not raw CIELAB coordinates.

## Next
Feature analysis and model-ready dataset preparation.
