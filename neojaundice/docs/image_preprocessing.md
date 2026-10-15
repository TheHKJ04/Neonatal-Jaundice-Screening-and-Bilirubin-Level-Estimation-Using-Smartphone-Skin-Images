# NeoJaundice Image Preprocessing Pipeline

## Phase

Phase 4 — Task 4.1

## Objective

Build a reproducible image preprocessing pipeline
for model training and inference.

## Input

Neonatal skin images in JPEG or PNG format.

## Pipeline

1. Validate image path.
2. Decode image using Pillow.
3. Apply EXIF orientation.
4. Convert to RGB.
5. Resize while preserving aspect ratio.
6. Add centered padding.
7. Generate valid-pixel mask.
8. Convert to float32.
9. Apply configurable normalization.
10. Return image tensor, mask, preview, and metadata.

## Baseline Configuration

Target size: 224 x 224

Interpolation: Bilinear

Padding: RGB (0, 0, 0)

Normalization: ImageNet

Maximum input pixels: 25,000,000

## Example

Original image: 640 x 480

Resized image: 224 x 168

Padding:
- Top: 28
- Bottom: 28
- Left: 0
- Right: 0

Output tensor: [3, 224, 224]

Valid mask: [1, 224, 224]

## Design Principles

- Original images remain unchanged.
- Training and inference share preprocessing logic.
- Geometry is recorded.
- Padding pixels are identified separately.
- Color enhancement is disabled by default.
- Configuration is externalized.
- Unit tests validate behavior.

## Research Considerations

ImageNet normalization is appropriate only when it
matches the expected preprocessing of the selected
model weights.

Padding pixels must not be treated as skin pixels.

Color calibration and illumination normalization
require separate experimental evaluation.

## Dataset Limitation

The current dataset has no verified numerical
bilirubin reference measurements.

Preprocessing validation does not establish
clinical model performance.

## Next Steps

- Validate real dataset preprocessing.
- Review generated previews.
- Develop ROI extraction.
- Integrate preprocessing with PyTorch Dataset.
- Track experiments with MLflow.
