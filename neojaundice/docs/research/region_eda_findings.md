# Region-Specific Image EDA Findings

## Project

NeoJaundice: Neonatal Jaundice Screening and
Bilirubin Level Estimation Using Smartphone Skin Images

## Dataset

forehead_sternum_mendeley

## Validation

- Total images: 478
- Forehead images: 239
- Sternum images: 239
- Failed images: 0
- Automated tests: 164 passed

## Descriptive Results

| Metric | Forehead Mean | Sternum Mean |
|---|---:|---:|
| Brightness | 207.0491 | 205.5493 |
| Contrast | 15.4315 | 8.4772 |
| Laplacian variance | 146.7341 | 28.5744 |
| Red intensity | 238.2420 | 240.2489 |
| Green intensity | 197.9143 | 195.3834 |
| Blue intensity | 172.1592 | 166.7341 |
| HSV saturation | 71.3030 | 78.1786 |
| Overexposed fraction | 0.005027 | 0.002226 |
| Underexposed fraction | 0 | 0 |

## Findings

1. Forehead images exhibit higher mean grayscale contrast.
2. Forehead images exhibit higher mean Laplacian variance.
3. Mean brightness is relatively similar between regions.
4. RGB channel distributions differ between regions.
5. Sternum images exhibit higher mean HSV saturation.
6. Grayscale underexposure was not detected using the
   current threshold-based metric.

## Interpretation Limitations

Laplacian variance is not a direct measurement of
clinical image quality.

Differences in image statistics may reflect anatomy,
lighting, framing, device processing, or acquisition
conditions.

No patient-level pairing has been verified.

No numerical TSB or TcB labels are available.

No clinical conclusions can be drawn from these
descriptive image statistics alone.

## Preprocessing Decisions

Initial baseline:

- Preserve RGB values.
- Preserve aspect ratio during resizing.
- Use explicit padding.
- Track valid image pixels separately from padding.
- Disable automatic color enhancement.
- Disable arbitrary blur rejection.
- Use configurable model normalization.

## Planned Experiments

P0: RGB + aspect-ratio-preserving resize + padding.

P1: ROI extraction before resizing.

P2: Controlled illumination normalization.

P3: Region-specific preprocessing.

These experiments require suitable labeled data
before clinical predictive performance can be assessed.

## Research Status

Image EDA completed.

Supervised bilirubin regression remains blocked
by unavailable numerical reference labels.
