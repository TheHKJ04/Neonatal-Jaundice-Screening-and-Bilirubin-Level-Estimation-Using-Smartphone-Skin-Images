# NeoJaundice — Preprocessing Quality Control

## Phase

Phase 4 — Task 4.2

## Goal

Validate image preprocessing outputs before ROI extraction
and machine learning model development.

## Input

Raw images and preprocessing configuration.

## Validations

### Tensor Integrity

- Expected tensor shape
- Float32 dtype
- Finite numerical values

### Mask Integrity

- Expected mask shape
- Float32 dtype
- Binary mask values
- Correct padding geometry

### Geometry

- Positive image dimensions
- Valid resized dimensions
- Padding consistency
- Target dimension consistency

### Color Integrity

- RGB preview dimensions
- RGB preview dtype
- Consistency between preview and normalized tensor

## Outputs

- qc_results.csv
- qc_summary.json
- Region-stratified preview images

## Failure Policy

Any image-level QC failure results in a failed batch status.

The CLI returns a nonzero exit code.

## Research Limitations

Passing technical QC does not establish:

- Clinical image quality
- Correct skin ROI selection
- Accurate bilirubin estimation
- Patient-level independence
- Generalization across devices and lighting conditions

## Next Step

Phase 5 — Forehead and Sternum ROI Extraction.
