# NeoJaundice Dataset Inspection Pipeline

## Status

Phase 2 — Dataset Engineering

Task 2.4: Image Manifest and Metadata Inspection

## 1. Purpose

The dataset inspection pipeline validates the technical
properties of image files before they enter downstream
machine learning workflows.

## 2. Supported Formats

- JPEG
- PNG

Other image formats are not included in the current
inspection inventory.

## 3. Inspection Operations

The pipeline performs:

1. Recursive image discovery
2. Extension filtering
3. Image decoding
4. Dimension extraction
5. Color-mode extraction
6. File-size inspection
7. SHA-256 calculation
8. Exact duplicate identification
9. Manifest generation
10. Summary generation

## 4. Image Manifest

Output:

image_manifest.csv

Fields:

- relative_path
- status
- image_format
- width
- height
- color_mode
- file_size_bytes
- sha256
- duplicate_of
- error_code

## 5. Inspection Summary

Output:

inspection_summary.json

Metrics:

- Total images
- Valid images
- Invalid images
- Exact duplicates
- Unique valid images
- Image format distribution
- Color mode distribution
- Error distribution

## 6. Dataset Safety

The inspector does not modify source images.

Symbolic-link directories are not followed.

The implementation imposes file-size and pixel-count limits.

The current implementation is intended for controlled
research environments.

## 7. Privacy

Image filenames and relative paths may contain
identifying information.

Manifest files must be handled according to the
dataset's privacy and access requirements.

Do not publish manifests containing sensitive filenames.

## 8. Limitations

The inspector does not:

- Identify newborns
- Extract patient identifiers
- Validate bilirubin measurements
- Verify image-reference pairing
- Detect near-duplicate images
- Perform skin segmentation
- Correct lighting conditions
- Determine clinical image suitability

## 9. Reproducibility

The image manifest provides a file inventory and
SHA-256 fingerprints.

The dataset version, source, approval record, and
processing code version must also be recorded.

## 10. Example Command

python -m scripts.inspect_dataset \
    --input data/raw/synthetic_demo \
    --output data/interim/synthetic_demo_inspection

## 11. Next Steps

- Clinical metadata mapping
- Patient identifier validation
- Bilirubin reference validation
- Duplicate and near-duplicate analysis
- Patient-level dataset splitting
- Dataset quality reporting
