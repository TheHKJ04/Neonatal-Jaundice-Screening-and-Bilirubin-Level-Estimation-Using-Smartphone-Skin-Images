# NeoJaundice Clinical Metadata Integration

## Purpose

Integrate technical image metadata with clinical
reference measurements.

## Inputs

1. Image inspection manifest
2. Clinical metadata CSV

## Outputs

Standardized dataset manifest.

## Required Clinical Fields

- sample_id
- patient_id
- image_path
- captured_at
- bilirubin_measured_at
- bilirubin_type
- bilirubin_mg_dl
- age_hours
- gestational_age_weeks
- body_region

## Validation

The pipeline checks:

- Required columns
- Pydantic metadata constraints
- Sample ID uniqueness
- Image path matching
- Reference measurement type
- Bilirubin value validity
- Timestamp formatting

## Important Limitations

A technically valid record is not necessarily
clinically eligible for model training.

The current pipeline does not:

- Validate clinical pairing windows
- Verify patient identities
- Perform unit conversion
- Establish treatment thresholds
- Detect near-duplicate images
- Determine clinical label reliability

## Research Policy

TSB and TcB measurements must remain distinguishable.

Only records with verified TSB references should enter
the primary TSB regression training dataset.

## Synthetic Data

Synthetic clinical metadata is used only for
engineering tests.

Synthetic bilirubin values are not clinical evidence.

## Next Steps

- Real dataset acquisition
- Dataset-specific column mapping
- Metadata quality reporting
- Patient-level leakage prevention
