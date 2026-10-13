# NeoJaundice Dataset Specification

## Status

Version: 0.1.0

Stage: Preliminary research design

Clinical validation: Not established

## 1. Purpose

Define metadata requirements for research into
smartphone image-based neonatal bilirubin estimation.

## 2. Primary Research Target

Total serum bilirubin (TSB), expressed in mg/dL.

TcB records may be retained for separate analyses
but must not be silently treated as TSB labels.

## 3. Sample Definition

A sample consists of:

- One smartphone image
- A pseudonymized patient identifier
- An image acquisition timestamp
- A reference bilirubin measurement
- A reference measurement timestamp
- Relevant acquisition and clinical metadata

## 4. Identifiers

### sample_id

Unique identifier for an image sample.

### patient_id

Pseudonymized identifier used to group samples
belonging to the same newborn.

Direct patient identifiers must not be included.

## 5. Essential Metadata

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

## 6. Recommended Metadata

- Smartphone model
- Lighting conditions
- Acquisition site
- Reference measurement method
- Image dimensions
- Phototherapy status
- Documented pigmentation descriptor

## 7. Image Requirements

Candidate formats:

- JPEG
- PNG

Images must be validated before preprocessing.

The schema does not establish image authenticity
or image quality.

## 8. Timestamp Requirements

Timestamps must include timezone information.

The time difference between image acquisition
and reference measurement must be recorded.

A clinically justified pairing interval has not
yet been selected.

## 9. Dataset Splitting

Use patient-level separation.

No patient may appear in more than one of:

- Training
- Validation
- Test

The preliminary split proportions are:

- Training: 70%
- Validation: 15%
- Test: 15%

These proportions may change depending on
dataset size, study design, and external validation.

## 10. Data Quality

Planned checks:

- Missing values
- Duplicate sample IDs
- Duplicate images
- Patient identifier consistency
- Invalid reference measurements
- Timestamp inconsistencies
- Invalid image files
- Image quality
- Metadata completeness

## 11. Privacy

Only appropriately authorized data may be used.

Store patient-identifiable data outside the
public repository.

## 12. Research Limitations

This specification does not establish:

- Clinical effectiveness
- A validated image acquisition protocol
- An acceptable image-reference pairing window
- A clinically validated screening threshold
- Model generalizability

## 13. Next Steps

1. Identify suitable datasets.
2. Review licenses and permissions.
3. Evaluate reference measurement availability.
4. Finalize acquisition and inclusion criteria.
5. Implement ingestion and validation pipelines.
