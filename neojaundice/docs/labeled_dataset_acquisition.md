# NeoJaundice — Labeled Dataset Acquisition

## Phase

5.6A — Dataset Acquisition and Label Validation

## Objective

Acquire and validate neonatal skin image datasets
containing verified bilirubin reference measurements.

## Existing Dataset

Dataset size: 478 images

Regions:
- Forehead: 239
- Sternum: 239

Known limitations:
- No verified numerical bilirubin labels
- No verified patient identifiers
- No confirmed image-patient mappings
- No verified clinical jaundice labels

## Regression Dataset Requirements

Required:
- Image identifier
- Pseudonymous patient identifier
- Image path
- Anatomical region
- Bilirubin measurement
- Measurement unit
- Measurement modality

Preferred:
- Image acquisition timestamp
- Reference measurement timestamp
- Gestational age
- Postnatal age in hours
- Camera metadata
- Lighting metadata

## Dataset Acquisition Status

No additional labeled dataset has been verified.

## Candidate Dataset Register

For every candidate, record:

- Dataset title
- Original source
- Publication
- Download or access location
- License and usage restrictions
- Number of images
- Number of patients
- Available anatomical regions
- Label type
- Reference measurement modality
- Measurement units
- Image-label pairing quality
- Patient identifiers
- Access status
- Validation status

## Dataset Acceptance Criteria

A dataset is not approved for supervised regression
until image-to-measurement pairing is verified.

Structural CSV validation alone is insufficient.

## Classification Requirements

Classification labels must be independently verified
or derived using a documented, clinically justified
definition with all required clinical variables.

Arbitrary bilirubin thresholds must not be treated
as clinically meaningful severity categories.

## Privacy and Ethics

- Do not commit patient data to Git.
- Do not publish identifiable neonatal images.
- Verify dataset licensing.
- Record provenance.
- Obtain required permissions.
- Preserve pseudonymous patient identifiers.
- Prevent patient-level train/test leakage.

## Next Tasks

1. Identify candidate labeled datasets.
2. Verify public access or request permissions.
3. Review dataset documentation.
4. Map source metadata into the standard schema.
5. Run structural validation.
6. Verify measurement provenance.
7. Establish patient-level splitting strategy.
