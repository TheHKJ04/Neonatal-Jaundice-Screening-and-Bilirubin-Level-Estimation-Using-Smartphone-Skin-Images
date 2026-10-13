# NeoJaundice Dataset Source Evaluation

## Status

Preliminary dataset sourcing review.

No external clinical dataset has been ingested.

## Research Objective

Identify datasets suitable for:

1. Neonatal jaundice image classification.
2. Total serum bilirubin regression.
3. Cross-device and cross-population evaluation.

## Primary Candidates

### NeoJaundice Benchmark

Source:
https://springernature.figshare.com/articles/dataset/NeoJaundice_Neonatal_Jaundice_Evaluation_in_Demographic_Images/22302559

Reported:
- 745 infants
- 2,235 images
- Head, face and chest regions
- TSB-based labels described in publications

Pending:
- Download access
- License verification
- Numerical TSB label availability
- Patient identifier availability
- Metadata inspection

### Forehead and Sternum Dataset

Source:
https://data.mendeley.com/datasets/yfsz6c36vc/1

Reported:
- 300 infants
- Forehead and sternum images
- TSB and TcB measurements
- Smartphone-connected microscope
- CC BY 4.0 license listed

Pending:
- File-level metadata inspection
- TSB label mapping
- Patient identifier verification
- Device-domain assessment

### NJN Dataset

Source:
https://zenodo.org/records/7825810

Reported:
- Normal and jaundiced image classes
- Smartphone images
- RGB and YCrCb features

Pending:
- Actual image count
- Archive contents
- Patient identifier availability
- License verification

## Dataset Selection Criteria

A dataset will be evaluated using:

- Relevance to newborn skin images
- Availability of reference measurements
- Patient-level identifiers
- Acquisition timestamps
- Image quality
- Device metadata
- Population representation
- Licensing
- Privacy
- Reproducibility

## Regression Eligibility

Numerical TSB labels must be verified.

Classification labels must not be converted into
invented numerical bilirubin measurements.

## Dataset Acquisition Policy

Before ingestion:

1. Verify the original source.
2. Record dataset version.
3. Review license and usage restrictions.
4. Review privacy and consent information.
5. Inspect file structure.
6. Confirm label definitions.
7. Record provenance.
8. Preserve the original files.
9. Exclude clinical images from Git.

## Current Decision

Primary regression candidates:

- NeoJaundice benchmark
- Forehead and Sternum dataset

Primary classification candidate:

- NJN dataset

Final selection remains pending file-level verification.

## Next Step

Task 2.3: Dataset privacy, ethics and acquisition controls.
