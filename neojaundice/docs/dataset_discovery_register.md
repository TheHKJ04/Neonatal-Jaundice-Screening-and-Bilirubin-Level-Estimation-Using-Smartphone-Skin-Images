# NeoJaundice — Dataset Discovery Register

## Research Date

2026-10-16

## Objective

Identify neonatal skin image datasets suitable for
classification and numerical bilirubin regression.

## Candidate 1: NeoJaundice

Source:
Springer Nature Figshare

Dataset ID:
22302559

Reported images:
2235

Reported infants:
745

Classification:
Binary labels available according to dataset description.

Regression:
Numerical TSB label availability not yet verified.

Priority:
High for classification.

Status:
Metadata reviewed; archive not yet inspected.

## Candidate 2: NJN

Source:
Zenodo

Record:
7825810

Reported images:
670

Note:
Published class counts are internally inconsistent.

Classification:
Normal versus jaundiced.

Regression:
No verified numerical bilirubin labels.

Priority:
High for classification experiments.

Status:
Metadata reviewed; archive not yet inspected.

## Candidate 3: Mendeley Forehead/Sternum

Source:
Mendeley Data

Dataset ID:
yfsz6c36vc

Local images:
478

TSB and TcB:
Described in the published dataset metadata.

Regression:
Individual image-to-measurement mappings
not verified in local files.

Priority:
Highest for further regression investigation.

Status:
Local metadata audit pending.

## Candidate 4: Sternal Skin Regression Examples

Source:
GitHub repository mj129/Surveillance-of-neonatal-jaundice

Available examples:
12 images from 4 newborns.

Regression:
Labeled examples reported.

Priority:
Pipeline smoke testing only.

Status:
Label schema and files not yet inspected.

## Candidate 5: Digital Skin Images and TSB

Source:
Mendeley Data

Dataset ID:
v5gdzmssf3

Access:
Embargoed until 2028-01-01.

Priority:
Future research.

Status:
Not currently available for acquisition.

## Dataset Acceptance Requirements

Before supervised training:

1. Verify image-to-label mappings.
2. Verify numerical measurement provenance.
3. Verify measurement modality and units.
4. Verify permitted research use.
5. Identify patients for leakage-safe splitting.
6. Document acquisition and label limitations.
7. Run structural validation.
8. Record dataset version and integrity hashes.

## Research Decision

Prioritize NeoJaundice for classification.

Reinvestigate the original Mendeley dataset for
numerical bilirubin labels.

Do not train supervised regression using invented,
synthetic, or unverified bilirubin measurements.

## Next Actions

1. Audit existing Mendeley files.
2. Inspect original Mendeley download contents.
3. Inspect NeoJaundice archive metadata.
4. Inspect the 12-image regression example.
5. Decide whether author contact is required.

## Local Mendeley Dataset Audit

Audit date: 2026-10-16

Total image files: 478
Non-image files: 0

No CSV, Excel, or other metadata files were found
in the local extracted dataset.

No filenames matching bilirubin, clinical, patient,
label, metadata, TSB, or TcB were found.

Conclusion:
The local dataset does not provide verified numerical
bilirubin labels or image-to-measurement mappings.

Approved uses:
- Image preprocessing
- ROI extraction
- Image quality analysis
- Unsupervised color feature analysis

Not approved for:
- Supervised bilirubin regression
- Supervised jaundice classification

Next action:
Investigate externally available labeled datasets
and contact the original authors if necessary.
