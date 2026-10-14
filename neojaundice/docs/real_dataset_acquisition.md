# NeoJaundice Real Dataset Acquisition

## Status

Phase 2 — Task 2.6

Controlled acquisition workflow implemented.

Real dataset acquisition remains subject to
source, licensing, privacy, and permission verification.

## Objective

Acquire suitable neonatal jaundice research datasets
through a documented and reproducible workflow.

## Primary Candidates

- NeoJaundice benchmark
- Forehead and Sternum dataset
- NJN classification dataset

## Acquisition Requirements

Before acquisition:

1. Confirm the original dataset source.
2. Confirm the dataset version.
3. Review the license.
4. Review privacy and consent requirements.
5. Confirm intended-use permission.
6. Record the approval decision.
7. Obtain the archive from the approved source.
8. Verify SHA-256 against a trusted reference.
9. Inspect archive contents.
10. Extract into controlled storage.

## Storage

Downloaded archives:

data/external/<dataset_id>/

Extracted source files:

data/raw/<dataset_id>/

Inspection reports:

data/interim/<dataset_id>/

## Security

- Do not commit clinical images.
- Do not commit private approval records.
- Do not bypass access restrictions.
- Do not treat a self-computed checksum as proof of authenticity.
- Do not automatically trust downloaded archives.
- Avoid publishing identifying filenames or metadata.

## Current Limitations

The workflow does not provide:

- Automatic legal permission verification
- Automated clinical consent review
- Malware scanning
- Enterprise-grade access controls
- Encrypted clinical storage
- Full protection against hostile filesystem races

## Next Step

Inspect the acquired dataset's file structure,
image inventory, and available clinical labels.
