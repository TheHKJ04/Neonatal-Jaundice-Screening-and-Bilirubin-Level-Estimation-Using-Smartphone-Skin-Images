# NeoJaundice Dataset Structure Analysis

## Status

Phase 2 — Task 2.7

## Objective

Inspect dataset file structures and identify candidate
metadata fields relevant to neonatal jaundice research.

## Supported Files

Image inventory:

- JPEG
- PNG

Metadata inspection:

- CSV headers

Metadata identification only:

- XLSX
- XLS

## Outputs

- structure_report.json
- structure_report.md

## Research Questions

1. Does the dataset contain neonatal images?
2. Are numerical bilirubin measurements available?
3. Are patient identifiers available?
4. Can images be linked to reference measurements?
5. Are reference measurement units documented?
6. Are acquisition and measurement timestamps available?
7. Is the dataset suitable for patient-level splitting?

## Important Limitations

Column-name matching is heuristic.

The presence of a column named TSB does not establish:

- Numerical validity
- Measurement units
- Clinical reference reliability
- Image-reference pairing
- Patient identity consistency

Excel files are detected but not parsed in this task.

## Privacy

File paths and column names may contain sensitive information.

Reports must be reviewed before sharing.

## Next Step

Dataset-specific metadata parsing, column mapping,
and regression eligibility assessment.
