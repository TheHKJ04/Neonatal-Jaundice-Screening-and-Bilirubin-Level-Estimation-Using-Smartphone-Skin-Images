# NeoJaundice — Project Handoff Context

## Project

Neonatal Jaundice Screening and Bilirubin Level
Estimation Using Smartphone Skin Images.

## Assistant Role

Act as Senior AI Engineer, Computer Vision Researcher,
MLOps Engineer, Software Architect, FastAPI Developer,
React Developer, and Technical Mentor.

Explain concepts before implementation.

Provide:
- Architecture decisions
- Exact file paths
- Complete code
- Windows PowerShell commands
- Automated tests
- Validation steps
- Debugging guidance
- Documentation updates
- Git commit instructions
- Research limitations
- Interview preparation

## Environment

OS: Windows
Shell: PowerShell

Repository:
D:\OneDrive\Desktop\CV Project\neojaundice

Branch:
develop

## Last Verified State

Commit:
7390664

Tests:
312 passed

Completed:
- Phase 1: Environment setup
- Phase 2: Dataset ingestion
- Phase 3: EDA
- Phase 4: Image preprocessing
- Phase 4.2: Preprocessing QC
- Phase 5.1: Geometric ROI extraction
- Phase 5.2: ROI quality control
- Phase 5.3: YCrCb and HSV segmentation
- Phase 5.4: ROI method comparison
- Phase 5.5: Pixel-level agreement and sensitivity

## Dataset

478 neonatal skin images.

Forehead: 239
Sternum: 239

No verified numerical bilirubin labels.

No verified patient identifiers.

No expert-annotated segmentation masks.

## Important Experimental Findings

Baseline coverage:
YCrCb: 0.998520628259341
HSV: 0.9999436835935707

Baseline mean IoU agreement:
0.9985317301684482

Moderate mean IoU:
0.8061877588627698

Restrictive mean IoU:
0.41910558830324224

Restrictive HSV empty masks:
33

## Current Engineering Decision

Use geometric ROI as the provisional baseline
for future feature engineering.

Keep color segmentation as experimental alternatives.

No segmentation method has established
clinical or anatomical accuracy.

## Supervised Learning Objectives

Primary:
Bilirubin regression using verified paired
reference measurements.

Secondary:
Jaundice classification using verified labels.

Both are currently blocked by missing labels.

## Current Phase

Phase 5.6A:
Labeled dataset acquisition and validation.

Status:
Implementation in progress.

## Next Phase

Phase 5.6B:
ROI image-quality assessment.

## Research Constraints

Do not claim:
- Clinical diagnostic accuracy
- Validated bilirubin estimation
- Patient-level independence
- Segmentation ground-truth accuracy

Until supported by appropriate data and evaluation.

## Handoff Instructions

When continuing in a new chat:

1. Read this document.
2. Read PROJECT_STATUS.md.
3. Read PROJECT_BLUEPRINT.md.
4. Read ARCHITECTURE.md.
5. Review the latest Git commit.
6. Ask for the latest test results if needed.
7. Continue from the current phase.

Do not restart completed modules unnecessarily.

Update this file after every validated phase.
