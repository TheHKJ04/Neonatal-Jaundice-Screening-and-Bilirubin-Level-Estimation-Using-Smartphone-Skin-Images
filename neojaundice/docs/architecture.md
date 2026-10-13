# NeoJaundice System Architecture

## 1. Overview

NeoJaundice is designed as a modular computer vision
research platform with a future web application.

The architecture separates:

- Presentation
- API transport
- Application services
- Domain logic
- Data access
- Machine learning
- Infrastructure

## 2. Architecture Status

### Currently Implemented

- Repository structure
- Application configuration
- ML configuration templates
- Development container
- PostgreSQL development service
- Testing and code-quality infrastructure

### Planned

- Image upload API
- Image preprocessing pipeline
- Skin ROI extraction
- Feature engineering
- Model training
- Model inference
- Bilirubin regression
- Explainability
- React interface
- Database persistence
- Model monitoring

## 3. Proposed System Diagram

```text
                   React Frontend
                         |
                         v
                    FastAPI API
                         |
                Application Services
                         |
              +----------+----------+
              |                     |
              v                     v
        ML Inference           Data Repository
              |                     |
              v                     v
       Preprocessing             PostgreSQL
              |
              v
         ROI Extraction
              |
              v
         Model Inference
              |
              v
       Prediction Response
```

## 4. Clean Architecture

The backend will separate concerns into layers.

### API Layer

Handles HTTP requests, responses, and validation.

### Service Layer

Coordinates application use cases.

### Domain Layer

Defines business rules and domain concepts.

### Repository Layer

Abstracts persistence operations.

### Infrastructure Layer

Implements database, storage, and external integrations.

## 5. ML Architecture

The proposed ML system includes:

1. Dataset ingestion
2. Data validation
3. Exploratory analysis
4. Image quality assessment
5. Preprocessing
6. Skin ROI extraction
7. Feature engineering
8. Baseline modeling
9. EfficientNet experiments
10. Bilirubin regression
11. Explainability
12. Evaluation
13. Model packaging
14. Inference

## 6. Dependency Direction

High-level application logic should not depend directly
on database or web framework implementation details.

Interfaces will be introduced when the corresponding
application use cases are implemented.

## 7. Data Storage

PostgreSQL is planned for structured metadata.

Raw clinical images should be managed using a separate
access-controlled storage strategy.

The final storage architecture is not yet implemented.

## 8. Reproducibility

Training experiments should record:

- Code version
- Dataset version
- Model configuration
- Training configuration
- Random seed
- Package versions
- Hardware information
- Evaluation metrics

## 9. Security

The architecture must account for:

- Authentication
- Authorization
- Secret management
- Secure storage
- Data minimization
- Audit logging
- Retention policies
- Access control

## 10. Current Limitations

The architecture is a design proposal.

No end-to-end clinical prediction workflow has been
implemented or validated.
