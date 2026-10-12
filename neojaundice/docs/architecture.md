# SYSTEM ARCHITECTURE

## High-Level Flow

Smartphone Image

↓

Image Validation

↓

Face Detection

↓

ROI Extraction

↓

Skin Segmentation

↓

Color Correction

↓

Color Normalization

↓

Feature Extraction

↓

EfficientNet Backbone

↓

Multi-Task Network

├── Classification Head
│
└── Regression Head

↓

GradCAM

↓

FastAPI

↓

Database

↓

React Frontend

---

# Backend Architecture

backend/

app/

api/
services/
repositories/
models/
schemas/
utils/
core/

---

# Frontend Architecture

frontend/

src/

pages/
components/
services/
hooks/
types/

---

# Database Architecture

patients

images

predictions

audit_logs

model_versions

---

# ML Pipeline

Raw Data
↓
Validation
↓
Preprocessing
↓
Feature Engineering
↓
Training
↓
Evaluation
↓
Registry
↓
Inference

---

# Deployment Architecture

React
↓
Nginx

FastAPI
↓
Gunicorn/Uvicorn

PostgreSQL

Docker Compose

Prometheus

Grafana