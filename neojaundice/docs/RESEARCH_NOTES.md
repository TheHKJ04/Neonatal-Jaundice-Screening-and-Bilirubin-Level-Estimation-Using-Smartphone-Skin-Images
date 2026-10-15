# RESEARCH NOTES

## Problem Statement

Current bilirubin testing often requires blood sampling or costly equipment.

Goal:

Develop a low-cost AI system using smartphone images.

---

# Reference Repositories

## NeoJaundice-AI

Ideas:
- EfficientNet
- Bilirubin Regression
- Explainability

---

## neonatal-jaundice-screening

Ideas:
- Multimodal pipeline
- Clinical feature integration

---

## neonatal-jaundice-tracker

Ideas:
- Screening workflow

---

https://github.com/shreyann7/Bilirubin-Quantification-and-Jaundice-Prediction/blob/main/Jscan-NITK.pdf
https://github.com/KIRIT-P-S/Neonatal-Jaundice-Detection
https://github.com/RAHULPATEL2002/NeoJaundice-AI
https://github.com/shubh-bhateja/neonatal-jaundice-screening/blob/main/development/Final_Synopsis_Report.pdf
https://github.com/Adi030103/Neonatal-Jaundice-Detection/blob/main/jaundice_v23_full.ipynb
https://github.com/SamiyaS5904/NeoScan-Minor-Project-
https://github.com/kchristensen17/neonatal-jaundice-tracker
https://github.com/shubh-bhateja/neonatal-jaundice-screening


# Research Questions

RQ1:
Can smartphone images predict jaundice severity?

RQ2:
Can bilirubin levels be estimated from skin images?

RQ3:
Can model explanations improve trust?

---

# Literature Review Notes

Add paper summaries here.

Example:

Paper:
"Smartphone Based Neonatal Jaundice Detection"

Dataset:
xxxx

Method:
CNN

Limitations:
Lighting sensitivity

Potential Improvements:
Color normalization

## Phase 5.3 — Initial Color-Based Segmentation Findings

Dataset:
- 478 neonatal images
- 239 forehead images
- 239 sternum images

Methods:
- YCrCb thresholding
- HSV thresholding

Technical execution:
- 956 method evaluations
- 0 execution failures
- 270 automated unit tests passed

Mean geometric ROI coverage:
- YCrCb: 99.8521%
- HSV: 99.9944%

High-coverage warnings:
- YCrCb: 475/478
- HSV: 478/478

Interpretation:

Both methods retained nearly all geometric ROI pixels.
The initial thresholds therefore provided limited additional
pixel selection relative to the geometric ROI baseline.

This is an observation about coverage, not segmentation accuracy.

Limitations:
- No annotated skin segmentation masks
- Visual evaluation pending
- No verified numerical bilirubin labels
- No established clinical validity

Next experiment:
Analyze region-specific coverage distributions and evaluate
threshold sensitivity using controlled experiments.
