# Parkinson_Rest_Tremor_Screening_Assistant

# 🧠 Resting Tremor Detection System

A deep learning project for **Neural Networks (AI303)** at **Pharos University** — detecting early signs of Parkinson’s disease using wearable sensor data.

---

## 📌 Overview

This project builds an intelligent system to detect **Resting Tremor** (a key Parkinson’s symptom) from numerical sensor features. It helps:
- Enable **early diagnosis** in low-resource areas.
- Reduce misdiagnosis and healthcare costs.
- Support doctors with AI-assisted decision tools.

Aligned with **UN SDG 3**: *Good Health and Well-being*.

---

## 📊 Dataset

- Type: Numerical features (from wearable sensors)
- Total samples: 4151
- Classes: Binary (0 = No Tremor, 1 = Tremor)
- Distribution:
  - Class 0: 2576 samples
  - Class 1: 1575 samples
- Slight imbalance → handled using `class_weight='balanced'`

---

## 🔧 Preprocessing Steps

1. Dropped non-numeric columns (`start_timestamp`, `end_timestamp`)
2. Applied `MinMaxScaler` for feature scaling
3. Reduced dimensions using `PCA(n_components=40)` (~95% variance retained)
4. Computed class weights to handle imbalance

---

## 🧠 Models Trained

### ✅ MLP (Selected for Web App)
- Architecture:  
  `Dense(128) → BatchNorm → Dropout(0.5) → Dense(64) → BatchNorm → Dropout(0.3) → Dense(32) → Dense(1, sigmoid)`
- Optimizer: Adam
- EarlyStopping with patience=10
- Final Test Results (MLP):
  - Accuracy: 0.9627
  - Recall: 0.9746
  - F1 Score: 0.9519
  - Precision: 0.9389
  - Confusion Matrix:
    - True Negatives: 496
    - False Positives: 20
    - False Negatives: 11
    - True Positives: 304
   
<p align="center">
  <img src="./download.png" width="45%">
  <img src="./download (2).png" width="45%">
  <img src="./download (1).png" width="45%">
</p>

### 🔍 CNN-1D (Evaluated, Not Deployed)
- Architecture:  
  `Conv1D(32) → MaxPool → Dropout(0.4) → Conv1D(64) → MaxPool → Dropout(0.4) → Flatten → Dense(64) → Dropout(0.5) → Dense(1, sigmoid)`
- Used L2 regularization + ReduceLROnPlateau
- Best threshold optimized to 0.5933 (via PR curve)
- Final Test Results (CNN-1D):
  - Accuracy: 0.9639
  - Recall: 0.9778
  - F1 Score: 0.9536

<p align="center">
  <img src="./download (3).png" width="45%">
  <img src="./download (4).png" width="45%">
  <img src="./download (5).png" width="45%">
</p>
---

## 🎯 Why MLP Was Chosen for Deployment

Even though CNN-1D achieved slightly higher recall (0.9778 vs 0.9746), **MLP was selected for the web application** because:
- Faster inference time (~2.1 ms/sample vs ~5.8 ms)
- Smaller model size (~150 KB vs ~480 KB)
- More stable across different runs (lower accuracy variance)
- Easier to integrate in Flask web apps (no reshaping needed)

This ensures real-world reliability — especially in clinics with limited hardware.

---

## 🌐 Web Application

- Built with **Flask + Gradio**
- Features:
  - Upload CSV file
  - Automatic preprocessing (scaling → PCA)
  - Real-time prediction using the trained MLP model
  - Clear result display (e.g., *"✅ Resting Tremor Detected — Confidence: 96.3%"*)

<p align="center">
  <img src="./Screenshot 2025-11-30 181201.png" width="45%">
  <img src="./Screenshot 2025-11-30 181221.png" width="45%">
</p>

---
