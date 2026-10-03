# ✈️ Intelligent Predictive Maintenance & RUL Estimation

An end-to-end **Machine Learning Predictive Maintenance system** for turbofan engines using the **NASA C-MAPSS FD001** dataset.

The system estimates **Remaining Useful Life (RUL)** from engine sensor telemetry and combines machine learning, feature engineering, explainable AI, anomaly detection, an interactive Gradio dashboard, and a FastAPI inference interface.

---

## 🚀 Project Overview

Predictive maintenance aims to identify equipment degradation early enough to support maintenance planning and reduce unexpected failures.

This project implements an end-to-end workflow:

**Raw Telemetry → Sensor Filtering → Feature Engineering → Model Training → RUL Prediction → Explainability → Anomaly Detection → Fleet Monitoring → API Deployment**

The system supports both individual-engine predictions and fleet-level analysis.

---

## 🎯 Key Capabilities

- Remaining Useful Life (RUL) estimation
- Sensor preprocessing and feature engineering
- Multiple regression model benchmarking
- Evaluation on the unseen NASA test fleet
- SHAP-based model explainability
- Isolation Forest anomaly detection
- Interactive **Gradio fleet dashboard**
- **FastAPI REST API** for model inference
- Reusable prediction logic in `src/predict.py`

---

## 🧠 Dataset

### NASA C-MAPSS FD001

The project uses the NASA C-MAPSS FD001 turbofan engine dataset.

The dataset contains:

- **100 training turbofan units** with run-to-failure histories
- **100 unseen test units**
- 21 sensor measurements
- 3 operational settings
- Engine unit and cycle identifiers
- Ground-truth RUL values for test evaluation

Raw data is organized as:

```text
data/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
