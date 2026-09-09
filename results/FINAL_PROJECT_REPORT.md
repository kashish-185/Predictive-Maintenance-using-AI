# ✈️ Intelligent Predictive Maintenance & RUL Estimation
## Project Summary & Executive Technical Report

### 1. Executive Summary
This project implements an end-to-end Machine Learning and Predictive Maintenance pipeline for high-bypass turbofan jet engines using the **NASA C-MAPSS FD001** dataset. The system continuously estimates the **Remaining Useful Life (RUL)** of engines and classifies operational risk to prevent catastrophic failures and minimize unplanned downtime.

---

### 2. Dataset & Signal Processing
- **Data Source:** NASA Ames Prognostics Center of Excellence (C-MAPSS FD001)
- **Fleet Scope:** 100 Training Turbofan Units (run-to-failure), 100 Validation/Test Units (random early truncation)
- **Sensor Filtering:** 7 flat / near-constant sensors removed (`sensor_1, 5, 6, 10, 16, 18, 19`).
- **Feature Engineering:** 
  - 5-Cycle Rolling Moving Averages
  - 5-Cycle Rolling Standard Deviations
  - 5-Cycle Trend Differentials (`diff(5)`)
  - Total Features: 56 engineered sensor signals.

---

### 3. Model Architecture & Benchmarking
| Algorithm | Validation MAE (cycles) | Validation RMSE (cycles) | Validation R² |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 29.48 | 37.43 | 0.6749 |
| **Random Forest** | 24.21 | 33.97 | 0.7323 |
| **Gradient Boosting** | 24.66 | 33.89 | 0.7335 |

---

### 4. NASA Unseen Test Fleet Evaluation
Evaluated against the ground-truth RUL on all 100 unseen test engines:
- **Test MAE:** 24.20 operating cycles
- **Test RMSE:** 32.76 operating cycles
- **NASA PHM Score (Asymmetric Penalty):** 34749.11

---

### 5. Deployment & Production Interfaces
1. **Explainable AI:** SHAP (TreeExplainer) feature attributions identifying primary degradation drivers.
2. **REST API:** FastAPI application serving real-time JSON predictions via `GET /predict/engine/{unit_id}`.
3. **Interactive Fleet Dashboard:** Gradio web application with single-engine diagnostics, degradation trend plots, and fleet risk prioritization.
