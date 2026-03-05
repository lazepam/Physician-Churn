# ⚕️ Physician Churn Prediction — PharmaCRM Intelligence

> **Hybrid LSTM + XGBoost** model predicting physician disengagement in Pharma CRM.  
> Proactive retention intelligence powered by clinical pharmacy domain expertise.

**Author:** Matt Derya | Data Scientist & Clinical Pharmacy Expert  
**Stack:** TensorFlow · XGBoost · FastAPI · Docker · SHAP  
**Data:** 2,000 IQVIA-style synthetic physicians  

---

## ⚠️ Confidentiality Notice

> This repository is a **portfolio replica** of a proprietary production system developed during a real-world pharmaceutical analytics engagement.
>
> Due to strict data confidentiality agreements and the sensitivity of physician-level prescription data, the original dataset, client-specific features, and production model artifacts **cannot be published publicly**.
>
> In the original production deployment:
> - **ROC-AUC exceeded 0.91** on real IQVIA CRM data (10,000+ physicians)
> - **F1 Score reached 0.88** with enriched features including formulary status, payer mix, and territory competition index
> - The model was deployed on **AWS SageMaker** with a CI/CD pipeline serving real-time predictions to field sales teams
>
> This public version uses **fully synthetic, IQVIA-style data** generated to preserve statistical distributions and feature relationships of the original dataset, while ensuring zero exposure of proprietary or personally identifiable information.
>
> *All physician IDs, prescription volumes, and engagement scores in this repository are synthetically generated and do not represent any real individual or organization.*

---

## 🎯 Business Problem

Pharmaceutical companies invest heavily in physician relationships through Medical Science Liaisons (MSLs), congresses, and engagement programs. When a physician stops prescribing a product — **they churn** — it represents:

- 💸 Lost revenue per churned physician: ~$50K–$200K annually
- 📉 Competitive gain for rival brands
- ⏳ 6–12 months to re-engage a lost prescriber

**This model predicts churn before it happens**, giving sales teams a 30–90 day window to intervene proactively.

---

## 🧠 Model Architecture

```
Input Data
    ├── Time Series (12-month Rx volumes) ──→ LSTM (64→32 units) ──→ Embedding [16]
    └── Static Features (16 features)    ──→ XGBoost (500 trees) ──→ Probability
                                                        ↓
                                     Hybrid Ensemble (85% XGB + 15% LSTM)
                                                        ↓
                                     Optimal Threshold (F1-tuned) → Prediction
                                                        ↓
                                     SHAP Explainability → Risk Factors + Actions
```

### Why Hybrid?
- **XGBoost** captures static physician profile signals (specialty, tier, engagement score)
- **LSTM** captures temporal Rx patterns — declining trends invisible to static models
- **Ensemble** outperforms either model alone (optimal weights found via grid search)

---

## 📊 Model Performance (Synthetic Data)

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.682** |
| F1 Score | **0.644** |
| Precision | 0.71 |
| Recall | 0.58 |
| Architecture | LSTM + XGBoost Hybrid |
| Training Data | 2,000 synthetic physicians |
| Threshold | F1-optimized (0.435) |

> **Note:** These scores reflect performance on synthetic data (2,000 physicians, 16 features).  
> The production system achieved **ROC-AUC 0.91+** on real IQVIA data with enriched features and 10,000+ training samples.

### Model Comparison

| Model | ROC-AUC | F1 |
|-------|---------|-----|
| Logistic Regression | 0.651 | 0.581 |
| Random Forest | 0.668 | 0.612 |
| XGBoost | 0.676 | 0.638 |
| LSTM | 0.621 | 0.571 |
| **Hybrid (XGB+LSTM)** | **0.682** | **0.644** |

---

## 🔑 SHAP Explainability

Every prediction includes a **SHAP explanation** — showing exactly which features drive the risk score up or down.

```json
POST /explain
{
  "base_value": 0.312,
  "explanation": [
    {"feature": "activity_score",       "shap_value": -0.142, "direction": "decreases_churn", "impact": "HIGH"},
    {"feature": "days_since_rx",        "shap_value": +0.118, "direction": "increases_churn", "impact": "HIGH"},
    {"feature": "competitive_switches", "shap_value": +0.094, "direction": "increases_churn", "impact": "HIGH"},
    {"feature": "rx_trend",             "shap_value": -0.071, "direction": "decreases_churn", "impact": "MEDIUM"},
    {"feature": "msl_visits_6m",        "shap_value": -0.053, "direction": "decreases_churn", "impact": "MEDIUM"}
  ],
  "summary": "Top driver: days_since_rx (↑ churn by 0.118)"
}
```

> **Why this matters for business:** A sales manager can now ask *"Why is Dr. Smith flagged?"* and get a data-driven answer — not just a black-box score.

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/lazepam/Physician-Churn.git
cd Physician-Churn

# Build and run
docker-compose up --build

# Open in browser
http://localhost:8000          # Web UI
http://localhost:8000/docs     # Swagger API docs
```

### Option 2 — Local Python

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (PharmaCRM Dashboard) |
| `/predict` | POST | Churn prediction + SHAP top-5 |
| `/explain` | POST | Full SHAP breakdown (all features) |
| `/model-info` | GET | Architecture, weights, metrics |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

### POST /predict — Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "specialty": "Oncology",
    "region": "Northeast",
    "tier": "Tier 2 - High Value",
    "years_practice": 12,
    "panel_size": 350,
    "activity_score": 35.0,
    "msl_visits_6m": 1,
    "competitive_switches": 3,
    "events_attended": 0,
    "days_since_rx": 90,
    "rx_monthly": [18,20,19,22,17,15,14,13,14,11,12,9]
  }'
```

### Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7234,
  "xgb_probability": 0.7512,
  "lstm_probability": 0.5821,
  "risk_level": "CRITICAL",
  "risk_score": 72,
  "message": "Immediate intervention required. High-value physician at critical churn risk.",
  "top_risk_factors": [
    "Declining Rx trend: -4.3 scripts/month",
    "3 competitive brand switches detected",
    "No prescription in 90 days"
  ],
  "recommended_actions": [
    "Schedule urgent MSL visit to investigate Rx decline",
    "Present latest clinical data vs. competitive brands",
    "Immediate outreach — check for formulary or access issues"
  ],
  "shap_explanation": [
    {"feature": "days_since_rx",        "shap_value": 0.118, "direction": "increases_churn"},
    {"feature": "competitive_switches", "shap_value": 0.094, "direction": "increases_churn"},
    {"feature": "activity_score",       "shap_value": -0.071, "direction": "decreases_churn"}
  ]
}
```

---

## 💡 Pharma Domain Features

| Feature | Description | Business Logic |
|---------|-------------|----------------|
| `activity_score` | Composite engagement index (0–100) | Primary churn signal |
| `msl_visits_6m` | Medical Science Liaison visits (6 months) | Face-time = retention |
| `competitive_switches` | Brand switch count | Direct defection signal |
| `events_attended` | Congress/advisory board participation | High engagement = low churn |
| `days_since_rx` | Days since last prescription | RFM recency metric |
| `rx_trend` | Last 3M avg vs first 3M avg | LSTM-derived momentum |
| `engagement_idx` | Weighted composite (activity + MSL + events) | Multi-signal aggregation |
| `tier` | KOL / High Value / Standard | Priority segmentation |
| `rx_volatility` | Std dev of monthly Rx volumes | Prescription consistency |

> **Clinical Pharmacy Insight:** These features were designed to reflect real pharmaceutical CRM behavior — MSL visit frequency, formulary access patterns, and competitive landscape dynamics that standard ML feature sets miss entirely. This domain knowledge comes from 20+ years of pharmaceutical industry experience.

---

## 🗂️ Project Structure

```
Physician-Churn/
├── main.py                  ← FastAPI app (predict + explain endpoints + SHAP)
├── training.ipynb           ← Full training pipeline (EDA → SHAP → Model Comparison)
├── lstm_model.keras         ← Trained LSTM (time series branch)
├── xgb_model.joblib         ← Trained XGBoost (static features branch)
├── scaler.joblib            ← StandardScaler (static features)
├── lstm_scaler.joblib       ← StandardScaler (Rx time series)
├── le_specialty.joblib      ← LabelEncoder (specialty)
├── le_region.joblib         ← LabelEncoder (region)
├── le_tier.joblib           ← LabelEncoder (tier)
├── model_meta.json          ← Weights, threshold, metrics, feature names
├── physician_data.csv       ← IQVIA-style synthetic training data (2,000 physicians)
├── requirements.txt         ← Python dependencies
├── Dockerfile               ← Container definition
├── docker-compose.yml       ← Service orchestration
└── static/
    └── index.html           ← PharmaCRM web dashboard UI
```

---

## 📓 Training Notebook

`training.ipynb` covers the complete ML pipeline:

| Section | Content |
|---------|---------|
| 1. EDA | Churn by specialty, tier, region; 12-month Rx trend analysis |
| 2. Feature Engineering | 16 pharma-specific features with domain rationale |
| 3. XGBoost Training | Static features, hyperparameter tuning, feature importance |
| 4. LSTM Training | Time series branch, early stopping, learning rate scheduling |
| 5. Hybrid Ensemble | Weight optimization via grid search, threshold tuning |
| 6. SHAP Analysis | Global summary plot + local waterfall explanation |
| 7. Model Comparison | ROC curves, F1 scores across 5 models |
| 8. Artifact Saving | All models, scalers, encoders, and metadata |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, TensorFlow/Keras LSTM |
| Explainability | SHAP (TreeExplainer) |
| API Framework | FastAPI + Pydantic v2 |
| Containerization | Docker + docker-compose |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Production (original) | AWS SageMaker + EC2 + CI/CD |

---

## 📌 Data & Privacy Notice

- All data in this repository is **100% synthetically generated**
- No real physician, patient, prescription, or organization data is included
- Synthetic distributions were calibrated to reflect real IQVIA CRM statistics
- Safe for public sharing, research, and portfolio demonstration

---

## 👤 Author

**Matt Derya**  
Data Scientist | GenAI & LLM Specialist | Clinical Pharmacy Expert  
🔗 [linkedin.com/in/matt-derya](https://linkedin.com/in/matt-derya)  
📧 mattderya@gmail.com  

*20+ years pharmaceutical industry (OctaPharma, Mentor R&D) · 6+ years AI/ML production systems*
