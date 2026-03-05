# ⚕ Physician Churn Prediction — PharmaCRM Intelligence

> Hybrid LSTM + XGBoost model predicting physician disengagement in Pharma CRM.  
> Built by **Matt Derya** | Data Scientist & Clinical Pharmacy Expert

---

## 🧠 Model Architecture

```
Input Data
    ├── Time Series (12-month Rx volumes) ──→ LSTM (64→32 units) ──→ Embedding [16]
    └── Static Features (13 features)    ──→ XGBoost (500 trees) ──→ Probability
                                                        ↓
                                          Hybrid Ensemble (85% XGB + 15% LSTM)
                                                        ↓
                                          Optimal Threshold → Final Prediction
```

| Metric     | Score  |
|------------|--------|
| ROC-AUC    | 0.682  |
| F1 Score   | 0.644  |
| Algorithm  | LSTM + XGBoost Hybrid |
| Training   | 2,000 IQVIA-style physicians |

---

## 🚀 Quick Start

```bash
# 1. Extract zip and navigate
cd pharma_app

# 2. Build & run
docker-compose up --build

# 3. Open browser
http://localhost:8000
```

---

## 📡 API Usage

### POST /predict

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
  "message": "Immediate intervention required.",
  "top_risk_factors": [
    "Declining Rx trend: -4.3 scripts/month",
    "3 competitive brand switches detected",
    "No prescription in 90 days"
  ],
  "recommended_actions": [
    "Schedule urgent MSL visit to investigate Rx decline",
    "Present latest clinical data vs. competitive brands",
    "Immediate outreach — check for formulary or access issues"
  ]
}
```

---

## 🗂 Project Structure

```
pharma_app/
├── main.py                  ← FastAPI app with hybrid inference
├── lstm_model.keras         ← Trained LSTM (time series branch)
├── xgb_model.joblib         ← Trained XGBoost (static features branch)
├── scaler.joblib            ← StandardScaler (static features)
├── lstm_scaler.joblib       ← StandardScaler (Rx time series)
├── le_specialty.joblib      ← LabelEncoder
├── le_region.joblib         ← LabelEncoder
├── le_tier.joblib           ← LabelEncoder
├── model_meta.json          ← Weights, threshold, metrics
├── physician_data.csv       ← IQVIA-style synthetic training data
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── static/
    └── index.html           ← PharmaCRM web UI
```

---

## 💡 Domain Features (Pharma-Specific)

| Feature | Description | Business Logic |
|---|---|---|
| activity_score | Composite engagement (0-100) | Primary churn signal |
| msl_visits_6m | Medical Science Liaison visits | Face-time = retention |
| competitive_switches | Brand switch count | Direct defection signal |
| events_attended | Congress/advisory participation | High engagement = low churn |
| days_since_rx | Recency of last prescription | Key RFM metric |
| rx_trend | Last 3M vs first 3M avg | LSTM-derived momentum |
| engagement_idx | Composite index | Multi-signal aggregation |

---

*Matt Derya — Data Scientist | GenAI & LLM Specialist | Clinical Pharmacy Expert*  
*linkedin.com/in/matt-derya*
