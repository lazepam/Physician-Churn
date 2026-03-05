from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from pathlib import Path
import joblib, json, numpy as np
import tensorflow as tf
import shap
tf.get_logger().setLevel('ERROR')

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Physician Churn Prediction API",
    description="Hybrid LSTM + XGBoost model predicting physician churn in Pharma CRM.",
    version="1.0.0"
)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
xgb_model   = joblib.load(BASE / "xgb_model.joblib")
scaler      = joblib.load(BASE / "scaler.joblib")
lstm_scaler = joblib.load(BASE / "lstm_scaler.joblib")
le_spec     = joblib.load(BASE / "le_specialty.joblib")
le_region   = joblib.load(BASE / "le_region.joblib")
le_tier     = joblib.load(BASE / "le_tier.joblib")
lstm_model  = tf.keras.models.load_model(BASE / "lstm_model.keras")
meta        = json.load(open(BASE / "model_meta.json"))

STATIC_FEATURES = meta["static_features"]
RX_COLS         = meta["rx_cols"]
XGB_W           = meta["xgb_weight"]
LSTM_W          = meta["lstm_weight"]
THRESHOLD       = meta["optimal_threshold"]

# ── SHAP Explainer (lazy init) ──────────────────────────────────────────────
_shap_explainer = None
def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(xgb_model)
    return _shap_explainer

# ── Schema ────────────────────────────────────────────────────────────────────
class PhysicianInput(BaseModel):
    specialty:            str   = Field(..., example="Oncology")
    region:               str   = Field(..., example="Northeast")
    tier:                 str   = Field(..., example="Tier 2 - High Value")
    years_practice:       int   = Field(..., ge=1,  le=50,   example=12)
    panel_size:           int   = Field(..., ge=50, le=1000, example=350)
    activity_score:       float = Field(..., ge=0,  le=100,  example=58.0)
    msl_visits_6m:        int   = Field(..., ge=0,  le=30,   example=3)
    competitive_switches: int   = Field(..., ge=0,  le=20,   example=1)
    events_attended:      int   = Field(..., ge=0,  le=20,   example=2)
    days_since_rx:        int   = Field(..., ge=0,  le=365,  example=25)
    rx_monthly:           list  = Field(..., min_items=12, max_items=12,
                                        example=[12,14,13,15,11,10,9,8,9,7,8,6])

    @validator('specialty')
    def validate_specialty(cls, v):
        valid = ['Oncology','Immunology','Cardiology','Neurology','Internal Medicine','Rheumatology']
        if v not in valid: raise ValueError(f"Must be one of: {valid}")
        return v

    @validator('region')
    def validate_region(cls, v):
        valid = ['Northeast','Southeast','Midwest','West','Southwest']
        if v not in valid: raise ValueError(f"Must be one of: {valid}")
        return v

    @validator('tier')
    def validate_tier(cls, v):
        valid = ['Tier 1 - KOL','Tier 2 - High Value','Tier 3 - Standard']
        if v not in valid: raise ValueError(f"Must be one of: {valid}")
        return v

class SHAPExplanation(BaseModel):
    feature:    str
    shap_value: float
    direction:  str  # "increases_churn" or "decreases_churn"

class PredictionResponse(BaseModel):
    churn_prediction:    int
    churn_probability:   float
    xgb_probability:     float
    lstm_probability:    float
    risk_level:          str
    risk_score:          int
    message:             str
    top_risk_factors:    list
    recommended_actions: list
    shap_explanation:    list

# ── Feature Builder ───────────────────────────────────────────────────────────
def build_features(inp: PhysicianInput):
    rx = np.array(inp.rx_monthly, dtype=float)
    rx_total    = rx.sum()
    rx_trend    = rx[-3:].mean() - rx[:3].mean()
    rx_vol      = rx.std()
    rx_last3_sh = rx[-3:].sum() / (rx_total + 1)
    eng_idx     = (inp.activity_score/100 * 0.5 +
                   min(inp.msl_visits_6m/20, 1.0) * 0.3 +
                   min(inp.events_attended/10, 1.0) * 0.2)
    rx_per_pat  = rx_total / (inp.panel_size + 1)

    spec_enc   = le_spec.transform([inp.specialty])[0]
    region_enc = le_region.transform([inp.region])[0]
    tier_enc   = le_tier.transform([inp.tier])[0]

    row = np.array([[
        inp.years_practice, inp.panel_size, inp.activity_score,
        inp.msl_visits_6m, inp.competitive_switches, inp.events_attended,
        inp.days_since_rx, rx_trend, rx_vol, rx_total, rx_last3_sh,
        eng_idx, rx_per_pat, spec_enc, region_enc, tier_enc
    ]])
    row_scaled = scaler.transform(row)

    rx_seq     = rx.reshape(1, 12, 1)
    rx_scaled  = lstm_scaler.transform(rx_seq.reshape(-1,1)).reshape(1,12,1)

    return row_scaled, rx_scaled, {
        'rx_trend': round(float(rx_trend),2),
        'engagement_idx': round(float(eng_idx),3),
        'rx_last3_share': round(float(rx_last3_sh),3),
        'rx_total_12m': round(float(rx_total),1),
        'competitive_switches': inp.competitive_switches,
        'activity_score': inp.activity_score,
        'days_since_rx': inp.days_since_rx,
    }

def risk_profile(prob, tier, feat):
    """Business-meaningful risk factors and actions"""
    risk_factors = []
    actions = []

    if feat['rx_trend'] < -2:
        risk_factors.append(f"Declining Rx trend: {feat['rx_trend']:+.1f} scripts/month")
        actions.append("Schedule urgent MSL visit to investigate Rx decline")
    if feat['engagement_idx'] < 0.35:
        risk_factors.append(f"Low engagement index: {feat['engagement_idx']:.2f}/1.00")
        actions.append("Enroll in KOL engagement program or advisory board")
    if feat['competitive_switches'] >= 2:
        risk_factors.append(f"{feat['competitive_switches']} competitive brand switches detected")
        actions.append("Present latest clinical data vs. competitive brands")
    if feat['days_since_rx'] > 60:
        risk_factors.append(f"No prescription in {feat['days_since_rx']} days")
        actions.append("Immediate outreach — check for formulary or access issues")
    if feat['activity_score'] < 45:
        risk_factors.append(f"Activity score below threshold: {feat['activity_score']:.0f}/100")
        actions.append("Assign dedicated sales rep for re-engagement")

    if not risk_factors:
        risk_factors.append("No single dominant risk factor — cumulative pattern")
        actions.append("Maintain current engagement cadence")

    if tier == 'Tier 1 - KOL':
        actions.append("Escalate to medical affairs — KOL retention is critical")

    level = "CRITICAL" if prob >= 0.70 else "HIGH" if prob >= 0.50 else "MODERATE" if prob >= 0.35 else "LOW"
    score = int(prob * 100)
    msg   = {
        "CRITICAL": "Immediate intervention required. High-value physician at critical churn risk.",
        "HIGH":     "Significant churn risk detected. Proactive retention strategy recommended.",
        "MODERATE": "Moderate risk. Monitor closely and consider targeted engagement.",
        "LOW":      "Physician appears stable. Maintain current relationship cadence."
    }[level]

    return level, score, msg, risk_factors[:3], actions[:3]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root(): return FileResponse(BASE / "static" / "index.html")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(physician: PhysicianInput):
    try:
        static_f, lstm_f, feat = build_features(physician)
        xgb_prob  = float(xgb_model.predict_proba(static_f)[0][1])
        lstm_prob = float(lstm_model.predict(lstm_f, verbose=0)[0][0])
        hybrid    = XGB_W * xgb_prob + LSTM_W * lstm_prob
        pred      = int(hybrid >= THRESHOLD)
        level, score, msg, factors, actions = risk_profile(hybrid, physician.tier, feat)

        # ── SHAP Explanation ──
        explainer   = get_shap_explainer()
        shap_values = explainer.shap_values(static_f)[0]
        shap_items  = sorted(
            zip(STATIC_FEATURES, shap_values),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        shap_explanation = [
            SHAPExplanation(
                feature=f,
                shap_value=round(float(v), 4),
                direction="increases_churn" if v > 0 else "decreases_churn"
            ) for f, v in shap_items
        ]

        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(hybrid, 4),
            xgb_probability=round(xgb_prob, 4),
            lstm_probability=round(lstm_prob, 4),
            risk_level=level, risk_score=score,
            message=msg,
            top_risk_factors=factors,
            recommended_actions=actions,
            shap_explanation=shap_explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": "LSTM+XGBoost Hybrid",
            "roc_auc": meta["roc_auc"], "f1": meta["f1_score"]}

@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "architecture": "Hybrid LSTM (time-series) + XGBoost (static features)",
        "xgb_weight": XGB_W, "lstm_weight": LSTM_W,
        "threshold": THRESHOLD,
        "roc_auc": meta["roc_auc"], "f1_score": meta["f1_score"],
        "training_data": "Synthetic IQVIA-style physician data (2,000 physicians)",
        "features": STATIC_FEATURES,
    }


@app.post("/explain", tags=["Explainability"])
def explain(physician: PhysicianInput):
    """
    Returns full SHAP explanation for a physician prediction.
    Shows which features drive churn risk up or down.
    """
    try:
        static_f, lstm_f, feat = build_features(physician)
        explainer   = get_shap_explainer()
        shap_values = explainer.shap_values(static_f)[0]
        base_value  = float(explainer.expected_value)

        explanation = []
        for feature, sv in sorted(zip(STATIC_FEATURES, shap_values),
                                   key=lambda x: abs(x[1]), reverse=True):
            explanation.append({
                "feature":    feature,
                "shap_value": round(float(sv), 4),
                "direction":  "increases_churn" if sv > 0 else "decreases_churn",
                "impact":     "HIGH" if abs(sv) > 0.1 else "MEDIUM" if abs(sv) > 0.05 else "LOW"
            })

        return {
            "base_value":  round(base_value, 4),
            "explanation": explanation,
            "summary": f"Top driver: {explanation[0]['feature']} "
                       f"({'↑' if explanation[0]['shap_value'] > 0 else '↓'} churn by {abs(explanation[0]['shap_value']):.3f})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
