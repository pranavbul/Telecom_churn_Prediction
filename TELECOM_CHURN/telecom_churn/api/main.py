"""
main.py  (FastAPI application)
------------------------------
Serves the churn prediction model via a REST API.

Endpoints:
  GET  /          → health check
  POST /predict   → churn probability + risk level
  GET  /health    → detailed health info

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import sys
from typing import Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Load Model ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

model = None  # loaded lazily on first request (or at startup below)


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(
            f"Model not found. Run: python pipelines/train.py first."
        )
    model = joblib.load(MODEL_PATH)
    log.info("✅ Model loaded successfully")


# ── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predicts the probability of a telecom customer churning.",
    version="1.0.0",
)

# Allow Streamlit UI to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load the model when the server starts."""
    try:
        load_model()
    except FileNotFoundError as e:
        log.warning(str(e))


# ── Request / Response Schemas ────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Input: one customer's features. All fields are validated."""

    age:               int   = Field(..., ge=18, le=100, description="Customer age (18–100)")
    gender:            str   = Field(..., description="'Male' or 'Female'")
    location:          str   = Field(..., description="'Urban', 'Suburban', or 'Rural'")
    tenure_months:     int   = Field(..., ge=0, le=240, description="Months as customer")
    contract_type:     str   = Field(..., description="'Month-to-Month', 'One Year', 'Two Year'")
    monthly_recharge:  float = Field(..., ge=0, le=1000, description="Monthly bill in $")
    data_usage_gb:     float = Field(..., ge=0, le=100, description="Monthly data usage (GB)")
    call_minutes:      int   = Field(..., ge=0, le=10000, description="Monthly call minutes")
    num_services:      int   = Field(..., ge=0, le=10, description="Number of subscribed services")
    num_complaints:    int   = Field(..., ge=0, le=50, description="Number of complaints filed")
    support_calls:     int   = Field(..., ge=0, le=50, description="Number of support calls")
    payment_method:    str   = Field(..., description="'Credit Card', 'Bank Transfer', 'E-Wallet', 'Cash'")
    paperless_billing: int   = Field(..., ge=0, le=1, description="1 = paperless, 0 = paper")
    sentiment_score:   float = Field(..., ge=-1, le=1, description="Sentiment score from -1 to +1")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("location")
    @classmethod
    def validate_location(cls, v):
        allowed = {"Urban", "Suburban", "Rural"}
        if v not in allowed:
            raise ValueError(f"location must be one of {allowed}")
        return v

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v):
        allowed = {"Month-to-Month", "One Year", "Two Year"}
        if v not in allowed:
            raise ValueError(f"contract_type must be one of {allowed}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v):
        allowed = {"Credit Card", "Bank Transfer", "E-Wallet", "Cash"}
        if v not in allowed:
            raise ValueError(f"payment_method must be one of {allowed}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "Male",
                "location": "Urban",
                "tenure_months": 8,
                "contract_type": "Month-to-Month",
                "monthly_recharge": 85.0,
                "data_usage_gb": 5.2,
                "call_minutes": 300,
                "num_services": 2,
                "num_complaints": 2,
                "support_calls": 3,
                "payment_method": "E-Wallet",
                "paperless_billing": 1,
                "sentiment_score": -0.3,
            }
        }


class PredictionResponse(BaseModel):
    """Output: churn probability and risk level with suggested actions."""
    churn_probability: float
    risk_level:        str
    suggested_actions: list[str]


# ── Helper: map probability → risk level + actions ─────────────────────

def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"


def get_suggested_actions(risk_level: str, features: CustomerFeatures) -> list[str]:
    """Return context-aware retention recommendations."""
    actions = []

    if risk_level == "High":
        actions.append("📞 Assign a dedicated retention agent to call this customer within 24 hours.")
        actions.append("💰 Offer a 20–30% discount on the next 3 months of service.")
        if features.contract_type == "Month-to-Month":
            actions.append("📝 Offer a free upgrade if customer switches to a 1-year contract.")
        if features.num_complaints > 0:
            actions.append("🛠️  Prioritize resolving open complaints immediately.")
        if features.sentiment_score < 0:
            actions.append("💬 Send a personalised apology + satisfaction survey.")

    elif risk_level == "Medium":
        actions.append("📧 Send a personalised loyalty email with an exclusive offer.")
        actions.append("🎁 Offer a free service upgrade (e.g., extra data) for 1 month.")
        if features.tenure_months < 12:
            actions.append("🤝 Enrol in the 'New Customer Care' program for extra support.")

    else:  # Low
        actions.append("✅ Customer appears satisfied — continue standard engagement.")
        actions.append("🌟 Consider inviting to a loyalty rewards program.")

    return actions


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"message": "Telecom Churn Prediction API is running!", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn_probability**: float between 0 and 1
    - **risk_level**: 'High', 'Medium', or 'Low'
    - **suggested_actions**: list of recommended retention actions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first: python pipelines/train.py",
        )

    # Convert Pydantic model → DataFrame (model expects a DataFrame)
    input_dict = customer.model_dump()
    input_df   = pd.DataFrame([input_dict])

    log.info(f"Prediction request: {input_dict}")

    # Get probability from the pipeline
    prob = float(model.predict_proba(input_df)[0][1])
    prob = round(prob, 4)

    risk    = get_risk_level(prob)
    actions = get_suggested_actions(risk, customer)

    log.info(f"Result: prob={prob:.4f}, risk={risk}")

    return PredictionResponse(
        churn_probability=prob,
        risk_level=risk,
        suggested_actions=actions,
    )


# ── Dev Server ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)