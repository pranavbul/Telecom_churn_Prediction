"""
test_api.py
-----------
Basic unit tests for the FastAPI prediction endpoint.

Run:
    pytest tests/test_api.py -v
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── We must import the app AFTER adding the path ──────────────────────────
from api.main import app

client = TestClient(app)

# ── Sample valid payload ──────────────────────────────────────────────────
VALID_CUSTOMER = {
    "age":               35,
    "gender":            "Male",
    "location":          "Urban",
    "tenure_months":     8,
    "contract_type":     "Month-to-Month",
    "monthly_recharge":  85.0,
    "data_usage_gb":     5.2,
    "call_minutes":      300,
    "num_services":      2,
    "num_complaints":    2,
    "support_calls":     3,
    "payment_method":    "E-Wallet",
    "paperless_billing": 1,
    "sentiment_score":   -0.3,
}


# ── Tests ─────────────────────────────────────────────────────────────────

def test_root_endpoint():
    """GET / should return 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """GET /health should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    """POST /predict with valid data returns expected shape."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    # If model is trained the status is 200; if not, it's 503
    if response.status_code == 503:
        pytest.skip("Model not trained yet — run pipelines/train.py first")
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "suggested_actions" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in {"High", "Medium", "Low"}
    assert isinstance(data["suggested_actions"], list)


def test_predict_invalid_gender():
    """Invalid gender should return 422 Unprocessable Entity."""
    bad_payload = {**VALID_CUSTOMER, "gender": "Robot"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_contract():
    """Invalid contract type should return 422."""
    bad_payload = {**VALID_CUSTOMER, "contract_type": "Weekly"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_age_out_of_range():
    """Age < 18 or > 100 should return 422."""
    bad_payload = {**VALID_CUSTOMER, "age": 10}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_sentiment_out_of_range():
    """Sentiment score must be between -1 and 1."""
    bad_payload = {**VALID_CUSTOMER, "sentiment_score": 5.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_field():
    """Missing required field should return 422."""
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422