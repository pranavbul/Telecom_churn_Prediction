"""
app.py  (Streamlit UI)
----------------------
A clean web interface to interact with the churn prediction API.

Features:
  - Input form for all customer features
  - Calls the FastAPI /predict endpoint
  - Displays churn probability gauge, risk badge, and actions

Usage:
    streamlit run ui/app.py
"""

import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

API_URL = "http://localhost:8000/predict"

# ── Custom CSS for a clean look ───────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; }
    .subtitle     { font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem; }
    .risk-high    { background:#FF4B4B; color:white; padding:8px 18px;
                    border-radius:20px; font-weight:700; font-size:1.1rem; }
    .risk-medium  { background:#FFA500; color:white; padding:8px 18px;
                    border-radius:20px; font-weight:700; font-size:1.1rem; }
    .risk-low     { background:#2ECC71; color:white; padding:8px 18px;
                    border-radius:20px; font-weight:700; font-size:1.1rem; }
    .action-box   { background:#F0F4FF; border-left:4px solid #4A90E2;
                    padding:12px 16px; border-radius:6px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📡 Telecom Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter customer details to predict churn risk and get retention recommendations.</p>', unsafe_allow_html=True)
st.divider()

# ── Input Form ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age    = st.slider("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

with col2:
    st.subheader("📋 Account Info")
    tenure        = st.slider("Tenure (months)", 0, 72, 8)
    contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    payment       = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "E-Wallet", "Cash"])
    paperless     = st.radio("Paperless Billing", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    st.subheader("📊 Usage & Support")
    monthly_recharge = st.number_input("Monthly Recharge ($)", 0.0, 500.0, 65.0, step=5.0)
    data_usage       = st.number_input("Data Usage (GB)", 0.0, 50.0, 4.5, step=0.5)
    call_minutes     = st.slider("Call Minutes / Month", 0, 600, 200)
    num_services     = st.slider("Number of Services", 1, 6, 2)
    num_complaints   = st.slider("Complaints Filed", 0, 10, 1)
    support_calls    = st.slider("Support Calls", 0, 10, 2)
    sentiment        = st.slider("Sentiment Score", -1.0, 1.0, 0.2, step=0.1,
                                  help="-1 = very unhappy, +1 = very happy")

st.divider()

# ── Predict Button ─────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):
    payload = {
        "age":               age,
        "gender":            gender,
        "location":          location,
        "tenure_months":     tenure,
        "contract_type":     contract_type,
        "monthly_recharge":  monthly_recharge,
        "data_usage_gb":     data_usage,
        "call_minutes":      call_minutes,
        "num_services":      num_services,
        "num_complaints":    num_complaints,
        "support_calls":     support_calls,
        "payment_method":    payment,
        "paperless_billing": paperless,
        "sentiment_score":   sentiment,
    }

    with st.spinner("Analysing customer profile …"):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure FastAPI is running on port 8000.\n\n"
                     "Run: `uvicorn api.main:app --reload --port 8000`")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e}")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────
    prob       = result["churn_probability"]
    risk       = result["risk_level"]
    actions    = result["suggested_actions"]
    prob_pct   = round(prob * 100, 1)

    r1, r2 = st.columns([1, 2])

    with r1:
        st.subheader("Churn Probability")
        # Colour the metric based on risk
        colour = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[risk]
        st.metric(label="Probability", value=f"{prob_pct}%", delta=f"{colour} {risk} Risk")

        # Risk badge
        badge_class = f"risk-{risk.lower()}"
        st.markdown(f'<span class="{badge_class}">{risk} Risk</span>', unsafe_allow_html=True)

        # Progress bar coloured by risk
        bar_colour = {"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#2ECC71"}[risk]
        st.markdown(f"""
            <div style="background:#e0e0e0;border-radius:10px;height:18px;margin-top:14px;">
                <div style="width:{prob_pct}%;background:{bar_colour};
                            height:18px;border-radius:10px;"></div>
            </div>
            <small style="color:#888;">{prob_pct}% chance of churning</small>
        """, unsafe_allow_html=True)

    with r2:
        st.subheader("🎯 Recommended Actions")
        for action in actions:
            st.markdown(f'<div class="action-box">{action}</div>', unsafe_allow_html=True)

    # ── Expandable: raw JSON ───────────────────────────────────────────────
    with st.expander("🔧 Raw API Response (for developers)"):
        st.json(result)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Telecom Churn Predictor v1.0 • Powered by XGBoost + FastAPI")