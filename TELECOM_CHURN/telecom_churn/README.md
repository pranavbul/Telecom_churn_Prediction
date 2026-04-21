# 📡 Telecom Churn Prediction System

A complete, beginner-friendly, production-ready ML system that predicts customer churn.

---

## 🗂️ Project Structure

```
telecom_churn/
│
├── data/
│   └── generate_data.py        ← Creates synthetic dataset
│
├── pipelines/
│   ├── preprocessor.py         ← sklearn preprocessing pipeline
│   └── train.py                ← Trains all 3 models, saves best
│
├── models/
│   └── churn_model.pkl         ← Saved XGBoost model (created after training)
│
├── api/
│   └── main.py                 ← FastAPI prediction service
│
├── ui/
│   └── app.py                  ← Streamlit web interface
│
├── tests/
│   └── test_api.py             ← Pytest unit tests
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml           ← GitHub Actions CI/CD pipeline
│
├── Dockerfile                  ← Docker build for API
├── docker-compose.yml          ← Runs API + UI together
├── requirements.txt            ← All Python dependencies
└── README.md                   ← This file
```

---

## 🚀 Step-by-Step Setup (Complete Beginner Guide)

### ✅ Prerequisites

Make sure you have these installed:
- **Python 3.10 or 3.11** → https://python.org/downloads
- **Git** → https://git-scm.com
- **Docker** (optional, for containerisation) → https://docker.com

Check your Python version:
```bash
python --version
```

---

### STEP 1 — Clone / Download the project

```bash
# If using Git:
git clone https://github.com/YOUR_USERNAME/telecom-churn.git
cd telecom_churn

# Or just unzip the folder and open a terminal inside it
```

---

### STEP 2 — Create a virtual environment

A virtual environment keeps your project's packages separate from your system Python.

```bash
# Create the environment (only once)
python -m venv venv

# Activate it:
# On Mac / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You'll see `(venv)` in your terminal prompt — that means it's active. ✅

---

### STEP 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, scikit-learn, xgboost, fastapi, uvicorn, streamlit, pytest, etc.

---

### STEP 4 — Generate the dataset

```bash
cd data
python generate_data.py
cd ..
```

This creates `data/telecom_churn.csv` with 5,000 synthetic customers.

---

### STEP 5 — Train the model

```bash
python pipelines/train.py
```

What happens:
1. Loads the CSV
2. Trains **Logistic Regression** (baseline)
3. Trains **Random Forest**
4. Trains **XGBoost** (best model)
5. Prints metrics (Recall, F1, ROC-AUC) for all three
6. Saves `models/churn_model.pkl`

Expected output:
```
Model: Logistic Regression   Recall=0.72  F1=0.68  ROC-AUC=0.85
Model: Random Forest         Recall=0.75  F1=0.71  ROC-AUC=0.88
Model: XGBoost               Recall=0.78  F1=0.74  ROC-AUC=0.91
✅ XGBoost pipeline saved to: models/churn_model.pkl
```

---

### STEP 6 — Start the FastAPI server

Open a **new terminal**, activate your venv, then run:

```bash
uvicorn api.main:app --reload --port 8000
```

The API is now live at: **http://localhost:8000**

- Swagger docs (try the API in browser): http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

### STEP 7 — Test the API manually

Open another terminal and run:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "sentiment_score": -0.3
  }'
```

Expected response:
```json
{
  "churn_probability": 0.7821,
  "risk_level": "High",
  "suggested_actions": [
    "📞 Assign a dedicated retention agent ...",
    "💰 Offer a 20–30% discount ...",
    "📝 Offer a free upgrade ..."
  ]
}
```

---

### STEP 8 — Start the Streamlit UI

With FastAPI already running, open a **third terminal**, activate your venv, then:

```bash
streamlit run ui/app.py
```

The web UI opens at: **http://localhost:8501**

You can:
- Fill in customer details using sliders and dropdowns
- Click "🔍 Predict Churn Risk"
- See the probability gauge, risk badge, and recommended actions

---

### STEP 9 — Run the tests

```bash
pytest tests/ -v
```

All 8 tests should pass ✅

---

## 🐳 Docker Setup (Optional)

### Option A — Docker Compose (API + UI together)

```bash
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501

### Option B — API only

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 🔁 CI/CD with GitHub Actions

The workflow at `.github/workflows/ci_cd.yml` runs automatically on every push to `main`:

1. Installs dependencies
2. Generates dataset
3. Trains the model
4. Runs all pytest tests
5. Builds and smoke-tests the Docker image

To enable it:
1. Push this project to GitHub
2. Go to your repo → **Actions** tab
3. The workflow runs automatically on the next push

---

## 📊 Model Details

| Model               | Recall | F1 Score | ROC-AUC |
|---------------------|--------|----------|---------|
| Logistic Regression | ~0.72  | ~0.68    | ~0.85   |
| Random Forest       | ~0.75  | ~0.71    | ~0.88   |
| **XGBoost** ✅      | ~0.78  | ~0.74    | ~0.91   |

> **Why Recall matters for churn?** We want to catch as many churners as possible, even if it means a few false alarms. Missing a churner costs more than sending an unnecessary retention offer.

---

## 🎛️ API Reference

### `POST /predict`

**Request body:**
| Field             | Type   | Example         |
|-------------------|--------|-----------------|
| age               | int    | 35              |
| gender            | string | "Male"          |
| location          | string | "Urban"         |
| tenure_months     | int    | 8               |
| contract_type     | string | "Month-to-Month"|
| monthly_recharge  | float  | 85.0            |
| data_usage_gb     | float  | 5.2             |
| call_minutes      | int    | 300             |
| num_services      | int    | 2               |
| num_complaints    | int    | 2               |
| support_calls     | int    | 3               |
| payment_method    | string | "E-Wallet"      |
| paperless_billing | int    | 1 (yes) / 0     |
| sentiment_score   | float  | -0.3 (-1 to +1) |

**Response:**
```json
{
  "churn_probability": 0.7821,
  "risk_level": "High",
  "suggested_actions": ["..."]
}
```

**Risk Levels:**
- 🔴 **High** → probability ≥ 70%
- 🟠 **Medium** → probability 40–70%
- 🟢 **Low** → probability < 40%

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure venv is active: `source venv/bin/activate` |
| `Model not found` | Run `python pipelines/train.py` first |
| `Connection refused` on UI | Start FastAPI: `uvicorn api.main:app --port 8000` |
| Tests failing | Run from project root, not from `tests/` folder |
| Docker build fails | Make sure `models/churn_model.pkl` exists before building |

---

## 🔧 Customisation Tips

- **Use real data**: Replace `data/telecom_churn.csv` with your actual dataset. Keep the same column names, or update `NUMERICAL_FEATURES` / `CATEGORICAL_FEATURES` in `pipelines/preprocessor.py`.
- **Tune XGBoost**: Edit `n_estimators`, `max_depth`, `learning_rate` in `pipelines/train.py`.
- **Change risk thresholds**: Edit `get_risk_level()` in `api/main.py`.
- **Add new actions**: Edit `get_suggested_actions()` in `api/main.py`.