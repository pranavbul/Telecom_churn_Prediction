# ── Dockerfile ────────────────────────────────────────────────────────────
# Builds the FastAPI churn-prediction service.
#
# Build:   docker build -t churn-api .
# Run:     docker run -p 8000:8000 churn-api
# ──────────────────────────────────────────────────────────────────────────

# Use a slim Python 3.11 base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────
# libgomp1 is needed by XGBoost for multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer (speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project code ─────────────────────────────────────────────────────
COPY api/        ./api/
COPY pipelines/  ./pipelines/
COPY models/     ./models/
COPY data/       ./data/

# ── Train model if it doesn't exist yet ──────────────────────────────────
# (In CI/CD the model is usually pre-built and copied in)
RUN python -c "import os; \
    os.path.exists('models/churn_model.pkl') or \
    __import__('subprocess').run(['python', 'pipelines/train.py'], check=True)"

# ── Expose port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start FastAPI with Uvicorn ────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]