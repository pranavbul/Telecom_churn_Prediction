"""
preprocessor.py
---------------
Defines the sklearn preprocessing pipeline:
  - Impute missing values
  - Encode categorical features
  - Scale numerical features

Import `build_preprocessor()` from other modules.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Feature Lists ─────────────────────────────────────────────────────────────

# Numerical features that need imputation + scaling
NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_recharge",
    "data_usage_gb",
    "call_minutes",
    "num_services",
    "num_complaints",
    "support_calls",
    "sentiment_score",
    "paperless_billing",
]

# Categorical features that need imputation + one-hot encoding
CATEGORICAL_FEATURES = [
    "gender",
    "location",
    "contract_type",
    "payment_method",
]

# Target column (not a feature)
TARGET = "churn"

# Column to drop (ID is not a feature)
DROP_COLS = ["customer_id"]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that handles:
      - Numerical: fill missing with median → standard scale
      - Categorical: fill missing with 'Unknown' → one-hot encode
    """

    # Step 1: Handle numerical columns
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # fill NaN with median
        ("scaler", StandardScaler()),                     # mean=0, std=1
    ])

    # Step 2: Handle categorical columns
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])

    # Step 3: Combine both into a single transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # silently drop any unlisted columns
    )

    return preprocessor