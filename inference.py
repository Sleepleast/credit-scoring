"""
Credit Scoring — Inference Module
DTSC6012001 – Model Deployment | Dataset A

Usage:
    from inference import CreditScoringPredictor
    predictor = CreditScoringPredictor()
    result = predictor.predict_single({...})
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import __main__
try:
    from pipeline import CreditDataPreprocessor
    __main__.CreditDataPreprocessor = CreditDataPreprocessor
except ImportError:
    print("[Warning] Could not import pipeline.py. Preprocessor loading may fail.")

BASE_DIR       = Path(__file__).parent
MODELS_DIR     = BASE_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Ordered list of model files from best → fallback
_MODEL_CANDIDATES = [
    MODELS_DIR / "lightgbm.pkl",
    MODELS_DIR / "xgboost.pkl",
    MODELS_DIR / "hist_gradient_boosting.pkl",
    MODELS_DIR / "random_forest.pkl",
    MODELS_DIR / "gradient_boosting.pkl",
    MODELS_DIR / "logistic_regression.pkl",
]

LABEL_MAP_INV = {0: "Poor", 1: "Standard", 2: "Good"}
LABEL_COLORS  = {"Poor": "🔴", "Standard": "🟡", "Good": "🟢"}

# Mapping for Streamlit display
FEATURE_DESCRIPTIONS = {
    "Age":                       "Age (years)",
    "Annual_Income":             "Annual Income (USD)",
    "Monthly_Inhand_Salary":     "Monthly In-hand Salary (USD)",
    "Num_Bank_Accounts":         "Number of Bank Accounts",
    "Num_Credit_Card":           "Number of Credit Cards",
    "Interest_Rate":             "Interest Rate (%)",
    "Num_of_Loan":               "Number of Active Loans",
    "Delay_from_due_date":       "Avg Delay from Due Date (days)",
    "Num_of_Delayed_Payment":    "Number of Delayed Payments",
    "Changed_Credit_Limit":      "Changed Credit Limit (%)",
    "Num_Credit_Inquiries":      "Number of Credit Inquiries",
    "Outstanding_Debt":          "Outstanding Debt (USD)",
    "Credit_Utilization_Ratio":  "Credit Utilization Ratio (%)",
    "Credit_History_Age":        "Credit History Age (e.g. '5 Years and 3 Months')",
    "Total_EMI_per_month":       "Total EMI per Month (USD)",
    "Amount_invested_monthly":   "Amount Invested Monthly (USD)",
    "Monthly_Balance":           "Monthly Balance (USD)",
    "Occupation":                "Occupation",
    "Credit_Mix":                "Credit Mix",
    "Payment_of_Min_Amount":     "Pays Minimum Amount",
    "Payment_Behaviour":         "Payment Behaviour",
}


class CreditScoringPredictor:
    """
    Loads the trained preprocessor and best model, then exposes a
    simple predict interface.
    """

    def __init__(
        self,
        preprocessor_path: str = str(PREPROCESSOR_PATH),
        model_path: str = None,
    ):
        self.preprocessor = joblib.load(preprocessor_path)

        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = self._load_best_available()

    @staticmethod
    def _load_best_available():
        for p in _MODEL_CANDIDATES:
            if p.exists():
                model = joblib.load(p)
                print(f"[Predictor] Loaded model: {p.name}")
                return model
        raise FileNotFoundError(
            "No model .pkl found in models/. Run pipeline.py first."
        )

    # ── public methods ────────────────────────────────────────────
    def predict_single(self, input_dict: dict) -> dict:
        df = pd.DataFrame([input_dict])
        X  = self.preprocessor.transform(df)
        label_id  = int(self.model.predict(X)[0])
        label_str = LABEL_MAP_INV[label_id]

        result = {"label": label_str, "label_id": label_id}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            result["probabilities"] = {
                LABEL_MAP_INV[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            }
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for a whole DataFrame; appends 'Predicted_Credit_Score' column."""
        X         = self.preprocessor.transform(df)
        label_ids = self.model.predict(X)
        df = df.copy()
        df["Predicted_Credit_Score"] = [LABEL_MAP_INV[i] for i in label_ids]
        return df


# ─────────────────────────────────────────────
# SAMPLE TEST CASES (one per class)
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "label": "Expected: Standard",
        "input": {
            "Age": 22,
            "Annual_Income": 12000, # Fixed from 1200 (too low)
            "Monthly_Inhand_Salary": 800,
            "Num_Bank_Accounts": 11,
            "Num_Credit_Card": 10,
            "Interest_Rate": 40,
            "Num_of_Loan": 9,
            "Delay_from_due_date": 80,
            "Num_of_Delayed_Payment": 22,
            "Changed_Credit_Limit": 28.0,
            "Num_Credit_Inquiries": 17,
            "Credit_Mix": "Bad",
            "Outstanding_Debt": 6000, # Fixed from 60000 (too high)
            "Credit_Utilization_Ratio": 86,
            "Credit_History_Age": "1 Years and 2 Months",
            "Payment_of_Min_Amount": "No",
            "Total_EMI_per_month": 340,
            "Amount_invested_monthly": 0,
            "Payment_Behaviour": "Low_spent_Small_value_payments",
            "Monthly_Balance": 30,
            "Occupation": "Developer"
        }
    },
    {
        "label": "Expected: Poor",
        "input": {
            "Age": 32,
            "Annual_Income": 52000,
            "Monthly_Inhand_Salary": 3800,
            "Num_Bank_Accounts": 6,          # Increased to add risk
            "Num_Credit_Card": 5,            # Increased to add risk
            "Interest_Rate": 18,             # Increased to add risk
            "Num_of_Loan": 4,                # Increased to add risk
            "Delay_from_due_date": 28,       # Increased to add risk
            "Num_of_Delayed_Payment": 12,    # Increased to add risk
            "Changed_Credit_Limit": 4.0,
            "Num_Credit_Inquiries": 6,       # Increased to add risk
            "Credit_Mix": "Standard",
            "Outstanding_Debt": 3800,        # Increased to add risk
            "Credit_Utilization_Ratio": 65,  # Increased to add risk
            "Credit_History_Age": "7 Years and 4 Months",
            "Payment_of_Min_Amount": "NM",
            "Total_EMI_per_month": 190,
            "Amount_invested_monthly": 150,
            "Payment_Behaviour": "Low_spent_Medium_value_payments",
            "Monthly_Balance": 180,          # Decreased to add risk
            "Occupation": "Accountant"
        }
    },
    {
        "label": "Expected: Good",
        "input": {
            "Age": 50,
            "Annual_Income": 250000,
            "Monthly_Inhand_Salary": 20000,
            "Num_Bank_Accounts": 3,
            "Num_Credit_Card": 4,           # Increased: Active cards are better than no cards
            "Interest_Rate": 3,             # Very low: Indicates prime/super-prime eligibility
            "Num_of_Loan": 1,               # Increased: Shows ability to handle a loan
            "Delay_from_due_date": 0,
            "Num_of_Delayed_Payment": 0,
            "Changed_Credit_Limit": 15.0,
            "Num_Credit_Inquiries": 1,      # Increased: A single recent inquiry is "normal" activity
            "Credit_Mix": "Good",
            "Outstanding_Debt": 500,        # Increased from 0: Perfect payment history on small debt
            "Credit_Utilization_Ratio": 5,  # Slightly above zero: Shows active card usage
            "Credit_History_Age": "20 Years and 0 Months",
            "Payment_of_Min_Amount": "Yes",
            "Total_EMI_per_month": 150,     # Consistent with having 1 loan
            "Amount_invested_monthly": 5000,
            "Payment_Behaviour": "High_spent_Large_value_payments",
            "Monthly_Balance": 10000,
            "Occupation": "Scientist"
        }
    }
]


# ─────────────────────────────────────────────
# ENTRY POINT — run test cases
# ─────────────────────────────────────────────
if __name__ == "__main__":
    predictor = CreditScoringPredictor()
    # In inference.py, inside your predict_single method:
    print("=" * 60)
    print("          CREDIT SCORING — TEST CASES")
    print("=" * 60)

    for tc in TEST_CASES:
        result = predictor.predict_single(tc["input"])
        emoji  = LABEL_COLORS.get(result["label"], "")
        print(f"\n{tc['label']}")
        print(f"  → Prediction : {emoji} {result['label']}")
        if "probabilities" in result:
            for cls, prob in result["probabilities"].items():
                bar = "█" * int(prob * 30)
                print(f"     {cls:10s} {prob:.4f}  {bar}")
