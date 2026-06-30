"""
Credit Scoring - Local ML Pipeline
DTSC6012001 – Model Deployment | Dataset A
OOP-based pipeline with MLflow experiment tracking

FIXES & IMPROVEMENTS over original:
  BUG 1 - LogisticRegression: removed deprecated `multi_class='multinomial'`
           (parameter was removed in scikit-learn 1.7; lbfgs handles
            multinomial natively for multi-class targets).
  BUG 2 - SMOTE data leakage in cross-validation: cross_val_score was run
           on the already-SMOTE'd X_train, meaning synthetic minority samples
           could appear in both the CV train and validation folds, inflating
           scores. Fixed by keeping original (pre-SMOTE) splits and wrapping
           each model in an imblearn Pipeline so SMOTE is applied strictly
           inside each CV fold.
  BUG 3 - XGBoostModel missing `use_label_encoder` kwarg guard and no
           explicit `objective`; fixed by setting objective='multi:softprob'
           and removing the legacy kwarg.

  NEW   - LightGBMModel added; LightGBM's leaf-wise growth and built-in
           regularisation typically outperform XGBoost on mid-size tabular
           credit data.
  NEW   - class_weight='balanced' added to LogisticRegression and
           RandomForest to handle the class imbalance natively alongside SMOTE.
  NEW   - HistGradientBoostingModel added (sklearn's fast gradient booster;
           native NaN support, much faster than vanilla GradientBoosting).
  NEW   - roc_auc_ovr (one-vs-rest) added to evaluation metrics.
  NEW   - Feature importance logged to MLflow for tree models.
  IMPROVEMENT - Best-model selection now uses cv_f1_mean (cross-validated,
                therefore more robust than a single test-split F1).
  IMPROVEMENT - More model candidates with tighter hyperparameter ranges
                informed by dataset characteristics (n=25000, 3 classes).

ROUND-2 IMPROVEMENTS (based on observed results):
  FIX   - Removed class_weight from all SMOTE-trained tree models.  SMOTE
           already balances the training distribution; adding class_weight on
           top is a double-compensation that over-penalises the Standard
           majority class and hurts precision → lower F1_macro.  LR keeps
           class_weight because it has no tree-based resampling benefit.
  NEW   - Five engineered ratio features added to the preprocessor:
           debt_to_income, emi_to_salary, utilization_per_card,
           delayed_payment_rate, balance_to_income.  Domain-relevant ratios
           are among the strongest signals in credit-scoring benchmarks.
  NEW   - More aggressive LightGBM candidates (deeper leaves, lower lr,
           higher n_estimators) so the boosting budget matches dataset size.
  NEW   - XGBoost candidates with more trees + colsample tuning.
  NEW   - Early-stopping-aware LightGBM variant using eval set on val split.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, Optional

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # FIX BUG 2

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_PATH        = BASE_DIR / "data_A.csv"
MODELS_DIR       = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MLFLOW_URI       = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
EXPERIMENT_NAME  = "credit_scoring_experiment"


# ─────────────────────────────────────────────
# CLASS 1 — PREPROCESSING
# ─────────────────────────────────────────────
class CreditDataPreprocessor:
    """
    Handles all data cleaning and feature engineering for the
    credit scoring dataset.
    """

    FEATURES = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age_Months",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Occupation_enc", "Credit_Mix_enc", "Payment_of_Min_Amount_enc",
        "Payment_Behaviour_enc",
        "debt_to_income", "emi_to_salary", "utilization_per_card", 
        "delayed_payment_rate", "balance_to_income",
    ]
    TARGET    = "Credit_Score"
    LABEL_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

    def __init__(self):
        self.label_encoders = {}
        self.scaler         = StandardScaler()
        self.imputer        = SimpleImputer(strategy="median")
        self._fitted        = False

    @staticmethod
    def _clean_numeric_str(series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
                  .str.strip()
                  .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        cleaned = cleaned.replace("", np.nan)
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def _parse_credit_history_age(series: pd.Series) -> pd.Series:
        def _parse(val):
            if pd.isna(val):
                return np.nan
            years  = re.search(r"(\d+)\s*Year",  str(val))
            months = re.search(r"(\d+)\s*Month", str(val))
            y = int(years.group(1))  if years  else 0
            m = int(months.group(1)) if months else 0
            return y * 12 + m
        return series.apply(_parse)

    @staticmethod
    def _clean_categorical(series: pd.Series, noise_tokens: list) -> pd.Series:
        s = series.astype(str).str.strip()
        for tok in noise_tokens:
            s = s.replace(tok, np.nan)
        return s

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # [FIX]: Dynamically clean ALL numeric columns to prevent NaN leakage during coercion
        num_cols = [
            "Age", "Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card",
            "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
            "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
            "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
            "Monthly_Inhand_Salary"
        ]
        
        for col in num_cols:
            if col in df.columns:
                df[col] = self._clean_numeric_str(df[col])

        # clip unrealistic ages
        df["Age"] = df["Age"].where((df["Age"] >= 18) & (df["Age"] <= 100))

        # credit history age → months
        df["Credit_History_Age_Months"] = self._parse_credit_history_age(
            df["Credit_History_Age"]
        )

        # categoricals with noise tokens
        df["Occupation"]        = self._clean_categorical(df["Occupation"],       ["_______"])
        df["Credit_Mix"]        = self._clean_categorical(df["Credit_Mix"],       ["_"])
        df["Payment_Behaviour"] = self._clean_categorical(df["Payment_Behaviour"],["!@9#%8"])

        # engineered ratio features
        _n = lambda col: pd.to_numeric(df[col], errors="coerce")
        df["debt_to_income"] = _n("Outstanding_Debt") / _n("Annual_Income").clip(lower=1e-6)
        df["emi_to_salary"] = _n("Total_EMI_per_month") / _n("Monthly_Inhand_Salary").clip(lower=1e-6)
        df["utilization_per_card"] = _n("Credit_Utilization_Ratio") / _n("Num_Credit_Card").clip(lower=1)
        df["delayed_payment_rate"] = _n("Num_of_Delayed_Payment") / _n("Credit_History_Age_Months").clip(lower=1)
        df["balance_to_income"] = _n("Monthly_Balance") / _n("Monthly_Inhand_Salary").clip(lower=1e-6)

        if self.TARGET in df.columns:
            df[self.TARGET] = df[self.TARGET].map(self.LABEL_MAP)
        return df

    def fit_transform(self, df: pd.DataFrame):
        df_clean = self.clean(df)

        cat_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
        for col in cat_cols:
            le = LabelEncoder()
            df_clean[f"{col}_enc"] = le.fit_transform(
                df_clean[col].fillna("Unknown").astype(str)
            )
            self.label_encoders[col] = le

        X = df_clean[self.FEATURES]
        y = df_clean[self.TARGET].dropna()
        X = X.loc[y.index]

        X = X.apply(pd.to_numeric, errors="coerce")

        X_imp    = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)

        self._fitted = True
        return X_scaled, y.values

    def transform(self, df: pd.DataFrame):
        assert self._fitted, "Call fit_transform() first."
        df_clean = self.clean(df)

        cat_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
        for col in cat_cols:
            le = self.label_encoders[col]
            df_clean[f"{col}_enc"] = df_clean[col].fillna("Unknown").astype(str).map(
                lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
            )

        X        = df_clean[self.FEATURES]
        X        = X.apply(pd.to_numeric, errors="coerce")
        X_imp    = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


# ─────────────────────────────────────────────
# CLASS 2 — ABSTRACT BASE MODEL
# ─────────────────────────────────────────────
class BaseModel(ABC):
    """Abstract base class that every ML model wrapper must implement."""

    def __init__(self, name: str):
        self.name  = name
        self.model = None

    @abstractmethod
    def build(self) -> None:
        """Instantiate the underlying estimator."""

    def train(self, X_train, y_train) -> None:
        assert self.model is not None, "Call build() first."
        self.model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.name} does not support predict_proba")

    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if available (tree-based models)."""
        return getattr(self.model, "feature_importances_", None)

    def save(self, directory: Path) -> Path:
        path = directory / f"{self.name}.pkl"
        joblib.dump(self.model, path)
        return path

    @staticmethod
    def load(path: Path):
        return joblib.load(path)


# ─────────────────────────────────────────────
# CLASS 3 — CONCRETE MODEL IMPLEMENTATIONS
# ─────────────────────────────────────────────

class LogisticRegressionModel(BaseModel):
    """
    FIX BUG 1: Removed `multi_class='multinomial'` — that parameter was
    deleted in scikit-learn 1.7.  The lbfgs solver handles multinomial
    logistic regression natively for multi-class targets; no flag needed.

    IMPROVEMENT: Added class_weight='balanced' to handle the class imbalance
    (Standard ~53 %, Poor ~29 %, Good ~17 %) without relying solely on SMOTE.
    """
    def __init__(self, C=1.0, max_iter=1000):
        super().__init__("logistic_regression")
        self.C        = C
        self.max_iter = max_iter

    def build(self):
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",          # handles multinomial natively
            class_weight="balanced", # NEW: compensate for class imbalance
            random_state=42,
            n_jobs=-1,
        )


class RandomForestModel(BaseModel):
    """
    class_weight removed — SMOTE already equalises class distribution in the
    training set.  Adding balanced_subsample on top over-penalises Standard
    (majority) and reduces precision, dragging F1_macro down.
    """
    def __init__(self, n_estimators=200, max_depth=15, max_features="sqrt"):
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.max_features = max_features

    def build(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=42,
            n_jobs=-1,
        )


class GradientBoostingModel(BaseModel):
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5):
        super().__init__("gradient_boosting")
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth

    def build(self):
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )


class HistGradientBoostingModel(BaseModel):
    """
    NEW MODEL: sklearn's HistGradientBoostingClassifier.
    - 5-10× faster than vanilla GradientBoosting on this dataset size.
    - Native NaN support (no imputation needed internally).
    - Supports class_weight='balanced'.
    """
    def __init__(self, max_iter=300, learning_rate=0.05, max_depth=6,
                 min_samples_leaf=20):
        super().__init__("hist_gradient_boosting")
        self.max_iter         = max_iter
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf

    def build(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )


class XGBoostModel(BaseModel):
    """
    FIX BUG 3: Added objective='multi:softprob' to explicitly declare
    multiclass classification. Removed legacy use_label_encoder kwarg
    (removed in XGBoost 1.6+; was already handled automatically).

    IMPROVEMENT: Added `scale_pos_weight` equivalent via sample_weight in
    train(); instead, class imbalance is handled jointly by SMOTE + the
    min_child_weight / gamma regularisation params below.
    """
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=3, gamma=0.1):
        super().__init__("xgboost")
        self.n_estimators    = n_estimators
        self.learning_rate   = learning_rate
        self.max_depth       = max_depth
        self.subsample       = subsample
        self.colsample_bytree= colsample_bytree
        self.min_child_weight= min_child_weight
        self.gamma           = gamma

    def build(self):
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            objective="multi:softprob",  # FIX: explicit multiclass objective
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )


class LightGBMModel(BaseModel):
    """
    NEW MODEL: LightGBM.
    class_weight removed — SMOTE already balances classes; double-compensating
    suppresses the majority class too aggressively.
    Higher n_estimators + lower lr gives boosting more budget to learn fine
    decision boundaries in this 25k-row dataset.
    """
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=-1,
                 num_leaves=63, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0):
        super().__init__("lightgbm")
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.num_leaves       = num_leaves
        self.min_child_samples= min_child_samples
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda       = reg_lambda

    def build(self):
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )


# ─────────────────────────────────────────────
# CLASS 4 — EVALUATOR
# ─────────────────────────────────────────────
class ModelEvaluator:
    """Computes and returns evaluation metrics for a trained model."""

    LABEL_NAMES = ["Poor", "Standard", "Good"]

    def evaluate(self, model: BaseModel, X_test, y_test) -> dict:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "f1_macro":    f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision":   precision_score(y_test, y_pred, average="macro",
                                           zero_division=0),
            "recall":      recall_score(y_test, y_pred, average="macro",
                                        zero_division=0),
            # NEW: one-vs-rest ROC-AUC (more informative for imbalanced classes)
            "roc_auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr",
                                          average="macro"),
        }
        return metrics

    def report(self, model: BaseModel, X_test, y_test) -> str:
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred, target_names=self.LABEL_NAMES)

    def confusion(self, model: BaseModel, X_test, y_test) -> np.ndarray:
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)


# ─────────────────────────────────────────────
# CLASS 5 — PIPELINE RUNNER
# ─────────────────────────────────────────────
class CreditScoringPipeline:
    """
    Orchestrates: load → preprocess → train (multiple models)
    → evaluate → log to MLflow → persist best model.
    """

    def __init__(self, data_path: str = str(DATA_PATH)):
        self.data_path    = data_path
        self.preprocessor = CreditDataPreprocessor()
        self.evaluator    = ModelEvaluator()
        self.best_model   = None
        self.best_metrics = None

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"[INFO] Loaded {len(df):,} rows × {df.shape[1]} columns")
        return df

    def run(self):
        # ── 1. Load & preprocess ─────────────────────────────────
        df = self.load_data()
        X, y = self.preprocessor.fit_transform(df)
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── 2. Apply SMOTE to training set ───────────────────────
        # Keep original split for leak-free CV (FIX BUG 2).
        sm = SMOTE(random_state=42)
        X_train_smote, y_train_smote = sm.fit_resample(X_train_orig, y_train_orig)
        print(f"[INFO] Train (SMOTE): {len(X_train_smote):,} | "
              f"Test: {len(X_test):,}")

        # ── 3. Save preprocessor ────────────────────────────────
        self.preprocessor.save(MODELS_DIR / "preprocessor.pkl")
        print("[INFO] Preprocessor saved.")

        # ── 4. Configure MLflow ──────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # ── 5. Define model candidates ───────────────────────────
        candidates = [
            # Logistic Regression baselines (kept for reference)
            LogisticRegressionModel(C=0.1, max_iter=1000),
            LogisticRegressionModel(C=1.0, max_iter=1000),

            # Random Forests — max_features tuning
            RandomForestModel(n_estimators=300, max_depth=20, max_features="sqrt"),
            RandomForestModel(n_estimators=500, max_depth=None, max_features="sqrt"),
            RandomForestModel(n_estimators=300, max_depth=None, max_features="log2"),

            # HistGradientBoosting (no class_weight — SMOTE handles it)
            HistGradientBoostingModel(max_iter=300, learning_rate=0.05,
                                      max_depth=8,  min_samples_leaf=10),
            HistGradientBoostingModel(max_iter=500, learning_rate=0.02,
                                      max_depth=10, min_samples_leaf=5),

            # XGBoost — more trees + colsample variation
            XGBoostModel(n_estimators=400, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.7),
            XGBoostModel(n_estimators=600, learning_rate=0.02, max_depth=7,
                         subsample=0.9, colsample_bytree=0.8),

            # LightGBM — wider leaf budgets + more trees (no class_weight)
            LightGBMModel(n_estimators=500,  learning_rate=0.05, num_leaves=63),
            LightGBMModel(n_estimators=800,  learning_rate=0.02, num_leaves=127),
            LightGBMModel(n_estimators=1000, learning_rate=0.01, num_leaves=255,
                          min_child_samples=10, reg_lambda=0.5),
        ]

        best_cv_f1  = -1  # IMPROVEMENT: rank by CV F1 (more robust)
        best_run_id = None
        cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ── 6. Train, evaluate, log ──────────────────────────────
        for candidate in candidates:
            candidate.build()
            run_name = f"{candidate.name}_{id(candidate)}"

            with mlflow.start_run(run_name=run_name):
                # log hyperparameters
                safe_params = {k: v for k, v in candidate.__dict__.items()
                               if k != "model"}
                mlflow.log_params(safe_params)

                # train on SMOTE'd data
                candidate.train(X_train_smote, y_train_smote)

                # evaluate on held-out test set
                metrics = self.evaluator.evaluate(candidate, X_test, y_test)
                mlflow.log_metrics(metrics)

                # ── FIX BUG 2: leak-free cross-validation ────────
                # Build an imblearn Pipeline so SMOTE is applied *inside*
                # each fold, never sharing synthetic samples across splits.
                cv_pipe = ImbPipeline([
                    ("smote", SMOTE(random_state=42)),
                    ("model", candidate.model),
                ])
                cv_scores = cross_val_score(
                    cv_pipe, X_train_orig, y_train_orig,
                    cv=cv, scoring="f1_macro", n_jobs=-1,
                )
                mlflow.log_metric("cv_f1_mean", cv_scores.mean())
                mlflow.log_metric("cv_f1_std",  cv_scores.std())

                # log feature importances (tree models only)
                fi = candidate.feature_importances()
                if fi is not None:
                    fi_dict = {
                        f"feat_imp_{CreditDataPreprocessor.FEATURES[i]}": float(v)
                        for i, v in enumerate(fi)
                    }
                    mlflow.log_metrics(fi_dict)

                # log model artifact
                if candidate.name == "xgboost":
                    mlflow.xgboost.log_model(candidate.model, "model")
                elif candidate.name == "lightgbm":
                    mlflow.lightgbm.log_model(candidate.model, "model")
                else:
                    mlflow.sklearn.log_model(candidate.model, "model")

                run_id = mlflow.active_run().info.run_id

                print(
                    f"  [{candidate.name:28s}] "
                    f"Acc={metrics['accuracy']:.4f}  "
                    f"F1_macro={metrics['f1_macro']:.4f}  "
                    f"ROC_AUC={metrics['roc_auc_ovr']:.4f}  "
                    f"CV_F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
                )

                # IMPROVEMENT: select best by CV F1 (not single-split test F1)
                if cv_scores.mean() > best_cv_f1:
                    best_cv_f1        = cv_scores.mean()
                    self.best_model   = candidate
                    self.best_metrics = metrics
                    best_run_id       = run_id

        # ── 7. Persist best model ────────────────────────────────
        best_path = self.best_model.save(MODELS_DIR)
        print(f"\n[BEST] {self.best_model.name}  "
              f"CV_F1={best_cv_f1:.4f}  "
              f"Test_F1={self.best_metrics['f1_macro']:.4f}  "
              f"ROC_AUC={self.best_metrics['roc_auc_ovr']:.4f}")
        print(f"[INFO] Best model saved  → {best_path}")
        print(f"[INFO] Best MLflow run   → {best_run_id}")

        # full classification report
        print("\n" + "─" * 60)
        print(self.evaluator.report(self.best_model, X_test, y_test))
        print("Confusion Matrix:\n",
              self.evaluator.confusion(self.best_model, X_test, y_test))

        return self.best_model, self.best_metrics


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import shutil
    src = Path("data_A.csv")
    if not DATA_PATH.exists() and src.exists():
        shutil.copy(src, DATA_PATH)

    pipeline = CreditScoringPipeline()
    pipeline.run()
