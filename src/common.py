"""Shared dataset schema and preprocessing used by training and inference."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "diabetes.csv"
MODEL_PATH = PROJECT_DIR / "models" / "diabetes_model.pkl"

TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Zero is a valid pregnancy count, but not a meaningful value for these measurements.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


class InvalidZeroToNaN(BaseEstimator, TransformerMixin):
    """Treat impossible zero measurements as missing in both training and prediction."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "InvalidZeroToNaN":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InvalidZeroToNaN expects a pandas DataFrame.")
        missing_columns = set(ZERO_AS_MISSING_COLUMNS).difference(X.columns)
        if missing_columns:
            raise ValueError(
                "Input is missing measurement columns: "
                + ", ".join(sorted(missing_columns))
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InvalidZeroToNaN expects a pandas DataFrame.")
        missing_columns = set(ZERO_AS_MISSING_COLUMNS).difference(X.columns)
        if missing_columns:
            raise ValueError(
                "Input is missing measurement columns: "
                + ", ".join(sorted(missing_columns))
            )

        transformed = X.copy()
        transformed[ZERO_AS_MISSING_COLUMNS] = transformed[
            ZERO_AS_MISSING_COLUMNS
        ].replace(0, np.nan)
        return transformed
