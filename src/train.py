"""Explore data, compare classifiers, evaluate them, and save the best pipeline."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from download_data import download_dataset
from src.common import (
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_COLUMN,
    ZERO_AS_MISSING_COLUMNS,
    InvalidZeroToNaN,
)


RANDOM_STATE = 42


def load_and_explore_data() -> pd.DataFrame:
    """Load and validate the dataset, then print a beginner-friendly overview."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download the Kaggle Pima Indians "
            "Diabetes CSV and place it at data/diabetes.csv."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise")

    if data.empty:
        raise ValueError("The dataset contains no rows.")
    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain missing values encoded as blank.")
    if not np.isfinite(data[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite or invalid numeric values.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values.")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The dataset must contain examples of both target classes.")
    if data[TARGET_COLUMN].value_counts().min() < 5:
        raise ValueError("Each Outcome class needs at least five rows for 5-fold CV.")

    print("\n--- Dataset overview ---")
    print(f"Rows and columns: {data.shape}")
    print("\nFirst five rows:")
    print(data.head())
    print("\nColumn data types and non-null counts:")
    data.info()
    print("\nDescriptive statistics:")
    print(data.describe().round(2))
    print("\nOutcome counts:")
    print(data[TARGET_COLUMN].value_counts().sort_index())
    print("\nOutcome proportions:")
    print(data[TARGET_COLUMN].value_counts(normalize=True).sort_index().round(3))

    print("\nZero values treated as missing by the model pipeline:")
    print(data[ZERO_AS_MISSING_COLUMNS].eq(0).sum())
    print("\nFeature correlations with Outcome (raw values):")
    print(
        data.corr(numeric_only=True)[TARGET_COLUMN]
        .sort_values(ascending=False)
        .round(3)
    )
    print("\nMissing values in source CSV:")
    print(data.isna().sum())
    return data


def build_models() -> dict[str, Pipeline]:
    """Build two comparable pipelines with fold-local preprocessing."""
    logistic_regression = Pipeline(
        steps=[
            ("invalid_zero_to_nan", InvalidZeroToNaN()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("invalid_zero_to_nan", InvalidZeroToNaN()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }


def evaluate_model(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cross_validation: StratifiedKFold,
) -> float:
    """Fit, score on held-out data, and return training-only CV ROC-AUC."""
    cv_auc_scores = cross_val_score(
        model,
        x_train,
        y_train,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=-1,
        error_score="raise",
    )
    if not np.isfinite(cv_auc_scores).all():
        raise ValueError(f"Cross-validation produced an invalid ROC-AUC for {name}.")

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    positive_class_probabilities = model.predict_proba(x_test)[:, 1]

    print(f"\n--- {name} ---")
    print(f"Mean training cross-validation ROC-AUC: {cv_auc_scores.mean():.3f}")
    print(f"Test accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Test precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Test recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Test F1-score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Test ROC-AUC:   {roc_auc_score(y_test, positive_class_probabilities):.3f}")
    print("Test confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    return float(cv_auc_scores.mean())


def main() -> None:
    """Run the complete training workflow and save the CV-selected pipeline."""
    if not DATA_PATH.is_file():
        download_dataset()
    data = load_and_explore_data()
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    cv_auc_by_model = {
        name: evaluate_model(
            name,
            model,
            x_train,
            x_test,
            y_train,
            y_test,
            cross_validation,
        )
        for name, model in models.items()
    }
    best_model_name = max(cv_auc_by_model, key=cv_auc_by_model.get)
    best_model = models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model by mean training CV ROC-AUC: {best_model_name}")
    print(f"Saved trained pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
