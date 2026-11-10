"""Reusable prediction logic and command-line interface."""

import argparse
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from src.common import FEATURE_COLUMNS, MODEL_PATH


def parse_arguments(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse patient measurements from command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate diabetes status from the eight Pima Indians dataset "
            "features. Educational use only; not a medical diagnosis."
        )
    )
    parser.add_argument("--pregnancies", type=int, required=True, help="Number of pregnancies.")
    parser.add_argument("--glucose", type=float, required=True, help="Plasma glucose concentration.")
    parser.add_argument("--blood-pressure", type=float, required=True, help="Diastolic blood pressure.")
    parser.add_argument("--skin-thickness", type=float, required=True, help="Triceps skin fold thickness.")
    parser.add_argument("--insulin", type=float, required=True, help="Two-hour serum insulin.")
    parser.add_argument("--bmi", type=float, required=True, help="Body mass index.")
    parser.add_argument(
        "--diabetes-pedigree-function",
        type=float,
        required=True,
        help="Diabetes pedigree function value.",
    )
    parser.add_argument("--age", type=int, required=True, help="Age in years.")
    return parser.parse_args(args)


def make_prediction(measurements: dict[str, int | float]) -> tuple[int, float]:
    """Load the saved pipeline and return class 1 probability and class prediction."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run python train.py first."
        )

    input_data = pd.DataFrame([measurements], columns=FEATURE_COLUMNS)
    if input_data.isna().any().any():
        raise ValueError("Provide a numeric value for each of the eight measurements.")
    values = input_data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Measurements must be finite numeric values.")
    if (values < 0).any():
        raise ValueError("Measurements cannot be negative.")

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("The saved model must support class and probability predictions.")

    class_one_positions = np.flatnonzero(model.classes_ == 1)
    if len(class_one_positions) != 1:
        raise ValueError("The saved model does not have a unique positive class (1).")

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, class_one_positions[0]])
    return prediction, probability


def main(args: Sequence[str] | None = None) -> None:
    """Run a prediction from command-line feature values."""
    arguments = parse_arguments(args)
    measurements = {
        "Pregnancies": arguments.pregnancies,
        "Glucose": arguments.glucose,
        "BloodPressure": arguments.blood_pressure,
        "SkinThickness": arguments.skin_thickness,
        "Insulin": arguments.insulin,
        "BMI": arguments.bmi,
        "DiabetesPedigreeFunction": arguments.diabetes_pedigree_function,
        "Age": arguments.age,
    }

    prediction, probability = make_prediction(measurements)
    label = "positive" if prediction == 1 else "negative"
    print(f"Model prediction: {label} (class {prediction})")
    print(f"Estimated probability of class 1: {probability:.1%}")
    print(
        "This educational model is not a medical diagnosis. "
        "Consult a qualified healthcare professional."
    )


if __name__ == "__main__":
    main()
