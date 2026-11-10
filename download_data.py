"""Download and validate the original Pima Indians Diabetes CSV."""

from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_PATH = DATA_DIR / "diabetes.csv"
EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# The first source includes headers. The fallback is the same public Pima
# dataset in the original headerless UCI field order.
DATASET_SOURCES = (
    (
        "Plotly public Pima dataset mirror",
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        True,
    ),
    (
        "Jason Brownlee public UCI Pima dataset mirror",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv",
        False,
    ),
)


def validate_dataset(content: bytes, has_header: bool) -> pd.DataFrame:
    """Parse and validate a downloaded source before saving it."""
    data = pd.read_csv(BytesIO(content), header=0 if has_header else None)

    if not has_header:
        if data.shape[1] != len(EXPECTED_COLUMNS):
            raise ValueError(
                f"Expected {len(EXPECTED_COLUMNS)} columns in headerless data, "
                f"found {data.shape[1]}."
            )
        data.columns = EXPECTED_COLUMNS

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "CSV columns did not match the expected Pima schema. "
            f"Found: {list(data.columns)}"
        )
    if len(data) != 768:
        raise ValueError(f"Expected 768 original records, found {len(data)}.")

    for column in EXPECTED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or non-finite numeric values.")
    if not set(data["Outcome"].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only binary values 0 and 1.")
    if set(data["Outcome"].unique()) != {0, 1}:
        raise ValueError("Dataset must include records from both Outcome classes.")
    return data


def download_dataset() -> Path:
    """Use an existing valid CSV, or download from a validated public source."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        validate_dataset(DATA_PATH.read_bytes(), has_header=True)
        print(f"Dataset already exists and is valid; leaving it unchanged: {DATA_PATH}")
        return DATA_PATH

    failures: list[str] = []
    for source_name, url, has_header in DATASET_SOURCES:
        try:
            request = Request(
                url,
                headers={"User-Agent": "DiabetesPredictionML/1.0"},
            )
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise URLError(f"HTTP status {response.status}")
                content = response.read()

            data = validate_dataset(content, has_header=has_header)
            temporary_path = DATA_PATH.with_suffix(".csv.tmp")
            try:
                data.to_csv(temporary_path, index=False)
                # Check the exact file representation before making it visible.
                validate_dataset(temporary_path.read_bytes(), has_header=True)
                temporary_path.replace(DATA_PATH)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

            print(f"Downloaded and verified 768 records from {source_name}.")
            print(f"Saved dataset to: {DATA_PATH}")
            print(f"Verified columns: {', '.join(data.columns)}")
            return DATA_PATH
        except (OSError, URLError, ValueError, pd.errors.ParserError) as error:
            failures.append(f"{source_name}: {error}")
            print(f"Could not use {source_name}: {error}")

    raise RuntimeError(
        "Unable to download and validate the Pima dataset from any configured "
        "public source:\n- " + "\n- ".join(failures)
    )


if __name__ == "__main__":
    download_dataset()
