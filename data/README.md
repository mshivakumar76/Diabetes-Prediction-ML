# Dataset

Run `python download_data.py` from the project root. The script downloads the
public Pima Indians Diabetes CSV, validates it, and saves it in this folder as
`diabetes.csv`. Training also invokes the downloader automatically if the
dataset is missing.

The CSV must include the standard features `Pregnancies`, `Glucose`,
`BloodPressure`, `SkinThickness`, `Insulin`, `BMI`,
`DiabetesPedigreeFunction`, and `Age`, plus the binary `Outcome` column.
`Outcome` values must be `0` or `1`.

The data file is excluded by `.gitignore`; each user should obtain the source
dataset directly rather than committing a copy. Kaggle source:
[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
