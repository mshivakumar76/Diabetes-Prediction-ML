"""Streamlit interface for the trained diabetes prediction pipeline."""

import streamlit as st

from src.common import FEATURE_COLUMNS, MODEL_PATH
from src.predict import make_prediction


st.set_page_config(
    page_title="Diabetes Prediction | ML Demo",
    page_icon="🩺",
    layout="centered",
)

st.title("Diabetes Prediction")
st.markdown(
    "An educational machine-learning demo using the Pima Indians Diabetes "
    "dataset. Enter all eight measurements to see the saved model's output."
)
st.warning(
    "**Medical disclaimer:** This application is for education only. It is "
    "not a medical device, cannot diagnose diabetes, and must not guide "
    "treatment or healthcare decisions. Consult a qualified healthcare "
    "professional about health concerns."
)

with st.expander("About the input values"):
    st.markdown(
        "- Glucose: plasma glucose concentration\n"
        "- Blood pressure: diastolic blood pressure (mm Hg)\n"
        "- Skin thickness: triceps skin fold thickness (mm)\n"
        "- Insulin: two-hour serum insulin (mu U/ml)\n"
        "- BMI: body mass index\n"
        "- Diabetes pedigree function: family-history score in the dataset\n\n"
        "A zero for Glucose, BloodPressure, SkinThickness, Insulin, or BMI is "
        "treated as a missing reading and imputed by the trained pipeline. "
        "Zero pregnancies is a valid value."
    )

if not MODEL_PATH.is_file():
    st.info(
        "No trained model was found. Add the Kaggle CSV at "
        "`data/diabetes.csv`, install the dependencies, and run "
        "`python train.py` to create the model."
    )

with st.form("prediction_form"):
    st.subheader("Patient measurements")
    first_row = st.columns(2)
    pregnancies = first_row[0].number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of pregnancies.",
    )
    glucose = first_row[1].number_input(
        "Glucose (mg/dL)",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        help="Plasma glucose concentration. A zero is treated as missing.",
    )

    second_row = st.columns(2)
    blood_pressure = second_row[0].number_input(
        "Diastolic blood pressure (mm Hg)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        help="A zero is treated as missing.",
    )
    skin_thickness = second_row[1].number_input(
        "Skin thickness (mm)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        help="Triceps skin fold thickness. A zero is treated as missing.",
    )

    third_row = st.columns(2)
    insulin = third_row[0].number_input(
        "Insulin (mu U/ml)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        help="Two-hour serum insulin. A zero is treated as missing.",
    )
    bmi = third_row[1].number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=28.0,
        help="Body mass index. A zero is treated as missing.",
    )

    fourth_row = st.columns(2)
    pedigree = fourth_row[0].number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        help="Family-history score used by the source dataset.",
    )
    age = fourth_row[1].number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=30,
    )
    submitted = st.form_submit_button("Estimate")

if submitted:
    measurements = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }
    if tuple(measurements) != tuple(FEATURE_COLUMNS):
        st.error("Input configuration does not match the model feature order.")
    else:
        try:
            prediction, probability = make_prediction(measurements)
        except FileNotFoundError as error:
            st.error(str(error))
        else:
            st.subheader("Model output")
            if prediction == 1:
                st.error("Model prediction: positive (class 1)")
            else:
                st.success("Model prediction: negative (class 0)")
            st.metric("Estimated probability of class 1", f"{probability:.1%}")
            st.progress(probability)
            st.caption(
                "This estimate reflects this model and dataset only. It is not "
                "a diagnosis or a substitute for clinical evaluation."
            )
