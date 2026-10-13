import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "heart_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SELECTOR_PATH = BASE_DIR / "selector.pkl"
BANNER_PATH = BASE_DIR / "heart_banner.png"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("Heart Disease Prediction System")
st.write("Enter the clinical measurements below to obtain a machine-learning prediction.")
st.caption("Educational project only. This result is not a medical diagnosis.")

missing = [p.name for p in (MODEL_PATH, SCALER_PATH, SELECTOR_PATH) if not p.exists()]
if missing:
    st.error("Required model file(s) not found: " + ", ".join(missing))
    st.info("Add heart.csv to this folder and run `python train_model.py` to generate the model files. Then upload the generated .pkl files to your repository.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.stop()

st.subheader("Patient Information")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=52)
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest pain type (cp)", options=[("Typical angina", 0), ("Atypical angina", 1), ("Non-anginal pain", 2), ("Asymptomatic", 3)], format_func=lambda x: x[0])[1]
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=125)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=50, max_value=700, value=212)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox("Resting ECG result", options=[("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)], format_func=lambda x: x[0])[1]
    with c2:
        thalach = st.number_input("Maximum heart rate achieved", min_value=50, max_value=250, value=168)
        exang = st.selectbox("Exercise-induced angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
        ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3, 4], index=2)
        thal = st.selectbox("Thalassemia result (thal)", options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)], format_func=lambda x: x[0])[1]

    submitted = st.form_submit_button("Predict")

if submitted:
    feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    input_df = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal
    ]], columns=feature_names)

    try:
        selected = selector.transform(input_df)
        scaled = scaler.transform(selected)
        prediction = int(model.predict(scaled)[0])

        st.subheader("Prediction Result")
        if prediction == 1:
            st.warning("The model predicts a positive heart-disease indicator.")
        else:
            st.success("The model predicts a negative heart-disease indicator.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled)[0]
            st.write(f"Model probability for class 1: **{probabilities[1] * 100:.1f}%**")
        st.caption("This prediction is for demonstration only. Consult a qualified healthcare professional for medical evaluation.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
