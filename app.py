import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import hashlib
import traceback
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "heart_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SELECTOR_PATH = BASE_DIR / "selector.pkl"

MODEL_FILES = {
    "Model": MODEL_PATH,
    "Scaler": SCALER_PATH,
    "Selector": SELECTOR_PATH,
}

# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

st.title("Heart Disease Prediction System")
st.write(
    "Enter the clinical measurements below to obtain "
    "a machine-learning prediction."
)
st.caption(
    "Educational project only. This result is not a medical diagnosis."
)

# --------------------------------------------------
# 3. CHECK REQUIRED FILES
# --------------------------------------------------
missing = [
    name for name, path in MODEL_FILES.items()
    if not path.is_file()
]

if missing:
    st.error("Missing model file(s): " + ", ".join(missing))
    st.info(
        "Upload heart_model.pkl, scaler.pkl, and selector.pkl "
        "to the root of your GitHub repository."
    )
    st.stop()

# --------------------------------------------------
# 4. DIAGNOSTIC INFORMATION
# --------------------------------------------------
with st.expander("Diagnostic information", expanded=True):
    st.write("Python package versions")

    st.write("scikit-learn:", sklearn.__version__)
    st.write("NumPy:", np.__version__)
    st.write("Pandas:", pd.__version__)
    st.write("Joblib:", joblib.__version__)

    st.subheader("Model file verification")

    for name, path in MODEL_FILES.items():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            st.write(f"{name} file size: {path.stat().st_size:,} bytes")
            st.write(f"{name} SHA256 fingerprint:")
            st.code(digest)

        except Exception as exc:
            st.error(f"Could not inspect {name}: {repr(exc)}")

# --------------------------------------------------
# 5. LOAD MODEL FILES INDIVIDUALLY
# --------------------------------------------------
loaded_objects = {}

for name, path in MODEL_FILES.items():
    try:
        loaded_objects[name] = joblib.load(path)

        st.success(
            f"{name} loaded successfully: "
            f"{type(loaded_objects[name]).__name__}"
        )

    except Exception as exc:
        st.error(f"{name} loading failed")

        st.code(
            f"Error type: {type(exc).__name__}\n"
            f"Error details: {repr(exc)}"
        )

        st.subheader("Full error traceback")
        st.code(traceback.format_exc())

        st.stop()

model = loaded_objects["Model"]
scaler = loaded_objects["Scaler"]
selector = loaded_objects["Selector"]

# --------------------------------------------------
# 6. PATIENT INFORMATION FORM
# --------------------------------------------------
st.subheader("Patient Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=52,
        )

        sex_label = st.selectbox(
            "Sex",
            ["Female", "Male"],
        )
        sex = 0 if sex_label == "Female" else 1

        cp_options = [
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic",
        ]
        cp_label = st.selectbox("Chest pain type (cp)", cp_options)
        cp = cp_options.index(cp_label)

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=125,
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=50,
            max_value=700,
            value=212,
        )

        fbs_label = st.selectbox(
            "Fasting blood sugar > 120 mg/dl",
            ["No", "Yes"],
        )
        fbs = 0 if fbs_label == "No" else 1

        restecg_options = [
            "Normal",
            "ST-T wave abnormality",
            "Left ventricular hypertrophy",
        ]
        restecg_label = st.selectbox(
            "Resting ECG result",
            restecg_options,
        )
        restecg = restecg_options.index(restecg_label)

    with col2:
        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=50,
            max_value=250,
            value=168,
        )

        exang_label = st.selectbox(
            "Exercise-induced angina",
            ["No", "Yes"],
        )
        exang = 0 if exang_label == "No" else 1

        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        slope_options = [
            "Upsloping",
            "Flat",
            "Downsloping",
        ]
        slope_label = st.selectbox(
            "Slope of peak exercise ST segment",
            slope_options,
        )
        slope = slope_options.index(slope_label)

        ca = st.selectbox(
            "Number of major vessels (ca)",
            [0, 1, 2, 3, 4],
            index=2,
        )

        thal_options = [
            "Normal",
            "Fixed defect",
            "Reversible defect",
        ]
        thal_label = st.selectbox(
            "Thalassemia result (thal)",
            thal_options,
        )
        thal = thal_options.index(thal_label) + 1

    submitted = st.form_submit_button("Predict")

# --------------------------------------------------
# 7. MAKE PREDICTION
# --------------------------------------------------
if submitted:
    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    input_df = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]],
        columns=feature_names,
    )

    try:
        # Apply preprocessing in the training order.
        selected_features = selector.transform(input_df)
        scaled_features = scaler.transform(selected_features)

        prediction = int(model.predict(scaled_features)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model predicts a positive heart-disease indicator."
            )
        else:
            st.success(
                "The model predicts a negative heart-disease indicator."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_features)[0]
            classes = list(model.classes_)

            if 1 in classes:
                positive_index = classes.index(1)
                probability = probabilities[positive_index] * 100

                st.write(
                    f"Model probability for class 1: **{probability:.1f}%**"
                )

        st.caption(
            "For educational demonstration only. This prediction is not "
            "a diagnosis or a substitute for professional medical evaluation."
        )

    except Exception:
        st.error("Prediction failed.")

        st.code(traceback.format_exc())
