
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="HeartCare AI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #edf6ff 0%, #ffffff 48%, #f3f8ff 100%);
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    color: #0865a5;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

.hero {
    background: linear-gradient(125deg, #063767, #087db8 65%, #20a8c9);
    padding: 35px;
    border-radius: 23px;
    color: white;
    min-height: 310px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 12px 30px rgba(6, 73, 125, 0.18);
}

.hero-tag {
    display: inline-block;
    color: #e4f6ff;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 0.76rem;
    font-weight: 700;
    width: fit-content;
    margin-bottom: 16px;
}

.hero h1 {
    color: white;
    font-size: clamp(2rem, 3.3vw, 3rem);
    line-height: 1.15;
    font-weight: 800;
    margin: 0 0 15px 0;
}

.hero p {
    color: #e5f4ff;
    font-size: 0.96rem;
    line-height: 1.8;
    margin: 0;
}

.image-frame {
    background: white;
    padding: 10px;
    border: 1px solid #d7e8f8;
    border-radius: 23px;
    box-shadow: 0 12px 30px rgba(6, 73, 125, 0.12);
    overflow: hidden;
}

.section-title {
    color: #103e68;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #667d92;
    margin-bottom: 20px;
    line-height: 1.7;
}

.metric-card {
    background: white;
    border: 1px solid #dceafa;
    border-radius: 16px;
    padding: 20px 12px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(20, 73, 120, 0.05);
}

.metric-value {
    color: #0878b9;
    font-size: 1.5rem;
    font-weight: 800;
}

.metric-label {
    color: #687e92;
    font-size: 0.82rem;
    margin-top: 5px;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #dceafa;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 7px 25px rgba(20, 73, 120, 0.06);
}

div[data-testid="stForm"] label {
    color: #244663;
    font-weight: 600;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #087db8, #0756a1);
    color: white;
    border: 0;
    border-radius: 12px;
    min-height: 3.2rem;
    font-weight: 700;
    font-size: 1rem;
}

div[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: 0;
    background: linear-gradient(90deg, #07659e, #063e7a);
}

.result-card {
    background: white;
    border: 1px solid #dceafa;
    border-radius: 18px;
    padding: 24px;
    margin-top: 18px;
    box-shadow: 0 7px 25px rgba(20, 73, 120, 0.06);
}

.result-positive {
    background: #fff1f1;
    border-left: 6px solid #d94343;
    border-radius: 12px;
    padding: 20px;
    color: #782626;
}

.result-negative {
    background: #eafaf1;
    border-left: 6px solid #159765;
    border-radius: 12px;
    padding: 20px;
    color: #145b3b;
}

.result-heading {
    font-size: 1.25rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.notice {
    background: #eaf5ff;
    border: 1px solid #cce5fb;
    border-radius: 14px;
    padding: 18px;
    color: #244b6c;
    line-height: 1.8;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #71869a;
    font-size: 0.82rem;
    line-height: 1.8;
    margin-top: 35px;
}

.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED MODEL FILES
# ==================================================
@st.cache_resource
def load_models():
    model_path = BASE_DIR / "heart_model.pkl"
    scaler_path = BASE_DIR / "scaler.pkl"
    selector_path = BASE_DIR / "selector.pkl"

    for path in [model_path, scaler_path, selector_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required model file is missing: {path.name}"
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    return model, scaler, selector


try:
    model, scaler, selector = load_models()
except Exception:
    st.error(
        "The prediction system could not load its model files. "
        "Please check that heart_model.pkl, scaler.pkl, and "
        "selector.pkl are uploaded correctly and are compatible "
        "with the installed Python libraries."
    )
    st.stop()


# ==================================================
# HEADER
# ==================================================
st.markdown(
    '<div class="brand">♥ HEARTCARE AI '
    '<span style="color:#6b8197;font-weight:500;">'
    '| INTELLIGENT HEALTH ANALYSIS</span></div>',
    unsafe_allow_html=True
)


# ==================================================
# HERO SECTION AND HEART IMAGE
# ==================================================
hero_col, image_col = st.columns([1.05, 0.95], gap="large")

with hero_col:
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">AI-POWERED HEALTH TECHNOLOGY</div>
        <h1>Understand Your Heart.<br>Protect Your Future.</h1>
        <p>
            Explore a machine-learning-based heart disease prediction
            system. Enter clinical measurements to see how the trained
            model classifies the provided data.
        </p>
    </div>
    """, unsafe_allow_html=True)

with image_col:
    image_path = BASE_DIR / "heart_banner.png"

if image_path.is_file():
    st.image(
        str(image_path),
        caption="Understanding the human heart",
        use_column_width=True
    )
else:
    st.warning(
        "Image not found. Upload heart_banner.png "
        "beside app.py in your GitHub repository."
    )

# ==================================================
# OVERVIEW CARDS
# ==================================================
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">AI + ML</div>
        <div class="metric-label">Machine learning technology</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">13</div>
        <div class="metric-label">Clinical input features</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">Real-time</div>
        <div class="metric-label">Prediction after submission</div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# INPUT FORM
# ==================================================
st.markdown(
    '<div class="section-title">Patient Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter all required measurements '
    'using the same units, category codes, and feature definitions '
    'used to train your model.</div>',
    unsafe_allow_html=True
)

with st.form("heart_prediction_form"):

    st.markdown("#### 01 — Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120,
            value=45, step=1
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            [1, 0],
            format_func=lambda x: "Male (1)" if x == 1 else "Female (0)"
        )

    with col3:
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50, max_value=250, value=120
        )

    st.markdown("#### 02 — Blood Measurements")

    col4, col5, col6 = st.columns(3)

    with col4:
        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=50, max_value=700, value=200
        )

    with col5:
        fbs = st.selectbox(
            "Fasting blood sugar above 120 mg/dl",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

    with col6:
        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=40, max_value=250, value=150
        )

    st.markdown("#### 03 — Heart Examination")

    col7, col8, col9 = st.columns(3)

    with col7:
        cp = st.selectbox(
            "Chest pain type (cp)",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Typical angina",
                1: "1 — Atypical angina",
                2: "2 — Non-anginal pain",
                3: "3 — Asymptomatic"
            }[x]
        )

    with col8:
        restecg = st.selectbox(
            "Resting ECG result",
            [0, 1, 2]
        )

    with col9:
        exang = st.selectbox(
            "Exercise-induced angina",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

    col10, col11, col12 = st.columns(3)

    with col10:
        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0, max_value=10.0,
            value=1.0, step=0.1, format="%.1f"
        )

    with col11:
        slope = st.selectbox(
            "ST segment slope",
            [0, 1, 2]
        )

    with col12:
        ca = st.selectbox(
            "Major vessels (ca)",
            [0, 1, 2, 3, 4]
        )

    st.markdown("#### 04 — Additional Information")

    thal = st.selectbox(
        "Thalassemia category (thal)",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1 — Normal",
            2: "2 — Fixed defect",
            3: "3 — Reversible defect"
        }[x]
    )

    submitted = st.form_submit_button(
        "♥  Analyze Measurements",
        use_container_width=True
    )


# ==================================================
# RUN PREDICTION
# ==================================================
if submitted:

    feature_names = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    input_data = pd.DataFrame(
        [[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]],
        columns=feature_names
    )

    try:
        selected_data = selector.transform(input_data)
        scaled_data = scaler.transform(selected_data)
        prediction = model.predict(scaled_data)[0]

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.markdown("""
            <div class="result-card">
                <div class="result-positive">
                    <div class="result-heading">
                        Potential Risk Indicated by the Model
                    </div>
                    The model classified the entered measurements as
                    class 1. This is a machine-learning output, not a
                    medical diagnosis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif prediction == 0:
            st.markdown("""
            <div class="result-card">
                <div class="result-negative">
                    <div class="result-heading">
                        Lower-Risk Class Predicted by the Model
                    </div>
                    The model classified the entered measurements as
                    class 0. This result does not rule out heart disease.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning(
                f"The model returned class {prediction}. "
                "Verify the target labels in the training dataset."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)] * 100
                )

                st.markdown("#### Model probability for class 1")

                st.progress(
                    int(np.clip(round(probability), 0, 100)),
                    text=f"Class 1 probability: {probability:.1f}%"
                )

                st.caption(
                    "This is the model's estimated probability for "
                    "class 1, not a clinically validated estimate of "
                    "personal heart disease risk."
                )

        st.markdown("""
        <div class="notice">
            <strong>Medical disclaimer</strong><br>
            This application is intended for educational and
            demonstration purposes only. It must not be used to
            diagnose or treat heart disease. Consult a qualified
            healthcare professional for medical advice.
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        st.error(
            "Prediction could not be completed. Verify that the "
            "feature order, category encodings, selector, scaler, "
            "and model match the original training pipeline."
        )


# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    <strong>HEARTCARE AI</strong><br>
    Machine Learning · Clinical Data · Educational Project<br>
    Not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)

