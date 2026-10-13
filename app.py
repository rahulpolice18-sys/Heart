import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="HeartCare AI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #f3f8ff 0%, #ffffff 55%, #f4f8ff 100%);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .brand {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1261a6;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    .hero {
        background: linear-gradient(120deg, #073b70 0%, #0878b9 55%, #21a5ce 100%);
        color: white;
        border-radius: 24px;
        padding: 34px 36px;
        box-shadow: 0 14px 35px rgba(12, 82, 142, 0.18);
        margin-bottom: 25px;
    }

    .hero-label {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 7px 13px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 800;
        line-height: 1.12;
        margin: 0 0 15px 0;
        color: white;
    }

    .hero p {
        color: #e2f3ff;
        font-size: 1rem;
        line-height: 1.8;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #103a64;
        margin: 12px 0 4px 0;
    }

    .section-subtitle {
        color: #62758a;
        margin-bottom: 20px;
        font-size: 0.94rem;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #dce9f7;
        box-shadow: 0 8px 25px rgba(25, 74, 120, 0.06);
    }

    div[data-testid="stForm"] label {
        color: #244663;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #0878b9, #0757a3);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 3.2rem;
        font-size: 1rem;
        font-weight: 700;
        transition: 0.2s;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #0757a3, #063e7a);
        color: white;
        border: none;
    }

    .result-card {
        background: white;
        border: 1px solid #dce9f7;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(25, 74, 120, 0.06);
        margin-top: 18px;
    }

    .result-positive {
        background: #fff3f2;
        border-left: 6px solid #df4545;
        border-radius: 12px;
        padding: 20px;
    }

    .result-negative {
        background: #edfbf4;
        border-left: 6px solid #159967;
        border-radius: 12px;
        padding: 20px;
    }

    .result-heading {
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .metric-card {
        background: white;
        border: 1px solid #dce9f7;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        height: 100%;
    }

    .metric-number {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0878b9;
    }

    .metric-label {
        color: #647b91;
        font-size: 0.82rem;
        margin-top: 5px;
    }

    .info-card {
        background: #eaf5ff;
        border: 1px solid #cde6fb;
        border-radius: 15px;
        padding: 19px;
        color: #214a6c;
        margin-top: 18px;
        line-height: 1.7;
    }

    .footer {
        text-align: center;
        color: #73869a;
        font-size: 0.82rem;
        padding-top: 30px;
        line-height: 1.8;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 25px 22px;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "heart_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    selector = joblib.load(BASE_DIR / "selector.pkl")
    return model, scaler, selector


try:
    model, scaler, selector = load_models()
except Exception:
    st.error(
        "The prediction model could not be loaded. "
        "Please check that heart_model.pkl, scaler.pkl, "
        "and selector.pkl are uploaded correctly."
    )
    st.stop()


# --------------------------------------------------
# BRAND HEADER
# --------------------------------------------------
st.markdown(
    '<div class="brand">♥ HEARTCARE <span style="color:#71869b;'
    'font-weight:500;">AI HEALTH TECHNOLOGY</span></div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# HERO SECTION WITH YOUR IMAGE
# --------------------------------------------------
hero_left, hero_right = st.columns([1.05, 0.95], gap="large")

with hero_left:
    st.markdown("""
    <div class="hero">
        <div class="hero-label">AI-POWERED HEALTH ANALYSIS</div>
        <h1>Understand Your Heart.<br>Protect Your Future.</h1>
        <p>
            Explore a machine-learning-based heart disease prediction
            system using clinical measurements. Enter the available
            patient details to generate an educational prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    image_path = BASE_DIR / "heart_banner.png"

    if image_path.exists():
        st.image(
            str(image_path),
            use_container_width=True
        )
    else:
        st.warning(
            "Upload heart_banner.png to your GitHub repository "
            "to display the heart image."
        )


# --------------------------------------------------
# QUICK OVERVIEW CARDS
# --------------------------------------------------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">AI + ML</div>
        <div class="metric-label">Prediction technology</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">13</div>
        <div class="metric-label">Clinical input features</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">Instant</div>
        <div class="metric-label">Model output</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# PATIENT INPUT SECTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Patient Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter the clinical measurements '
    'below. Use the same units and feature definitions as your '
    'training dataset.</div>',
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
            options=[1, 0],
            format_func=lambda x: "Male (1)" if x == 1 else "Female (0)"
        )

    with col3:
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50, max_value=250, value=120, step=1
        )

    st.markdown("#### 02 — Blood Test Measurements")

    col4, col5, col6 = st.columns(3)

    with col4:
        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=50, max_value=700, value=200, step=1
        )

    with col5:
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

    with col6:
        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=40, max_value=250, value=150, step=1
        )

    st.markdown("#### 03 — Heart Examination Details")

    col7, col8, col9 = st.columns(3)

    with col7:
        cp = st.selectbox(
            "Chest pain type (cp)",
            options=[0, 1, 2, 3],
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
            options=[0, 1, 2]
        )

    with col9:
        exang = st.selectbox(
            "Exercise-induced angina",
            options=[0, 1],
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
            options=[0, 1, 2]
        )

    with col12:
        ca = st.selectbox(
            "Major vessels (ca)",
            options=[0, 1, 2, 3, 4]
        )

    st.markdown("#### 04 — Additional Clinical Information")

    col13, col14 = st.columns(2)

    with col13:
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 — Normal",
                2: "2 — Fixed defect",
                3: "3 — Reversible defect"
            }[x]
        )

    with col14:
        st.info(
            "Check the category encodings against the dataset "
            "used to train your model before interpreting results."
        )

    submitted = st.form_submit_button(
        "Analyze Measurements",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
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

        prediction = int(model.predict(scaled_data)[0])

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.markdown("""
            <div class="result-card">
                <div class="result-positive">
                    <div class="result-heading">
                        Model indicates a potential risk
                    </div>
                    The model classified this input as class 1.
                    This is not a diagnosis. Consult a qualified
                    healthcare professional for medical assessment.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif prediction == 0:
            st.markdown("""
            <div class="result-card">
                <div class="result-negative">
                    <div class="result-heading">
                        Model indicates a lower predicted risk
                    </div>
                    The model classified this input as class 0.
                    This does not rule out heart disease or replace
                    professional medical evaluation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning(
                f"The model returned class {prediction}. "
                "Verify the target-label mapping in your training code."
            )

        # Show probability only if the model supports it.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                risk_probability = float(
                    probabilities[classes.index(1)] * 100
                )

                st.markdown("#### Model output probability")

                st.progress(
                    int(round(risk_probability)),
                    text=f"Probability of class 1: {risk_probability:.1f}%"
                )

                st.caption(
                    "This is the model's estimated probability for class 1, "
                    "not a clinically validated estimate of your personal "
                    "risk. Its meaning depends on the training labels and "
                    "model calibration."
                )

        st.markdown("""
        <div class="info-card">
            <strong>Important:</strong> This application is an educational
            machine-learning project. It cannot diagnose, prevent, or treat
            heart disease. If you have symptoms or concerns, seek medical
            advice from a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)

    except Exception as error:
        st.error(
            "Prediction failed. Please verify the input feature order, "
            "category encodings, and compatibility of the saved model "
            "files with the training pipeline."
        )
        st.caption(f"Error details: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    <strong>HEARTCARE AI</strong><br>
    Machine Learning • Clinical Data • Educational Project<br>
    This tool is not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)
