# app.py — Streamlit Diabetes Prediction (Styled, Safe, Final)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Load model & scaler
# ---------------------------
model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# CSS / Theming (glass + accent colors)
# Keep emoji separate: emoji will not inherit gradient color
# ---------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(180deg, #f6f9ff 0%, #f1f6ff 100%);
        color: #0b2545;
    }

    /* Title area */
    .title-row {
        display:flex;
        align-items:center;
        gap:14px;
        justify-content:center;
        margin-bottom: 8px;
    }
    .title-text {
        font-size:34px;
        font-weight:800;
        color:#0b3d91;
        margin:0;
    }
    .title-sub {
        color:#274b8d;
        margin-top:4px;
        margin-bottom:22px;
    }

    /* Card */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.95));
        border-radius:14px;
        padding:18px;
        box-shadow: 0 8px 24px rgba(9,30,66,0.08);
        border: 1px solid rgba(33, 150, 243, 0.08);
    }

    /* Small cards row */
    .row-cards {
        display:flex;
        gap:16px;
        flex-wrap:wrap;
    }
    .small-card {
        flex:1;
        min-width:220px;
        padding:12px;
        border-radius:12px;
        background: white;
        box-shadow: 0 6px 16px rgba(16,24,40,0.06);
        border:1px solid rgba(16,24,40,0.03);
    }

    /* Prediction box */
    .prediction-box {
        border-radius:12px;
        padding:18px;
        color: white;
        text-align:center;
        font-weight:800;
        font-size:20px;
        background: linear-gradient(90deg,#1565c0,#42a5f5);
        box-shadow: 0 8px 20px rgba(21,101,192,0.18);
    }

    /* Sidebar header */
    .sidebar .stRadio > div { margin-top: 8px; }

    /* Help small text */
    .help-text { color: #5b6b8a; font-size:13px; margin-top:6px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Feature descriptions and mapping
# ---------------------------
feature_info = {
    "HighBP": "High blood pressure. 0 = No, 1 = Yes",
    "HighChol": "High cholesterol. 0 = No, 1 = Yes",
    "CholCheck": "Had cholesterol check within past 5 years. 0 = No, 1 = Yes",
    "BMI": "Body Mass Index (kg/m²). Normal: 18.5–24.9",
    "Smoker": "Smoked ≥100 cigarettes in lifetime. 0 = No, 1 = Yes",
    "Stroke": "Ever told you had a stroke. 0 = No, 1 = Yes",
    "HeartDiseaseorAttack": "History of heart disease / attack. 0 = No, 1 = Yes",
    "PhysActivity": "Physical activity in past 30 days (not job). 0 = No, 1 = Yes",
    "Fruits": "Consumes fruit ≥1/day. 0 = No, 1 = Yes",
    "Veggies": "Consumes vegetables ≥1/day. 0 = No, 1 = Yes",
    "HvyAlcoholConsump": "Heavy alcohol use (men >14/wk, women >7/wk). 0 = No, 1 = Yes",
    "DiffWalk": "Difficulty walking/climbing stairs. 0 = No, 1 = Yes",
    "Sex": "Female or Male (Female -> 0, Male -> 1)",
    "Age": (
        "Age categories (BRFSS):\n"
        "1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44,\n"
        "6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69,\n"
        "11=70–74, 12=75–79, 13=80+"
    ),
    "GenHlth": "General health: 1=Excellent → 5=Poor"
}

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("", ["BMI Calculator", "Input & Predict"])

# ---------------------------
# Title (emoji separate from gradient text)
# ---------------------------
st.markdown("<div class='title-row'>"
            "<div style='font-size:42px;'>🩺</div>"
            "<div style='text-align:left;'>"
            "<div class='title-text'>Diabetes Prediction</div>"
            "<div class='title-sub'>BRFSS 2015 — ML prediction (XGBoost)</div>"
            "</div>"
            "</div>", unsafe_allow_html=True)

# ---------------------------
# PAGE: BMI Calculator (vertical)
# ---------------------------
if page == "BMI Calculator":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📏 BMI Calculator", unsafe_allow_html=True)

    # vertical layout: inputs stacked nicely
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5, help="Enter weight in kilograms")
    height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5, help="Enter height in centimeters")

    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"<div style='margin-top:12px;'><strong>Your BMI:</strong> <span style='font-size:18px; color:#0b3d91'>{bmi:.2f}</span></div>", unsafe_allow_html=True)

    if bmi < 18.5:
        st.warning("Underweight — consider consulting nutrition guidance.")
    elif bmi < 25:
        st.success("Normal weight — keep maintaining a healthy lifestyle.")
    elif bmi < 30:
        st.info("Overweight — consider diet and exercise adjustments.")
    else:
        st.error("Obese — consider medical/nutritional consultation.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# PAGE: Input & Predict
# ---------------------------
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Patient inputs", unsafe_allow_html=True)

    # helper convertor
    def yesno_to_int(choice: str) -> int:
        return 1 if choice == "Yes" else 0

    # layout two columns
    left_col, right_col = st.columns(2)

    with left_col:
        HighBP = yesno_to_int(st.selectbox("High blood pressure?", options=["No", "Yes"], index=0, help=feature_info["HighBP"]))
        HighChol = yesno_to_int(st.selectbox("High cholesterol?", options=["No", "Yes"], index=0, help=feature_info["HighChol"]))
        CholCheck = yesno_to_int(st.selectbox("Cholesterol checked in last 5 years?", ["No", "Yes"], index=0, help=feature_info["CholCheck"]))
        BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=25.0, step=0.1, help=feature_info["BMI"])
        Smoker = yesno_to_int(st.selectbox("Smoked 100+ cigarettes ever?", ["No", "Yes"], index=0, help=feature_info["Smoker"]))
        Stroke = yesno_to_int(st.selectbox("Ever had a stroke?", ["No", "Yes"], index=0, help=feature_info["Stroke"]))

    with right_col:
        HeartDiseaseorAttack = yesno_to_int(st.selectbox("Heart disease / heart attack?", ["No", "Yes"], index=0, help=feature_info["HeartDiseaseorAttack"]))
        PhysActivity = yesno_to_int(st.selectbox("Physical activity (last 30 days)?", ["No", "Yes"], index=0, help=feature_info["PhysActivity"]))
        Fruits = yesno_to_int(st.selectbox("Eats fruit daily?", ["No", "Yes"], index=0, help=feature_info["Fruits"]))
        Veggies = yesno_to_int(st.selectbox("Eats vegetables daily?", ["No", "Yes"], index=0, help=feature_info["Veggies"]))
        HvyAlcoholConsump = yesno_to_int(st.selectbox("Heavy alcohol consumption?", ["No", "Yes"], index=0, help=feature_info["HvyAlcoholConsump"]))
        DiffWalk = yesno_to_int(st.selectbox("Difficulty walking/climbing stairs?", ["No", "Yes"], index=0, help=feature_info["DiffWalk"]))

        sex_choice = st.selectbox("Sex", ["Female", "Male"], index=0, help=feature_info["Sex"])
        Sex = 1 if sex_choice == "Male" else 0
        Age = st.slider("Age category (1–13)", min_value=1, max_value=13, value=8, help=feature_info["Age"])
        GenHlth = st.slider("General health (1=Excellent → 5=Poor)", min_value=1, max_value=5, value=3, help=feature_info["GenHlth"])

    st.markdown("</div>", unsafe_allow_html=True)

    # small summary cards
    st.markdown("<div class='row-cards' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-card'><strong>Age</strong><div class='help-text'>{Age}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-card'><strong>Sex</strong><div class='help-text'>{sex_choice}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-card'><strong>BMI</strong><div class='help-text'>{BMI:.2f}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-card'><strong>GenHlth</strong><div class='help-text'>{GenHlth}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Build DataFrame for model
    features = pd.DataFrame([{
        "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck, "BMI": BMI,
        "Smoker": Smoker, "Stroke": Stroke, "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump, "DiffWalk": DiffWalk, "Sex": Sex,
        "Age": Age, "GenHlth": GenHlth
    }])

    # Scale numeric columns safely
    try:
        features[["BMI", "Age", "GenHlth"]] = scaler.transform(features[["BMI", "Age", "GenHlth"]])
    except Exception:
        # if scaler expects different ordering / types, ensure conversion to float
        features[["BMI", "Age", "GenHlth"]] = features[["BMI", "Age", "GenHlth"]].astype(float)
        features[["BMI", "Age", "GenHlth"]] = scaler.transform(features[["BMI", "Age", "GenHlth"]])

    # Robustly retrieve model column order and align
    try:
        model_cols = list(model.feature_names_in_)
    except Exception:
        try:
            model_cols = list(model.get_booster().feature_names)
        except Exception:
            # fallback: use the canonical order used during training (safe default)
            model_cols = [
                "HighBP", "HighChol", "CholCheck", "BMI",
                "Smoker", "Stroke", "HeartDiseaseorAttack",
                "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "GenHlth", "DiffWalk",
                "Sex", "Age"
            ]

    # Add any missing columns (fill with 0) and reorder exactly
    for c in model_cols:
        if c not in features.columns:
            features[c] = 0
    features = features[model_cols]

    # Predict button
    if st.button("Predict Diabetes Status"):
        # debug prints (comment out if not needed)
        # st.write("Model expects:", model_cols)
        # st.write("Features sent:", features.columns.tolist())

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        label_map = {0: "No Diabetes / Pregnancy Only", 1: "Prediabetes", 2: "Diabetes"}

        # Prediction box
        st.markdown(f"<div class='prediction-box'>{label_map[pred]}</div>", unsafe_allow_html=True)

        # Probability bar (plotly)
        fig = px.bar(
            x=[label_map[i] for i in range(len(proba))],
            y=proba,
            labels={"x": "Class", "y": "Probability"},
            title="Prediction Probabilities",
            range_y=[0, 1],
            color=[label_map[i] for i in range(len(proba))]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gauge for diabetes probability (index 2)
        risk = float(proba[2]) * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={'suffix': "%"},
            title={"text": "Estimated Diabetes Risk"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#ff4757"}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # small advice card
        advice = ""
        if pred == 0:
            advice = "No diabetes detected by model. Keep a healthy lifestyle."
        elif pred == 1:
            advice = "Prediabetes detected — consult a healthcare professional for lifestyle changes."
        else:
            advice = "Diabetes detected — seek medical advice and tests (A1C, fasting glucose)."

        st.markdown(f"<div style='margin-top:12px; padding:12px; border-radius:10px; background:#ffffff; box-shadow:0 6px 14px rgba(16,24,40,0.06);'><strong>Note:</strong> {advice}</div>", unsafe_allow_html=True)

# ---------------------------
# End of app
# ---------------------------
