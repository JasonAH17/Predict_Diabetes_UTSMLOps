import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG + THEME
# =====================================================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for modern UI styling
st.markdown("""
    <style>
        body {
            background-color: #F8F9FA;
        }
        .main {
            background-color: #F8F9FA;
        }
        .title-text {
            font-size: 40px !important;
            font-weight: 900 !important;
            color: #1E88E5;
            text-align: center;
        }
        .subheader-text {
            font-size: 24px !important;
            font-weight: 700 !important;
            color: #0D47A1;
        }
        .card {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        }
        .prediction-box {
            background: linear-gradient(135deg, #1E88E5, #42A5F5);
            border-radius: 15px;
            padding: 20px;
            color: white;
            text-align: center;
            font-size: 26px;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# PAGE TITLE
# =====================================================

st.markdown("<h1 class='title-text'>🩺 Diabetes Prediction (BRFSS2015)</h1>", unsafe_allow_html=True)
st.write("Provide patient details below and the machine learning model will predict diabetes status.")

# =====================================================
# LOAD MODEL + SCALER
# =====================================================

model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================================
# FEATURE DESCRIPTIONS
# =====================================================
feature_info = {
    "HighBP": "High blood pressure. 0 = No, 1 = Yes",
    "HighChol": "High cholesterol. 0 = No, 1 = Yes",
    "CholCheck": "Cholesterol check within past 5 years?",
    "BMI": "Body Mass Index (normal: 18.5–24.9).",
    "Smoker": "Smoked 100+ cigarettes in lifetime. 0 = No, 1 = Yes",
    "Stroke": "Ever told by a doctor you had a stroke.",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack.",
    "PhysActivity": "Physical activity in last 30 days.",
    "Fruits": "Consumes fruit 1+ times per day.",
    "Veggies": "Consumes vegetables 1+ times per day.",
    "HvyAlcoholConsump": "Heavy drinking (>14 drinks/week men, >7 women).",
    "DiffWalk": "Difficulty walking or climbing stairs.",
    "Sex": "0 = Female, 1 = Male",
    "Age": (
        "BRFSS Age Groups:\n"
        "1 = 18–24\n"
        "2 = 25–29\n"
        "3 = 30–34\n"
        "4 = 35–39\n"
        "5 = 40–44\n"
        "6 = 45–49\n"
        "7 = 50–54\n"
        "8 = 55–59\n"
        "9 = 60–64\n"
        "10 = 65–69\n"
        "11 = 70–74\n"
        "12 = 75–79\n"
        "13 = 80+"
    ),
    "GenHlth": "General health: 1=Excellent → 5=Poor"
}

# =====================================================
# BMI CALCULATOR
# =====================================================

st.markdown("<h3 class='subheader-text'>📏 BMI Calculator (Optional)</h3>", unsafe_allow_html=True)

bmi_col1, bmi_col2, bmi_col3 = st.columns(3)

with bmi_col1:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)

with bmi_col2:
    height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)

with bmi_col3:
    bmi_calc = weight / ((height / 100) ** 2)
    st.metric("Calculated BMI", f"{bmi_calc:.2f}")

# =====================================================
# PATIENT INPUT FORM
# =====================================================

st.markdown("<h3 class='subheader-text'>🧍 Patient Health Inputs</h3>", unsafe_allow_html=True)

left, right = st.columns(2)

def yesno_to_int(c): return 1 if c == "Yes" else 0

with left:
    HighBP = yesno_to_int(st.selectbox("High Blood Pressure?", ["No", "Yes"], help=feature_info["HighBP"]))
    HighChol = yesno_to_int(st.selectbox("High Cholesterol?", ["No", "Yes"], help=feature_info["HighChol"]))
    CholCheck = yesno_to_int(st.selectbox("Cholesterol Check (5 yrs)?", ["No", "Yes"], help=feature_info["CholCheck"]))
    BMI = st.number_input("BMI", 10.0, 80.0, float(round(bmi_calc, 2)), help=feature_info["BMI"])
    Smoker = yesno_to_int(st.selectbox("Smoked 100+ Cigarettes?", ["No", "Yes"], help=feature_info["Smoker"]))
    Stroke = yesno_to_int(st.selectbox("Ever Had Stroke?", ["No", "Yes"], help=feature_info["Stroke"]))

with right:
    HeartDiseaseorAttack = yesno_to_int(st.selectbox("Heart Disease / Heart Attack?", ["No", "Yes"],
                                                     help=feature_info["HeartDiseaseorAttack"]))
    PhysActivity = yesno_to_int(st.selectbox("Physical Activity (last 30 days)?", ["No", "Yes"],
                                             help=feature_info["PhysActivity"]))
    Fruits = yesno_to_int(st.selectbox("Eats Fruit Daily?", ["No", "Yes"], help=feature_info["Fruits"]))
    Veggies = yesno_to_int(st.selectbox("Eats Veggies Daily?", ["No", "Yes"], help=feature_info["Veggies"]))
    HvyAlcoholConsump = yesno_to_int(st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"],
                                                  help=feature_info["HvyAlcoholConsump"]))
    DiffWalk = yesno_to_int(st.selectbox("Difficulty Walking?", ["No", "Yes"], help=feature_info["DiffWalk"]))
    Sex = 1 if st.selectbox("Sex", ["Female", "Male"], help=feature_info["Sex"]) == "Male" else 0
    Age = st.slider("Age Category (1–13)", 1, 13, 8, help=feature_info["Age"])
    GenHlth = st.slider("General Health (1–5)", 1, 5, 3, help=feature_info["GenHlth"])

# Prepare data
features = pd.DataFrame({
    "HighBP":[HighBP], "HighChol":[HighChol], "CholCheck":[CholCheck], "BMI":[BMI],
    "Smoker":[Smoker], "Stroke":[Stroke], "HeartDiseaseorAttack":[HeartDiseaseorAttack],
    "PhysActivity":[PhysActivity], "Fruits":[Fruits], "Veggies":[Veggies],
    "HvyAlcoholConsump":[HvyAlcoholConsump], "DiffWalk":[DiffWalk], "Sex":[Sex],
    "Age":[Age], "GenHlth":[GenHlth]
})

num_cols = ["BMI", "Age", "GenHlth"]
features[num_cols] = scaler.transform(features[num_cols])

# =====================================================
# PREDICTION
# =====================================================

st.markdown("<h3 class='subheader-text'>🔮 Prediction Results</h3>", unsafe_allow_html=True)

if st.button("Predict Diabetes Status"):
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    label_map = {
        0: "No diabetes / Only during pregnancy",
        1: "Prediabetes",
        2: "Diabetes"
    }

    st.markdown(f"<div class='prediction-box'>{label_map[pred]}</div>", unsafe_allow_html=True)

    # Probability Chart
    fig = px.bar(
        x=list(label_map.values()),
        y=proba,
        title="Class Probabilities",
        labels={"x": "Class", "y": "Probability"},
        color=list(label_map.values()),
        range_y=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Gauge Chart
    risk = proba[2] * 100
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Diabetes Risk (%)"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)
