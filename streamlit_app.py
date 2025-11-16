import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page Setup & Theme
# ---------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    page_icon="🩺"
)

st.markdown(
    """
    <style>
        .big-font { font-size:22px !important; }
        .center { text-align: center; }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: #f7f7f7;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Prediction (BRFSS2015)")
st.write("Provide health indicators below and the model will predict diabetes classification.")

# ---------------------------
# Feature Descriptions
# ---------------------------
feature_info = {
    "HighBP": "High blood pressure. 0 = No, 1 = Yes",
    "HighChol": "High cholesterol level. 0 = No, 1 = Yes",
    "CholCheck": "Had cholesterol check within the past 5 years. 0 = No, 1 = Yes",
    "BMI": "Body Mass Index (normal: 18.5–24.9). You may use the BMI calculator above.",
    "Smoker": "Smoked at least 100 cigarettes in lifetime. 0 = No, 1 = Yes",
    "Stroke": "Ever told by a doctor you had a stroke. 0 = No, 1 = Yes",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack. 0 = No, 1 = Yes",
    "PhysActivity": "Physical activity in the past 30 days (not including job). 0 = No, 1 = Yes",
    "Fruits": "Consumes fruit 1 or more times per day. 0 = No, 1 = Yes",
    "Veggies": "Consumes vegetables 1 or more times per day. 0 = No, 1 = Yes",
    "HvyAlcoholConsump": "Heavy drinking: >14 drinks/week (men), >7 drinks/week (women). 0 = No, 1 = Yes",
    "DiffWalk": "Difficulty walking or climbing stairs. 0 = No, 1 = Yes",
    "Sex": "0 = Female, 1 = Male",
    "Age": (
        "Age category based on BRFSS:\n"
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
    "GenHlth": "General health rating: 1 = Excellent → 5 = Poor"
}

# ---------------------------
# Load model & scaler
# ---------------------------
model = joblib.load('xgb_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------------------
# BMI Calculator (Optional)
# ---------------------------
st.subheader("📏 BMI Calculator (Optional)")

col_bmi1, col_bmi2, col_bmi3 = st.columns(3)

with col_bmi1:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)

with col_bmi2:
    height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)

with col_bmi3:
    if height > 0:
        bmi_calc = weight / ((height / 100) ** 2)
        st.metric("Calculated BMI", f"{bmi_calc:.2f}")

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("🧍 Patient Health Inputs")

left, right = st.columns(2)

with left:
    HighBP = st.selectbox('HighBP (0=no,1=yes)', [0,1], help=feature_info["HighBP"])
    HighChol = st.selectbox('HighChol (0=no,1=yes)', [0,1], help=feature_info["HighChol"])
    CholCheck = st.selectbox('CholCheck (0=no,1=yes)', [0,1], help=feature_info["CholCheck"])
    BMI = st.number_input('BMI', 10.0, 80.0, float(round(bmi_calc, 2)), help=feature_info["BMI"])
    Smoker = st.selectbox('Smoker (0=no,1=yes)', [0,1], help=feature_info["Smoker"])
    Stroke = st.selectbox('Stroke (0=no,1=yes)', [0,1], help=feature_info["Stroke"])

with right:
    HeartDiseaseorAttack = st.selectbox('HeartDiseaseorAttack', [0,1], help=feature_info["HeartDiseaseorAttack"])
    PhysActivity = st.selectbox('PhysActivity', [0,1], help=feature_info["PhysActivity"])
    Fruits = st.selectbox('Fruits', [0,1], help=feature_info["Fruits"])
    Veggies = st.selectbox('Veggies', [0,1], help=feature_info["Veggies"])
    HvyAlcoholConsump = st.selectbox('HvyAlcoholConsump', [0,1], help=feature_info["HvyAlcoholConsump"])
    DiffWalk = st.selectbox('Difficulty Walking', [0,1], help=feature_info["DiffWalk"])
    Sex = st.selectbox('Sex', [0,1], help=feature_info["Sex"])
    Age = st.slider('Age Category (1–13)', 1, 13, 8, help=feature_info["Age"])
    GenHlth = st.slider('General Health (1=excellent → 5=poor)', 1, 5, 3, help=feature_info["GenHlth"])

# Prepare DataFrame
features = pd.DataFrame({
    'HighBP':[HighBP],
    'HighChol':[HighChol],
    'CholCheck':[CholCheck],
    'BMI':[BMI],
    'Smoker':[Smoker],
    'Stroke':[Stroke],
    'HeartDiseaseorAttack':[HeartDiseaseorAttack],
    'PhysActivity':[PhysActivity],
    'Fruits':[Fruits],
    'Veggies':[Veggies],
    'HvyAlcoholConsump':[HvyAlcoholConsump],
    'DiffWalk':[DiffWalk],
    'Sex':[Sex],
    'Age':[Age],
    'GenHlth':[GenHlth]
})

# Scale numeric columns
num_cols = ['BMI','Age','GenHlth']
features[num_cols] = scaler.transform(features[num_cols])

# ---------------------------
# Prediction Button
# ---------------------------
st.markdown("### 🔮 Prediction & Analysis")

if st.button("Predict Diabetes Status"):
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    label_map = {
        0: 'No diabetes / Only during pregnancy',
        1: 'Prediabetes',
        2: 'Diabetes'
    }

    st.success(f"### 🧾 Prediction: **{label_map[pred]}**")

    # Probability Bar Chart
    proba_fig = px.bar(
        x=[label_map[i] for i in range(3)],
        y=proba,
        title="Probability Distribution",
        labels={'x':'Class', 'y':'Probability'},
        range_y=[0, 1],
    )
    st.plotly_chart(proba_fig, use_container_width=True)

    # Gauge Chart
    gauge_value = proba[2] * 100
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': "Diabetes Risk (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

