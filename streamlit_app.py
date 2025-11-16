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

# Load model & scaler
model = joblib.load('xgb_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------------------
# BMI Calculator (New Feature)
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
    HighBP = st.selectbox('HighBP (0=no, 1=yes)', [0, 1])
    HighChol = st.selectbox('HighChol (0=no, 1=yes)', [0, 1])
    CholCheck = st.selectbox('CholCheck (0=no, 1=yes)', [0, 1])
    BMI = st.number_input('BMI (if unsure, use the calculator above)', 10.0, 80.0, float(round(bmi_calc, 2)))
    Smoker = st.selectbox('Smoker (0=no,1=yes)', [0, 1])
    Stroke = st.selectbox('Stroke (0=no,1=yes)', [0, 1])

with right:
    HeartDiseaseorAttack = st.selectbox('HeartDiseaseorAttack', [0, 1])
    PhysActivity = st.selectbox('PhysActivity', [0, 1])
    Fruits = st.selectbox('Fruits', [0, 1])
    Veggies = st.selectbox('Veggies', [0, 1])
    HvyAlcoholConsump = st.selectbox('HvyAlcoholConsump', [0, 1])
    DiffWalk = st.selectbox('Difficulty Walking', [0, 1])
    Sex = st.selectbox('Sex', [0, 1])
    Age = st.slider('Age Category (1–13)', 1, 13, 8)
    GenHlth = st.slider('General Health (1=excellent → 5=poor)', 1, 5, 3)

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
num_cols = ['BMI', 'Age', 'GenHlth']
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

    # ---------------------------
    # Probability Bar Chart
    # ---------------------------
    proba_fig = px.bar(
        x=[label_map[i] for i in range(3)],
        y=proba,
        title="Probability Distribution",
        labels={'x':'Class', 'y':'Probability'},
        range_y=[0, 1],
        color=[0,1,2]
    )
    st.plotly_chart(proba_fig, use_container_width=True)

    # ---------------------------
    # Gauge for Diabetes Risk
    # ---------------------------
    gauge_value = (proba[2] * 100)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': "Diabetes Risk (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(gauge, use_container_width=True)
