import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title='Diabetes Prediction', layout='centered')
st.title('Diabetes Prediction (BRFSS2015)')
st.write('Input patient features and click Predict')

# Load model & scaler
model = joblib.load('xgb_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Input form
def user_input_features():
    HighBP = st.selectbox('HighBP (0=no,1=yes)', [0,1])
    HighChol = st.selectbox('HighChol (0=no,1=yes)', [0,1])
    CholCheck = st.selectbox('CholCheck (0=no,1=yes)', [0,1])
    BMI = st.number_input('BMI', min_value=10.0, max_value=80.0, value=27.0)
    Smoker = st.selectbox('Smoker (0=no,1=yes)', [0,1])
    Stroke = st.selectbox('Stroke (0=no,1=yes)', [0,1])
    HeartDiseaseorAttack = st.selectbox('HeartDiseaseorAttack (0=no,1=yes)', [0,1])
    PhysActivity = st.selectbox('PhysActivity (0=no,1=yes)', [0,1])
    Fruits = st.selectbox('Fruits (0=no,1=yes)', [0,1])
    Veggies = st.selectbox('Veggies (0=no,1=yes)', [0,1])
    HvyAlcoholConsump = st.selectbox('HvyAlcoholConsump (0=no,1=yes)', [0,1])
    DiffWalk = st.selectbox('DiffWalk (0=no,1=yes)', [0,1])
    Sex = st.selectbox('Sex (0=female,1=male)', [0,1])
    Age = st.slider('Age category (1..13)', 1, 13, 8)
    GenHlth = st.slider('GenHlth (1=excellent .. 5=poor)', 1, 5, 3)

    data = {
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
    }

    return pd.DataFrame(data)

input_df = user_input_features()

# Scale numerical columns
num_cols = ['BMI','Age','GenHlth']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Prediction
if st.button('Predict'):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label_map = {
        0: 'No diabetes / Only during pregnancy',
        1: 'Prediabetes',
        2: 'Diabetes'
    }

    st.subheader("Prediction Result")
    st.write(label_map[pred])

    st.subheader("Prediction Probabilities")
    st.write({label_map[i]: float(round(p, 4)) for i, p in enumerate(proba)})

