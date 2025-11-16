import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# PAGE CONFIG + THEME COLORS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL THEME & CSS
st.markdown("""
    <style>
        body {
            background-color: #F2F4F8;
        }
        .main {
            background-color: #F2F4F8;
        }

        /* Stylish Header */
        .title {
            font-size: 36px;
            font-weight: 900;
            background: -webkit-linear-gradient(45deg, #1E88E5, #42A5F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding-bottom: 5px;
        }

        /* Card UI */
        .card {
            background: white;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0px 6px 14px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        /* Prediction Box */
        .prediction-box {
            background: linear-gradient(120deg, #1E88E5, #42A5F5);
            color: white;
            border-radius: 15px;
            padding: 25px;
            font-size: 26px;
            text-align: center;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD MODEL + SCALER
# -------------------------------------------------------------
model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------------------
# FEATURE DESCRIPTIONS
# -------------------------------------------------------------
feature_info = {
    "HighBP": "High blood pressure.",
    "HighChol": "High cholesterol.",
    "CholCheck": "Cholesterol check within last 5 years.",
    "BMI": "Body Mass Index.",
    "Smoker": "Smoked 100+ cigarettes in lifetime.",
    "Stroke": "Ever told you had a stroke.",
    "HeartDiseaseorAttack": "Heart attack or heart disease.",
    "PhysActivity": "Physical activity in last 30 days.",
    "Fruits": "Eats fruit daily.",
    "Veggies": "Eats vegetables daily.",
    "HvyAlcoholConsump": "Heavy drinking.",
    "DiffWalk": "Difficulty walking or climbing stairs.",
    "Sex": "Sex at birth.",
    "Age": (
        "Age Groups:\n"
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
    "GenHlth": "General health: 1=Excellent → 5=Poor."
}

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📏 BMI Calculator", "🧪 Diabetes Prediction"])

# =============================================================
# PAGE 1 — BMI CALCULATOR
# =============================================================
if page == "📏 BMI Calculator":
    st.markdown("<h1 class='title'>📏 BMI Calculator</h1>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)

        bmi = weight / ((height / 100) ** 2)
        st.metric("Your BMI", f"{bmi:.2f}")

        if bmi < 18.5:
            st.info("Underweight")
        elif bmi < 25:
            st.success("Normal weight")
        elif bmi < 30:
            st.warning("Overweight")
        else:
            st.error("Obese")

        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================
# PAGE 2 — DIABETES PREDICTION
# =============================================================
if page == "🧪 Diabetes Prediction":

    st.markdown("<h1 class='title'>🧪 Diabetes Prediction</h1>", unsafe_allow_html=True)

    # YES/NO helper
    def yn(x): return 1 if x == "Yes" else 0

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        HighBP = yn(st.selectbox("High Blood Pressure?", ["No", "Yes"], help=feature_info["HighBP"]))
        HighChol = yn(st.selectbox("High Cholesterol?", ["No", "Yes"], help=feature_info["HighChol"]))
        CholCheck = yn(st.selectbox("Recent Cholesterol Check?", ["No", "Yes"], help=feature_info["CholCheck"]))
        BMI = st.number_input("BMI", 10.0, 80.0, 25.0, help=feature_info["BMI"])
        Smoker = yn(st.selectbox("Ever Smoked?", ["No", "Yes"], help=feature_info["Smoker"]))
        Stroke = yn(st.selectbox("Ever Had A Stroke?", ["No", "Yes"], help=feature_info["Stroke"]))

    with right:
        HeartDiseaseorAttack = yn(st.selectbox("Heart Disease / Attack?", ["No", "Yes"]))
        PhysActivity = yn(st.selectbox("Physical Activity Recently?", ["No", "Yes"]))
        Fruits = yn(st.selectbox("Eats Fruit Daily?", ["No", "Yes"]))
        Veggies = yn(st.selectbox("Eats Vegetables Daily?", ["No", "Yes"]))
        HvyAlcoholConsump = yn(st.selectbox("Heavy Alcohol Use?", ["No", "Yes"]))
        DiffWalk = yn(st.selectbox("Difficulty Walking?", ["No", "Yes"]))
        Sex = 1 if st.selectbox("Sex", ["Female", "Male"]) == "Male" else 0
        Age = st.slider("Age Group (1–13)", 1, 13, 8, help=feature_info["Age"])
        GenHlth = st.slider("General Health (1–5)", 1, 5, 3)

    st.markdown("</div>", unsafe_allow_html=True)

    # Prepare DataFrame
    features = pd.DataFrame({
        "HighBP":[HighBP], "HighChol":[HighChol], "CholCheck":[CholCheck], "BMI":[BMI],
        "Smoker":[Smoker], "Stroke":[Stroke], "HeartDiseaseorAttack":[HeartDiseaseorAttack],
        "PhysActivity":[PhysActivity], "Fruits":[Fruits], "Veggies":[Veggies],
        "HvyAlcoholConsump":[HvyAlcoholConsump], "DiffWalk":[DiffWalk], "Sex":[Sex],
        "Age":[Age], "GenHlth":[GenHlth]
    })

    # Scale
    features[["BMI","Age","GenHlth"]] = scaler.transform(features[["BMI","Age","GenHlth"]])

    # ---------------------
    # Prediction
    # ---------------------
    if st.button("Predict Diabetes Status"):
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        label_map = {
            0: "No Diabetes / Pregnancy Only",
            1: "Prediabetes",
            2: "Diabetes"
        }

        st.markdown(f"<div class='prediction-box'>{label_map[pred]}</div>", unsafe_allow_html=True)

        # Probability bar plot
        fig = px.bar(
            x=list(label_map.values()),
            y=proba,
            labels={"x": "Class", "y": "Probability"},
            title="Prediction Probability",
            color=list(label_map.values()),
            range_y=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk gauge
        risk = proba[2] * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Diabetes Risk (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)
