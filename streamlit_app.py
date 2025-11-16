import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIG
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model & scaler
model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg, #f7faff 0%, #eef3ff 100%);
}

/* Title container */
.title-row {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:15px;
    margin-bottom:10px;
}
.title-text {
    font-size:36px;
    font-weight:900;
    color:#0b3d91;
}
.title-sub {
    font-size:16px;
    color:#2b4f88;
    margin-bottom:10px;
}

/* Base card */
.card {
    background: rgba(255,255,255,0.9);
    padding:22px;
    border-radius:16px;
    border:1px solid rgba(14, 30, 37, 0.06);
    box-shadow:0px 8px 20px rgba(0,0,0,0.08);
    transition:0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
}

/* Small cards row */
.row-cards {
    display:flex;
    gap:18px;
    flex-wrap:wrap;
}
.small-card {
    flex:1;
    min-width:240px;
    padding:16px;
    border-radius:14px;
    background:white;
    border:1px solid rgba(200,200,255,0.4);
    box-shadow:0px 6px 16px rgba(0,0,0,0.06);
    transition:0.2s ease;
}
.small-card:hover {
    transform: translateY(-3px);
}
.small-card strong {
    font-size:15px;
}

/* Prediction Box */
.prediction-box {
    border-radius:12px;
    padding:20px;
    color:white;
    text-align:center;
    font-weight:800;
    font-size:22px;
    background:linear-gradient(90deg,#1565c0,#42a5f5);
    box-shadow:0 8px 20px rgba(21,101,192,0.20);
}

.help-text {
    font-size:14px;
    color:#4b5d7a;
}

/* Color theming for summary cards */
.card-good {
    border-left:5px solid #4caf50;
}
.card-medium {
    border-left:5px solid #ffb300;
}
.card-bad {
    border-left:5px solid #e53935;
}

</style>
""", unsafe_allow_html=True)

# Feature descriptions

feature_info = {
    "HighBP": "High blood pressure. 0 = No, 1 = Yes",
    "HighChol": "High cholesterol. 0 = No, 1 = Yes",
    "CholCheck": "Cholesterol check within last 5 years.",
    "BMI": "Body Mass Index (kg/m²).",
    "Smoker": "Smoked ≥100 cigarettes in lifetime.",
    "Stroke": "Ever told you had a stroke.",
    "HeartDiseaseorAttack": "Heart disease / heart attack.",
    "PhysActivity": "Physical activity in past 30 days.",
    "Fruits": "Consumes fruit ≥1/day.",
    "Veggies": "Consumes vegetables ≥1/day.",
    "HvyAlcoholConsump": "Heavy drinking (men>14/wk, women>7/wk).",
    "DiffWalk": "Difficulty walking/climbing stairs.",
    "Sex": "Female=0, Male=1",
    "GenHlth": "1=Excellent → 5=Poor",
}

# Age mapping
age_map = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64",
    10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
}

genhlth_map = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}


# Sidebar navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("", ["BMI Calculator", "Input & Predict"])

# Title header
st.markdown("""
<div class="title-row">
    <div style="font-size:42px;">🩺</div>
    <div>
        <div class="title-text">Diabetes Prediction</div>
        <div class="title-sub">BRFSS 2015 — XGBoost Machine Learning Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# PAGE 1: BMI Calculator
if page == "BMI Calculator":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📏 BMI Calculator")

    weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
    height = st.number_input("Height (cm)", 120.0, 230.0, 170.0)

    bmi = weight / ((height / 100) ** 2)

    st.metric("Your BMI", f"{bmi:.2f}")

    if bmi < 18.5:
        st.warning("Underweight")
    elif bmi < 25:
        st.success("Normal")
    elif bmi < 30:
        st.info("Overweight")
    else:
        st.error("Obese")

    st.markdown("</div>", unsafe_allow_html=True)

# PAGE 2: Input & Predict
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Patient Inputs")

    def yn(x): return 1 if x == "Yes" else 0

    left, right = st.columns(2)

    with left:
        HighBP = yn(st.selectbox("High blood pressure?", ["No", "Yes"], help=feature_info["HighBP"]))
        HighChol = yn(st.selectbox("High cholesterol?", ["No", "Yes"], help=feature_info["HighChol"]))
        CholCheck = yn(st.selectbox("Recent cholesterol check?", ["No", "Yes"], help=feature_info["CholCheck"]))
        BMI = st.number_input("BMI (kg/m²)", 10.0, 80.0, 25.0, help=feature_info["BMI"])
        Smoker = yn(st.selectbox("Ever smoked 100+ cigarettes?", ["No", "Yes"], help=feature_info["Smoker"]))
        Stroke = yn(st.selectbox("Ever had a stroke?", ["No", "Yes"], help=feature_info["Stroke"]))

    with right:
        HeartDiseaseorAttack = yn(st.selectbox("Heart disease / attack?", ["No", "Yes"], help=feature_info["HeartDiseaseorAttack"]))
        PhysActivity = yn(st.selectbox("Physical activity recently?", ["No", "Yes"], help=feature_info["PhysActivity"]))
        Fruits = yn(st.selectbox("Eats fruit daily?", ["No", "Yes"], help=feature_info["Fruits"]))
        Veggies = yn(st.selectbox("Eats vegetables daily?", ["No", "Yes"], help=feature_info["Veggies"]))
        HvyAlcoholConsump = yn(st.selectbox("Heavy alcohol use?", ["No", "Yes"], help=feature_info["HvyAlcoholConsump"]))
        DiffWalk = yn(st.selectbox("Difficulty walking?", ["No", "Yes"], help=feature_info["DiffWalk"]))

        sex_choice = st.selectbox("Sex", ["Female", "Male"])
        Sex = 1 if sex_choice == "Male" else 0

        Age = st.slider("Age category", 1, 13, 8)
        Age_label = age_map[Age]

        GenHlth = st.slider("General Health (1=Excellent → 5=Poor)", 1, 5, 3)
        GenHlth_label = genhlth_map[GenHlth]

    st.markdown("</div>", unsafe_allow_html=True)

    # Summary cards (with icons + live labels)
    st.markdown("<div class='row-cards'>", unsafe_allow_html=True)

    # AGE card
    st.markdown(
        f"<div class='small-card card-medium'>"
        f"<strong>🧓 Age</strong>"
        f"<div class='help-text'>{Age} ({Age_label})</div>"
        f"</div>", unsafe_allow_html=True)

    # SEX card
    st.markdown(
        f"<div class='small-card card-good'>"
        f"<strong>🚻 Sex</strong>"
        f"<div class='help-text'>{sex_choice}</div>"
        f"</div>", unsafe_allow_html=True)

    # BMI card
    st.markdown(
        f"<div class='small-card card-medium'>"
        f"<strong>⚖️ BMI</strong>"
        f"<div class='help-text'>{BMI:.2f}</div>"
        f"</div>", unsafe_allow_html=True)

    # GENERAL HEALTH card
    card_class = "card-good" if GenHlth <= 2 else "card-medium" if GenHlth == 3 else "card-bad"

    st.markdown(
        f"<div class='small-card {card_class}'>"
        f"<strong>❤️ General Health</strong>"
        f"<div class='help-text'>{GenHlth} ({GenHlth_label})</div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Prepare data for model
    features = pd.DataFrame([{
        "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck, "BMI": BMI,
        "Smoker": Smoker, "Stroke": Stroke, "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump, "DiffWalk": DiffWalk, "Sex": Sex,
        "Age": Age, "GenHlth": GenHlth
    }])

    features[["BMI", "Age", "GenHlth"]] = scaler.transform(features[["BMI", "Age", "GenHlth"]])

    # XGBoost column alignment fix
    try:
        model_cols = list(model.feature_names_in_)
    except:
        try:
            model_cols = list(model.get_booster().feature_names)
        except:
            model_cols = [
                "HighBP","HighChol","CholCheck","BMI",
                "Smoker","Stroke","HeartDiseaseorAttack",
                "PhysActivity","Fruits","Veggies",
                "HvyAlcoholConsump","GenHlth","DiffWalk",
                "Sex","Age"
            ]

    for c in model_cols:
        if c not in features:
            features[c] = 0

    features = features[model_cols]

    # Prediction
    if st.button("Predict Diabetes Status"):
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        label_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}

        st.markdown(f"<div class='prediction-box'>{label_map[pred]}</div>", unsafe_allow_html=True)

        fig = px.bar(
            x=[label_map[i] for i in range(len(proba))],
            y=proba,
            labels={"x": "Class", "y": "Probability"},
            range_y=[0, 1],
            color=[label_map[i] for i in range(len(proba))]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gauge chart
        risk = proba[2] * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={'suffix': "%"},
            title={"text": "Diabetes Risk (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#e74c3c"}}
        ))
        st.plotly_chart(gauge, use_container_width=True)
