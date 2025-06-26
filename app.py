#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('heart_disease_model.pkl')
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title(" Heart Disease Risk Predictor")
st.markdown("Enter your health data below to assess heart disease risk.")

st.subheader("📊 Model Performance")
st.metric(label="Accuracy", value="92%", delta="+")

# Create a single column layout (full width)
col1 = st.columns(1)[0]

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], help="1 = Male, 0 = Female")
    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], help="1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic")
    bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=250)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], help="1 = True; 0 = False")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], help="0 = Normal, 1 = ST-T abnormality, 2 = Left ventricular hypertrophy")
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
    angina = st.selectbox("Exercise Induced Angina", options=[0, 1], help="1 = Yes; 0 = No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3], help="1 = Upsloping, 2 = Flat, 3 = Downsloping")
   

# Prediction button
if st.button("🔍 Predict Risk"):
    input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, restecg, max_hr, angina, oldpeak, slope]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High risk of heart disease detected!")
    else:
        st.success("✅ Low risk of heart disease.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and XGBoost- Dinuri Gamage ~heart disease risk predictor")
