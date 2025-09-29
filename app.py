import streamlit as st
import joblib
import numpy as np

model = joblib.load("insurance_premium_model.pkl")

st.title("🏥 Medical Insurance Premium Prediction")

age = st.number_input("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_map = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}
region = region_map[region]

if st.button("Predict Premium"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Premium: ${prediction:,.2f}")

