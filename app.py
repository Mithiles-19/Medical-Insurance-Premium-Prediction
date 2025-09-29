import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("insurance_model.pkl")

st.title("💊 Medical Insurance Premium Predictor")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Encode categorical inputs (must match training encoding)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_dict = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
region = region_dict[region]

# Prepare features
features = np.array([[age, sex, bmi, children, smoker, region]])

# Predict button
if st.button("Predict Premium"):
    prediction = model.predict(features)
    st.success(f"Estimated Premium: ${prediction[0]:,.2f}")
