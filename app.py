
import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk")

st.sidebar.header("Patient Inputs")

pregnancies = st.sidebar.text_input("Pregnancies", "0")
glucose = st.sidebar.text_input("Glucose", "120")
bp = st.sidebar.text_input("Blood Pressure", "70")
skin_thickness = st.sidebar.text_input("Skin Thickness", "20")
insulin = st.sidebar.text_input("Insulin", "79")
bmi = st.sidebar.text_input("BMI", "25.0")
dpf = st.sidebar.text_input("Diabetes Pedigree Function", "0.5")
age = st.sidebar.text_input("Age", "33")

# Prediction
if st.button("Predict Diabetes Risk"):

    try:
        features = np.array([[
            int(pregnancies),
            int(glucose),
            int(bp),
            int(skin_thickness),
            int(insulin),
            float(bmi),
            float(dpf),
            int(age)
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        st.subheader("Result")

        if prediction == 1:
            st.error(f"Likely Diabetic\n\nRisk Score: {probability:.1f}%")
        else:
            st.success(f"Likely Not Diabetic\n\nRisk Score: {probability:.1f}%")

        st.progress(float(probability / 100))

    except ValueError:
        st.error("Please enter valid numerical values in all fields.")
