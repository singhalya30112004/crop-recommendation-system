import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("Crop Recommendation System")
st.markdown("Enter soil and climate conditions to get the best crop suggestion.")

# Input form
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", 0, 140)
    P = st.number_input("Phosphorus (P)", 5, 145)
    K = st.number_input("Potassium (K)", 5, 205)
    temperature = st.number_input("Temperature (°C)", 8.0, 43.0)
    humidity = st.number_input("Humidity (%)", 14.0, 100.0)
    ph = st.number_input("Soil pH", 3.5, 10.0)
    rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0)

    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    st.success(f"Recommended Crop: **{prediction.capitalize()}**")