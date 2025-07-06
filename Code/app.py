import streamlit as st
import pandas as pd
import joblib
import os


# Load model and scaler
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '../Models/crop_model.pkl')
scaler_path = os.path.join(base_dir, '../Models/scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


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


# Prediction
if submitted:
    user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    st.success(f"**Recommended Crop:** {prediction.capitalize()}")


# Ask for lat/lon
lat = st.number_input("Latitude", value=26.9124, format="%.4f")
lon = st.number_input("Longitude", value=75.7873, format="%.4f")

if st.button("Auto-fill from Weather API"):
    from utils.weather_api import get_weather
    temperature, humidity, rainfall = get_weather(lat, lon)
    st.success(f"Weather fetched! Temp: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm")
