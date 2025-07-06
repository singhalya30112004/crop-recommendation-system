#!/usr/bin/env python3

import requests

API_KEY = "YOUR_API_KEY_HERE"

def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&appid={API_KEY}&units=metric"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            raise Exception(data.get("message", "Weather data unavailable"))

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0.0)

        return temperature, humidity, rainfall

    except Exception as e:
        print(f"[Weather API Error] {e}")
        return 25.0, 60.0, 50.0