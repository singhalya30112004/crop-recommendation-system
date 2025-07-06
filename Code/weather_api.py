import requests

API_KEY = "YOUR_API_KEY_HERE"

def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&appid={API_KEY}&units=metric"
    )
    response = requests.get(url)
    data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 0)  # Default to 0 if no rain data

    return temperature, humidity, rainfall
