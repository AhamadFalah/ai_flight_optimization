import requests
import json
import os

# OpenWeather API URL & Key 
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1cee8056d94879a5018a7f0d339dad98"

# Specific region as in this case Heathrow Airport
params = {
    "lat": 51.4700,
    "lon": -0.4543,
    "appid": API_KEY,
    "units": "metric"
}

DATA_PATH = "data/weather.json"

# Retrieeve real time weather data
def get_weather_data():
    
    try:
        response = requests.get(OPENWEATHER_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None

# Save retrieved weather data to a JSON file
def save_weather_data(data):

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if data:
        with open(DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print("Weather data saved!")
    else:
        print("No weather data to save!")

if __name__ == "__main__":
    weather = get_weather_data()
    if weather:
        save_weather_data(weather)
        print(json.dumps(weather, indent=2))
