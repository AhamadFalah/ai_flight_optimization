import requests
import json
import os
import time
from datetime import datetime

# OpenSky API endpoint and bounding box parameters for flight data
OPENSKY_URL = "https://opensky-network.org/api/states/all"
params_flight = {
    "lamin": 51.0,
    "lomin": -0.6,
    "lamax": 51.6,
    "lomax": 0.2
}

# OpenWeather API endpoint and parameters for weather data
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1cee8056d94879a5018a7f0d339dad98"
params_weather = {
    "lat": 51.4700,
    "lon": -0.4543,
    "appid": API_KEY,
    "units": "metric"
}

# Directory and file paths for caching data
DATA_DIR = "data"
FLIGHT_DATA_FILE = os.path.join(DATA_DIR, "flights.json")
WEATHER_DATA_FILE = os.path.join(DATA_DIR, "weather.json")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Data Fetching Functions:

def fetch_flight_data():
    try:
        response = requests.get(OPENSKY_URL, params=params_flight)
        response.raise_for_status()
        data = response.json()
        states = data.get("states") or []
        return {
            "timestamp": datetime.now().isoformat(),
            "states": states
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching flight data: {e}")
        return None

def fetch_weather_data():
    try:
        response = requests.get(OPENWEATHER_URL, params=params_weather)
        response.raise_for_status()
        data = response.json()
        return {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Data Caching Function:

def append_data_to_file(file_path, new_data):
    if os.path.exists(file_path):
            try:
                data_list = json.load(f)
                if not isinstance(data_list, list):
                    data_list = [data_list]
            except json.JSONDecodeError:
                data_list = []
    else:
        data_list = []
    data_list.append(new_data)
    with open(file_path, "w") as f:
        json.dump(data_list, f, indent=2)

# Main: 

if __name__ == "__main__":
    print("Starting continuous data collection every 5 seconds...")
    try:
        while True:
            flight_data = fetch_flight_data()
            weather_data = fetch_weather_data()

            if flight_data:
                append_data_to_file(FLIGHT_DATA_FILE, flight_data)
                print(f"Flight data appended at {flight_data['timestamp']}")
            if weather_data:
                append_data_to_file(WEATHER_DATA_FILE, weather_data)
                print(f"Weather data appended at {weather_data['timestamp']}")

            time.sleep(15)  # Wait for 5 seconds before next collection
    except KeyboardInterrupt:
        print("Data collection stopped by user.")
