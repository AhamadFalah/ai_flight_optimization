import requests
import json
import os
from datetime import datetime

# Flight Data:

# OpenSky API URL
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# Bounding box for a region 
params_flight = {
    "lamin": 51.0,
    "lomin": -0.6,
    "lamax": 51.6,
    "lomax": 0.2
}

# File path to save flight data
FLIGHT_DATA_PATH = "data/flights.json"

def get_flight_data():
    try:
        response = requests.get(OPENSKY_URL, params=params_flight)
        response.raise_for_status()
        data = response.json()
        # Debug: print raw data to inspect API response
        print("Raw flight data response:", json.dumps(data, indent=2))
        states = data.get("states") or []
        return states
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return []

def save_flight_data(data):
    os.makedirs(os.path.dirname(FLIGHT_DATA_PATH), exist_ok=True)
    if data:
        with open(FLIGHT_DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Flight data saved to: {FLIGHT_DATA_PATH}")
    else:
        print("No flight data to save!")


# Weather Data:

# OpenWeather API URL and API key
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1cee8056d94879a5018a7f0d339dad98"

# Heathrow Airport coordinates for weather data
params_weather = {
    "lat": 51.4700,
    "lon": -0.4543,
    "appid": API_KEY,
    "units": "metric"
}

# File path to save weather data
WEATHER_DATA_PATH = "data/weather.json"

def get_weather_data():
    try:
        response = requests.get(OPENWEATHER_URL, params=params_weather)
        response.raise_for_status()
        data = response.json()
        # Check for error code in response
        try:
            code = int(data.get("cod", 200))
        except ValueError:
            code = 200
        if code != 200:
            raise Exception(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Weather API Request Failed: {e}")
        return None

def save_weather_data(data):
    os.makedirs(os.path.dirname(WEATHER_DATA_PATH), exist_ok=True)
    if data:
        with open(WEATHER_DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Weather data saved to: {WEATHER_DATA_PATH}")
    else:
        print("No weather data to save!")


# Main Execution:

if __name__ == "__main__":
    print(f"Data ingestion started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch and save flight data
    flights = get_flight_data()
    if flights:
        save_flight_data(flights)
        print("First three flight states:")
        print(json.dumps(flights[:3], indent=2))
    else:
        print("No flight data retrieved.")

    # Fetch and save weather data
    weather = get_weather_data()
    if weather:
        save_weather_data(weather)
        print("Weather data:")
        print(json.dumps(weather, indent=2))
    else:
        print("No weather data retrieved.")
