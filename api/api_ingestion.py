import requests
import json
import os
from datetime import datetime

# Flight Data:
OPENSKY_URL = "https://opensky-network.org/api/states/all"
params_flight = {
    "lamin": 51.0,
    "lomin": -0.6,
    "lamax": 51.6,
    "lomax": 0.2
}
FLIGHT_DATA_PATH = os.path.join("data", "flights.json")

def get_flight_data():
    try:
        response = requests.get(OPENSKY_URL, params=params_flight)
        response.raise_for_status()
        data = response.json()
        print("Raw flight data response:", json.dumps(data, indent=2))
        states = data.get("states") or []
        return {
            "timestamp": datetime.now().isoformat(),
            "states": states
        }
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return None

def save_flight_data(data):
    os.makedirs(os.path.dirname(FLIGHT_DATA_PATH), exist_ok=True)
    if data:
        with open(FLIGHT_DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Flight data saved to: {FLIGHT_DATA_PATH}")
    else:
        print("No flight data to save!")

# Weather Data (Grid based):
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1cee8056d94879a5018a7f0d339dad98"
LAT_MIN, LAT_MAX = 51.0, 51.6
LON_MIN, LON_MAX = -0.6, 0.2
NUM_ROWS = 10
NUM_COLS = 10
WEATHER_DATA_PATH = os.path.join("data", "weather.json")

def fetch_weather_data_grid():
    weather_results = []
    lat_step = (LAT_MAX - LAT_MIN) / NUM_ROWS
    lon_step = (LON_MAX - LON_MIN) / NUM_COLS

    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            lat = LAT_MIN + (i + 0.5) * lat_step
            lon = LON_MIN + (j + 0.5) * lon_step
            params = {
                "lat": lat,
                "lon": lon,
                "appid": API_KEY,
                "units": "metric"
            }
            try:
                response = requests.get(OPENWEATHER_URL, params=params)
                response.raise_for_status()
                data = response.json()
                try:
                    code = int(data.get("cod", 200))
                except ValueError:
                    code = 200
                if code != 200:
                    print(f"Error at ({lat}, {lon}): {data.get('message', 'Unknown error')}")
                    continue
                weather_results.append({
                    "lat": lat,
                    "lon": lon,
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                })
            except requests.exceptions.RequestException as e:
                print(f"Weather API Request Failed at ({lat}, {lon}): {e}")
                continue
    return weather_results

def save_weather_data_grid(data):
    os.makedirs(os.path.dirname(WEATHER_DATA_PATH), exist_ok=True)
    if data:
        with open(WEATHER_DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Weather grid data saved to: {WEATHER_DATA_PATH}")
    else:
        print("No weather data to save!")

def append_data_to_file(file_path, new_data):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
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

def process_rain_data(rain_field):
    if isinstance(rain_field, dict) and '1h' in rain_field:
        return rain_field['1h']
    elif isinstance(rain_field, (int, float)):
        return rain_field
    else:
        # Default value if data is missing or malformed
        return 0.0