import requests
import json
import os

# OpenSky API URL
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# Specific region as in this case Heathrow Airport
params = {
    "lamin": 51.0,
    "lomin": -0.6,
    "lamax": 51.6,
    "lomax": 0.2
}

# Data Save Path
DATA_PATH = "data/flights.json"

# Retrieve real-time flight data from OpenSky API
def get_flight_data():
    try:
        response = requests.get(OPENSKY_URL, params=params)
        response.raise_for_status()  
        data = response.json()
        return data.get("states", [])
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return None

# Save retrieved flight data
def save_flight_data(data):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if data:
        with open(DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Flight data saved: {DATA_PATH}")
    else:
        print("No data to save!")

if __name__ == "__main__":
    flights = get_flight_data()
    if flights:
        save_flight_data(flights)
        print(json.dumps(flights[:3], indent=2))  
