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

# Define path to save data
DATA_PATH = "data/flights.json"

# Retrieve real time flight data from OpenSky API
def get_flight_data():
    try:
        response = requests.get(OPENSKY_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('states', [])
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None

# Save retrieved flight data to a JSON file
def save_flight_data(data):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)  # Ensure folder exists
    if data:
        with open(DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print("Flight data saved successfully!")
    else:
        print("No data to save!")

# Run function
if __name__ == "__main__":
    flights = get_flight_data()
    if flights:
        save_flight_data(flights)
        print(json.dumps(flights[:5], indent=2))  # Print sample data
