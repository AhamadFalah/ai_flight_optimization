import pandas as pd
import json

FLIGHT_DATA_PATH = "data/processed_flights.csv"
WEATHER_DATA_PATH = "data/weather.json"

# Load flight and weather data
def load_data():
    df_flights = pd.read_csv(FLIGHT_DATA_PATH)

    with open(WEATHER_DATA_PATH, "r") as f:
        weather_data = json.load(f)
    
    return df_flights, weather_data

# Example function to process data
def optimize_routes():
    df_flights, weather_data = load_data()
    print("Optimizing routes based on flight and weather data...")

if __name__ == "__main__":
    optimize_routes()
