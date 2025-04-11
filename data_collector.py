#data_collector.py
import time
import json
import os
from datetime import datetime
from opensky_api import OpenSkyApi
import requests
from math import radians, cos, sin, sqrt, atan2

# === Configuration ===
BBOX = (51.0, 51.6, -0.6, 0.2)  # Area around Heathrow
LAT_MIN, LAT_MAX = 51.0, 51.6
LON_MIN, LON_MAX = -0.6, 0.2
NUM_ROWS, NUM_COLS = 10, 10
WEATHER_API_KEY = "1cee8056d94879a5018a7f0d339dad98"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# === Helper: Haversine Distance ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# === Weather Grid Collector ===
def collect_weather_grid():
    lat_step = (LAT_MAX - LAT_MIN) / NUM_ROWS
    lon_step = (LON_MAX - LON_MIN) / NUM_COLS
    weather_grid = []

    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            lat = LAT_MIN + (i + 0.5) * lat_step
            lon = LON_MIN + (j + 0.5) * lon_step
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                weather_grid.append({
                    "lat": lat,
                    "lon": lon,
                    "temperature": data["main"]["temp"],
                    "pressure": data["main"]["pressure"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "wind_deg": data["wind"].get("deg", 0),
                    "weather": data["weather"][0]["description"]
                })
            except Exception as e:
                print(f"Weather API failed at ({lat}, {lon}): {e}")
    return weather_grid

# === Flight Data Collector ===
def collect_flight_data():
    api = OpenSkyApi()
    states = api.get_states(bbox=BBOX)

    if not states or not states.states:
        print(" No flight data received.")
        return []

    flight_list = []
    for s in states.states:
        if s.on_ground or s.latitude is None or s.longitude is None:
            continue  # Skip ground flights or missing coordinates
        flight = {
            "icao24": s.icao24,
            "callsign": s.callsign.strip() if s.callsign else "",
            "origin_country": s.origin_country,
            "time_position": s.time_position,
            "last_contact": s.last_contact,
            "longitude": s.longitude,
            "latitude": s.latitude,
            "geo_altitude": s.geo_altitude,
            "velocity": s.velocity,
            "true_track": s.true_track,
            "vertical_rate": s.vertical_rate,
            "baro_altitude": s.baro_altitude,
            "squawk": s.squawk,
            "spi": s.spi
        }
        flight_list.append(flight)
    return flight_list

# === Merge Flights with Nearest Weather Grid Cell ===
def match_flight_to_weather(flights, weather_grid):
    merged = []
    for flight in flights:
        min_dist = float("inf")
        nearest_weather = None
        for cell in weather_grid:
            dist = haversine(flight["latitude"], flight["longitude"], cell["lat"], cell["lon"])
            if dist < min_dist:
                min_dist = dist
                nearest_weather = cell
        merged.append({**flight, **nearest_weather})
    return merged

# === Save Combined Data ===
def save_merged_data(data):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(DATA_DIR, f"merged_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f" Merged data saved to: {filename}")

# === Main Collector Runner ===
def run_collector():
    print(f"\n[{datetime.utcnow()}]  Collecting data...")
    flights = collect_flight_data()
    weather_grid = collect_weather_grid()

    if flights and weather_grid:
        merged_data = match_flight_to_weather(flights, weather_grid)
        save_merged_data(merged_data)
        print(f"[{datetime.utcnow()}] Data collected successfully.")
    else:
        print(f"[{datetime.utcnow()}] Partial or missing data. Skipping save.")

# === Schedule Every 5 Minutes ===
if __name__ == "__main__":
    while True:
        run_collector()
        time.sleep(30)  # 5 minutes but for now its at 30 seconds for testing
        # time.sleep(300)  # Uncomment for real 5-minute interval
