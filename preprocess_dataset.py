# preprocess_dataset.py
import os
import json
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import traceback
import random

# === Configuration ===
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sample_data(sample_data_path):

    # Define the Heathrow area
    LAT_MIN, LAT_MAX = 51.0, 51.6
    LON_MIN, LON_MAX = -0.6, 0.2
    
    sample_data = []
    
    # Create 20 sample flight records
    for i in range(20):
        # Generate random position within Heathrow area
        lat = round(LAT_MIN + random.random() * (LAT_MAX - LAT_MIN), 4)
        lon = round(LON_MIN + random.random() * (LON_MAX - LON_MIN), 4)
        
        # Create a sample flight record
        flight = {
            "icao24": f"ab{i}cd{i+1}",
            "callsign": f"BA{1000+i}",
            "origin_country": "United Kingdom",
            "time_position": int(datetime.now().timestamp()) - i*60,
            "last_contact": int(datetime.now().timestamp()) - i*60,
            "longitude": lon,
            "latitude": lat,
            "geo_altitude": random.randint(5000, 10000),
            "velocity": random.randint(150, 250),
            "true_track": random.randint(0, 359),
            "vertical_rate": random.randint(-10, 10),
            "baro_altitude": random.randint(5000, 10000),
            "squawk": str(random.randint(1000, 9999)),
            "spi": False,
            "temperature": round(15 + random.uniform(-5, 5), 1),
            "pressure": round(1013 + random.uniform(-10, 10), 0),
            "humidity": round(60 + random.uniform(-20, 20), 0),
            "wind_speed": round(random.uniform(0, 15), 1),
            "wind_deg": random.randint(0, 359)
        }
        
        sample_data.append(flight)
    
    # Save sample data
    with open(sample_data_path, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f" Created sample data at: {sample_data_path}")

def compute_turbulence_score(wind_speed, vertical_rate, geo_altitude):

    try:
        # Handle None values with defaults
        wind_speed = 0.0 if wind_speed is None else float(wind_speed)
        vertical_rate = 0.0 if vertical_rate is None else float(vertical_rate)
        geo_altitude = 5000.0 if geo_altitude is None else float(geo_altitude)
        
        # Wind speed factor
        wind_factor = min(wind_speed / 20.0, 1.0)
        
        # Vertical rate factor (absolute value to capture both climbing and descending)
        vertical_factor = min(abs(vertical_rate) / 10.0, 1.0)
        
        # Altitude factor (more turbulence at certain altitudes)
        # Peak turbulence typically occurs around 6000-9000 meters
        altitude_factor = np.exp(-((geo_altitude - 7500)**2) / (2 * 2000**2))
        
        # Combine factors with weighted average
        turbulence_score = (
            0.4 * wind_factor + 
            0.3 * vertical_factor + 
            0.3 * altitude_factor
        )
        
        return round(max(0, min(turbulence_score, 1)), 2)
    except (TypeError, ValueError) as e:
        print(f"Error computing turbulence score: {e}")
        print(f"Values: wind_speed={wind_speed}, vertical_rate={vertical_rate}, geo_altitude={geo_altitude}")
        return 0.1  # Return a safe default value

def estimate_fuel_usage(velocity, baro_altitude, true_track, vertical_rate):

    try:
        # Handle None values with defaults
        velocity = 0.0 if velocity is None else float(velocity)
        baro_altitude = 10000.0 if baro_altitude is None else float(baro_altitude)
        true_track = 0.0 if true_track is None else float(true_track)
        vertical_rate = 0.0 if vertical_rate is None else float(vertical_rate)
        
        # Base fuel consumption
        base_consumption = 0.01 * velocity
        
        # Altitude impact (higher altitudes generally more fuel-efficient)
        altitude_factor = max(0.5, min(baro_altitude / 30000, 1.5))
        
        # Maneuvering penalty
        # Harder turns and significant vertical rates increase fuel consumption
        heading_penalty = abs(true_track % 360 - 180) / 180  # Maximum penalty when turning 90 degrees
        vertical_penalty = min(abs(vertical_rate) / 10, 1)
        
        # Combine factors
        fuel_estimate = base_consumption * altitude_factor * (1 + 0.2 * heading_penalty + 0.3 * vertical_penalty)
        
        return round(max(0.1, fuel_estimate), 2)
    except (TypeError, ValueError) as e:
        print(f"Error estimating fuel usage: {e}")
        print(f"Values: velocity={velocity}, baro_altitude={baro_altitude}, true_track={true_track}, vertical_rate={vertical_rate}")
        return 0.5  # Return a safe default value

def extract_route_information(df):

    if df.empty:
        print("Warning: DataFrame is empty, no routes to extract")
        return pd.DataFrame()
        
    # Group by unique flights
    grouped = df.groupby('icao24')
    
    routes = []
    for icao24, group in grouped:
        try:
            # Sort by timestamp to get route progression
            group_sorted = group.sort_values('time_position')
            
            if len(group_sorted) > 1:
                route = {
                    'icao24': icao24,
                    'callsign': group_sorted['callsign'].iloc[0],
                    'origin_country': group_sorted['origin_country'].iloc[0],
                    'start_latitude': group_sorted['latitude'].iloc[0],
                    'start_longitude': group_sorted['longitude'].iloc[0],
                    'end_latitude': group_sorted['latitude'].iloc[-1],
                    'end_longitude': group_sorted['longitude'].iloc[-1],
                    'total_distance': calculate_total_distance(group_sorted),
                    'average_velocity': group_sorted['velocity'].mean(),
                    'total_flight_time': (group_sorted['time_position'].iloc[-1] - group_sorted['time_position'].iloc[0])
                }
                routes.append(route)
        except Exception as e:
            print(f"Error extracting route for {icao24}: {e}")
            continue
    
    return pd.DataFrame(routes)

def calculate_total_distance(group):

    from math import radians, sin, cos, sqrt, atan2


# Title: Flight Path (Great Circle Distance Calculation and Haversine Formula)
# Description: This code calculates the great circle distance between two points on the Earth using the Haversine formula. 
# Author: Shrivenkatesh SS
# Date: Published approximately 3 months ago (as of April 2025)
# Code version: Version 1 of 1
# Availability: Kaggle (https://www.kaggle.com/code/venkateshss04/flight-path/notebook)

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in kilometers
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

    total_distance = 0
    for i in range(1, len(group)):
        try:
            total_distance += haversine_distance(
                group['latitude'].iloc[i-1], group['longitude'].iloc[i-1],
                group['latitude'].iloc[i], group['longitude'].iloc[i]
            )
        except Exception as e:
            print(f"Error calculating distance for points {i-1} and {i}: {e}")
            continue
    
    return round(total_distance, 2)

def preprocess_dataset():

    print("Starting preprocessing of flight data...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create a sample data file if no data exists
    sample_data_path = os.path.join(DATA_DIR, "merged_sample.json")
    if not os.path.exists(sample_data_path) and not glob(os.path.join(DATA_DIR, "merged_*.json")):
        print("No data files found. Creating a sample data file...")
        create_sample_data(sample_data_path)
        
    print("Looking for merged_*.json files in data directory...")
    
    # Find all merged JSON files
    files = glob(os.path.join(DATA_DIR, "merged_*.json"))
    
    if not files:
        print(" No merged_*.json files found in data directory.")
        return
    
    print(f"Found {len(files)} data files.")
    
    # Consolidated data storage
    all_flight_data = []
    
    # Process each merged data file
    for file in files:
        try:
            print(f"Processing file: {file}")
            with open(file, "r") as f:
                records = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f" Error reading file {file}: {e}")
            continue
        
        # Process each flight record
        for entry in records:
            try:
                # Skip ground aircraft or entries with missing critical data
                if (entry.get("on_ground", False) or 
                    not entry.get("latitude") or 
                    not entry.get("longitude")):
                    continue
                
                # Compute advanced features
                turbulence_score = compute_turbulence_score(
                    entry.get("wind_speed", 0.0),
                    entry.get("vertical_rate", 0.0),
                    entry.get("geo_altitude", 0.0)
                )
                
                fuel_estimate = estimate_fuel_usage(
                    entry.get("velocity", 0.0),
                    entry.get("baro_altitude", 0.0),
                    entry.get("true_track", 0.0),
                    entry.get("vertical_rate", 0.0)
                )
                
                # Processed flight record
                flight_record = {
                    "icao24": entry.get("icao24", ""),
                    "callsign": entry.get("callsign", ""),
                    "origin_country": entry.get("origin_country", "Unknown"),
                    "latitude": entry.get("latitude", 0.0),
                    "longitude": entry.get("longitude", 0.0),
                    "velocity": entry.get("velocity", 0.0),
                    "baro_altitude": entry.get("baro_altitude", 0.0),
                    "geo_altitude": entry.get("geo_altitude", 0.0),
                    "vertical_rate": entry.get("vertical_rate", 0.0),
                    "true_track": entry.get("true_track", 0.0),
                    "time_position": entry.get("time_position", 0),
                    "temperature": entry.get("temperature", 0.0),
                    "pressure": entry.get("pressure", 0.0),
                    "humidity": entry.get("humidity", 0.0),
                    "wind_speed": entry.get("wind_speed", 0.0),
                    "wind_deg": entry.get("wind_deg", 0),
                    "turbulence_score": turbulence_score,
                    "fuel_estimate": fuel_estimate
                }
                
                all_flight_data.append(flight_record)
            except Exception as e:
                print(f"Error processing flight record: {e}")
                print(f"Record: {entry}")
                continue
    
    if not all_flight_data:
        print(" No valid flight data processed.")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_flight_data)
    
    # Timestamp for current processing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure OUTPUT_DIR exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save processed flight data
    flight_data_path = os.path.join(OUTPUT_DIR, f"processed_flights_{timestamp}.csv")
    df.to_csv(flight_data_path, index=False)
    
    # Extract and save route information
    routes_df = extract_route_information(df)
    routes_path = os.path.join(OUTPUT_DIR, f"flight_routes_{timestamp}.csv")
    routes_df.to_csv(routes_path, index=False)
    
    # Print processing summary
    print(f" Processed flight data saved to {flight_data_path}")
    print(f" Flight routes saved to {routes_path}")
    print(f" Total flights processed: {len(df)}")
    print(f" otal unique routes: {len(routes_df)}")
    
    # Optional: Descriptive statistics
    print("\n=== Flight Data Summary ===")
    print(df.describe())
    
    return df, routes_df

if __name__ == "__main__":
    preprocess_dataset()