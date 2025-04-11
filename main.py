# main.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from glob import glob

# Import custom modules
from preprocess_dataset import preprocess_dataset  # Updated import
from turbulence_model import train_turbulence_model
from flight_route_env import FlightRouteEnv, create_synthetic_weather_grid
from ppo_agent import train_flight_route_optimizer, evaluate_model_on_routes, generate_synthetic_test_routes

# === Configuration ===
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_or_create_weather_grid():
    # Try to load the most recent weather data
    weather_files = glob(os.path.join(DATA_DIR, "merged_*.json"))
    if weather_files:
        latest_file = sorted(weather_files)[-1]
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Extract unique weather points
        weather_points = {}
        for entry in data:
            key = f"{entry.get('lat', entry.get('latitude', 0))}_{entry.get('lon', entry.get('longitude', 0))}"
            if key not in weather_points:
                weather_points[key] = {
                    "lat": entry.get("lat", entry.get("latitude", 0)),
                    "lon": entry.get("lon", entry.get("longitude", 0)),
                    "temperature": entry.get("temperature", 0),
                    "pressure": entry.get("pressure", 0),
                    "humidity": entry.get("humidity", 0),
                    "wind_speed": entry.get("wind_speed", 0),
                    "wind_deg": entry.get("wind_deg", 0),
                    "weather": entry.get("weather", "unknown")
                }
        
        return list(weather_points.values())
    else:
        # Create synthetic weather grid
        print("No real weather data found. Creating synthetic grid...")
        return create_synthetic_weather_grid()

def make_env(start_point=None, destination=None):
    # Check for turbulence model
    model_path = os.path.join(MODEL_DIR, "turbulence_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "turbulence_scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(" Turbulence model not found. Please train the model first.")
        return None
    
    # Load weather grid
    weather_grid = load_or_create_weather_grid()
    
    # Define default start and destination points if not provided
    if start_point is None:
        start_point = {
            'latitude': 51.1, 
            'longitude': -0.5, 
            'baro_altitude': 5000,
            'velocity': 200,  # m/s
            'true_track': 90,  # degrees
            'vertical_rate': 0
        }
    
    if destination is None:
        destination = {
            'latitude': 51.5,
            'longitude': 0.1
        }
    
    # Create environment
    env = FlightRouteEnv(
        turbulence_model_path=model_path,
        scaler_path=scaler_path,
        weather_grid=weather_grid,
        start_point=start_point,
        destination=destination
    )
    
    return env

def main():
    print("=== Flight Route Optimization System ===")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 0: Preprocess data
    print("\n0. Preprocessing flight and weather data...")
    preprocess_dataset()  # Updated function call
    
    # Step 1: Train turbulence model if needed
    model_path = os.path.join(MODEL_DIR, "turbulence_model.joblib")
    if not os.path.exists(model_path):
        print("\n1. Training turbulence prediction model...")
        train_turbulence_model()
    else:
        print("\n1. Turbulence model already exists. Skipping training.")
    
    # Step 2: Create environment
    print("\n2. Creating flight route environment...")
    env = make_env()
    if env is None:
        print("Failed to create environment. Exiting.")
        return
    
    # Step 3: Train RL agent
    print("\n3. Training route optimization agent...")
    model = train_flight_route_optimizer(
        env_fn=make_env,
        total_timesteps=1000,  # For quick testing, increase for better results
        n_envs=2,
        save_freq=5000
    )
    
    # Step 4: Generate test routes and evaluate model
    print("\n4. Evaluating model on test routes...")
    test_routes = generate_synthetic_test_routes(num_routes=5)
    results = evaluate_model_on_routes(
        model=model,
        env_fn=make_env,
        routes=test_routes,
        save_dir=RESULTS_DIR
    )
    
    print("\n Pipeline complete!")
    print(f"Results saved to {RESULTS_DIR}")
    
    # Optional: Print summary of results
    print("\nRoute Evaluation Summary:")
    for result in results:
        print(f"Route {result['route_id']}:")
        print(f"  Total Reward: {result['total_reward']:.2f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Total Turbulence: {result['total_turbulence']:.2f}")
        print(f"  Total Fuel Used: {result['total_fuel']:.2f}")
        print(f"  Reached Destination: {result['reached_destination']}")
        print()

if __name__ == "__main__":
    main()