import networkx as nx
import pandas as pd
import joblib
import os

# Load processed flight data
df_flights = pd.read_csv("data/processed_flights.csv")

# Load trained models
if not os.path.exists("models/traffic_model.pkl") or not os.path.exists("models/turbulence_model.pkl"):
    print("Trained models not found! Run traffic_model.py and turbulence_model.py first.")
    exit()

traffic_model = joblib.load("models/traffic_model.pkl")
turbulence_model = joblib.load("models/turbulence_model.pkl")

# Create graph for routes
G = nx.Graph()

# Heathrow Airport Code
HEATHROW_AIRPORT = "LHR"

# Feature columns (ensuring they match the model training data)
FEATURE_COLUMNS_TRAFFIC = ["longitude", "latitude", "velocity", "baro_altitude"]
FEATURE_COLUMNS_TURBULENCE = ["longitude", "latitude", "baro_altitude"]

# Add edges based on available flights
for _, flight in df_flights.iterrows():
    try:
        # Ignore grounded flights
        if flight["baro_altitude"] == 0:
            continue

        # Convert input to DataFrame with correct column names
        flight_features_traffic = pd.DataFrame([[flight["longitude"], flight["latitude"], flight["velocity"], flight["baro_altitude"]]], 
                                               columns=FEATURE_COLUMNS_TRAFFIC)

        flight_features_turbulence = pd.DataFrame([[flight["longitude"], flight["latitude"], flight["baro_altitude"]]], 
                                                  columns=FEATURE_COLUMNS_TURBULENCE)

        # Predict congestion and turbulence
        congestion = traffic_model.predict(flight_features_traffic)[0]
        turbulence = turbulence_model.predict(flight_features_turbulence)[0]

        # Define edge weight (higher congestion & turbulence = worse path)
        weight = congestion + (turbulence * 2)

        # Add flight as a node, with Heathrow as the destination
        G.add_edge(flight["callsign"], HEATHROW_AIRPORT, weight=weight)

    except Exception as e:
        print(f"Error processing flight {flight['callsign']}: {e}")
        continue

# Find the optimized route to Heathrow
def find_best_route_to_heathrow(flight_callsign):
    try:
        path = nx.shortest_path(G, source=flight_callsign, target=HEATHROW_AIRPORT, weight="weight")
        print(f"Optimized Route for {flight_callsign} to Heathrow: {path}")
        return path
    except nx.NetworkXNoPath:
        print(f"No available route for {flight_callsign} to Heathrow")
        return None

# Test route optimization
if __name__ == "__main__":
    print("\n Heathrow Route Optimization")
    flight_callsign = input("Enter Flight Callsign: ").strip().upper()
    find_best_route_to_heathrow(flight_callsign)
