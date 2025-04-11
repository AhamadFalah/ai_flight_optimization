#dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import folium
import numpy as np
import math
import random
import requests
import json
import os
import branca
from jinja2 import Template
from datetime import datetime

# === Global configuration and helper functions ===

# OpenWeather API configuration
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1cee8056d94879a5018a7f0d339dad98"  # Your API key

# OpenSky API URL (not used in these map functions, provided for completeness)
OPENSKY_URL = "https://opensky-network.org/api/states/all"

def haversine_distance(coord1, coord2):
    """Return the distance in km between two lat/lon pairs."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Global simulation variables: set the expected exact coordinates.
EXPECTED_DESTINATION = {"latitude": 51.4700, "longitude": -0.4543}
EXPECTED_START = {"latitude": 51.1, "longitude": -0.5}

airport_coord = [EXPECTED_DESTINATION["latitude"], EXPECTED_DESTINATION["longitude"]]  # Heathrow (exact)
flight_coord = [51.60, -0.60]  # Starting flight position (simulated)
flight_data = {
    "callsign": "AB123",
    "origin_country": "Ireland",
    "speed": 250,    # km/h
    "altitude": 3200 # m
}

def safe_str(value):
    if value is None:
        return "N/A"
    try:
        return str(value)
    except Exception:
        return "Error"

def get_weather_data():
    params = {
        "lat": airport_coord[0],
        "lon": airport_coord[1],
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(OPENWEATHER_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Weather API Request Failed: {e}")
        return None

def generate_turbulence_data(center_lat, center_lon, radius=0.1):
    """Generate simulated turbulence data points around a center location."""
    points = []
    for _ in range(40):
        angle = 2 * math.pi * random.random()
        r = radius * math.sqrt(random.random())
        lat = center_lat + r * math.cos(angle)
        lon = center_lon + r * math.sin(angle)
        intensity = random.uniform(0.1, 1.0)
        points.append([lat, lon, intensity])
    return points

def load_route_data(file_path=None):
    """
    Load route data from a JSON file or return a default route.
    Override destination and start coordinates with expected values.
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        print("Loaded route data from file:")
        print("Original destination:", data.get('destination', {}))
        # Override destination coordinates
        data["destination"]["latitude"] = EXPECTED_DESTINATION["latitude"]
        data["destination"]["longitude"] = EXPECTED_DESTINATION["longitude"]
        # Override start coordinates
        data["start"]["latitude"] = EXPECTED_START["latitude"]
        data["start"]["longitude"] = EXPECTED_START["longitude"]
        print("Overridden destination:", data["destination"])
        print("Overridden start:", data["start"])
        return data
    else:
        default_data = {
            "start": {
                "latitude": EXPECTED_START["latitude"],
                "longitude": EXPECTED_START["longitude"],
                "baro_altitude": 5000,
                "velocity": 200,
                "true_track": 90
            },
            "destination": EXPECTED_DESTINATION,
            "points": [
                {"latitude": 51.1, "longitude": -0.5, "baro_altitude": 5000, "velocity": 200, "true_track": 90},
                {"latitude": 51.2, "longitude": -0.48, "baro_altitude": 6000},
                {"latitude": 51.3, "longitude": -0.46, "baro_altitude": 7000},
                {"latitude": 51.4, "longitude": -0.45, "baro_altitude": 5000},
                {"latitude": 51.4700, "longitude": -0.4543, "baro_altitude": 3000}
            ]
        }
        print("Using default route data with destination:")
        print("Destination:", default_data['destination'])
        return default_data

# Load route data (adjust the file path as needed)
route_data = load_route_data("results/route_1_trajectory.json")
# Uncomment the next line to always use the default route:
# route_data = load_route_data()

# === Folium Map Generator for Route Display ===
def generate_folium_map_route(route, step):
    try:
        # Ensure step does not exceed available points.
        step = min(step, len(route["points"]))
        route_points = route["points"][:step]
        route_coords = [(pt["latitude"], pt["longitude"]) for pt in route_points]

        # Debug: Print each route step's coordinates.
        print("\n--- Route Points ---")
        for i, pt in enumerate(route_points):
            print(f"Step {i+1} Expected Coordinates: Latitude = {pt['latitude']}, Longitude = {pt['longitude']}")
            print(f"Step {i+1} Used Coordinates: {route_coords[i]}")
        
        # Include destination for centering.
        destination = route["destination"]
        all_coords = route_coords + [(destination["latitude"], destination["longitude"])]
        center_lat = np.mean([pt[0] for pt in all_coords])
        center_lon = np.mean([pt[1] for pt in all_coords])
        
        # Create the Folium map.
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add the flight route polyline.
        if route_coords:
            folium.PolyLine(
                locations=route_coords,
                color="blue",
                weight=3,
                opacity=0.8,
                tooltip="Flight Route"
            ).add_to(m)
            
            # Add circle markers for each step.
            for i, coord in enumerate(route_coords):
                folium.CircleMarker(
                    location=coord,
                    radius=4,
                    color="yellow",
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=1.0,
                    tooltip=f"Step {i+1}"
                ).add_to(m)
        
        # Add the start marker.
        start = route["start"]
        print("\nStart Marker Expected Coordinates:")
        print(f"Latitude = {start['latitude']}, Longitude = {start['longitude']}")
        folium.Marker(
            location=[start["latitude"], start["longitude"]],
            popup="Start",
            icon=folium.Icon(color="green")
        ).add_to(m)
        
        # Always add the destination marker with exact coordinates.
        print("\nDestination Marker Expected Coordinates:")
        print(f"Latitude = {destination['latitude']}, Longitude = {destination['longitude']}")
        folium.Marker(
            location=[destination["latitude"], destination["longitude"]],
            popup="Destination (Heathrow)",
            icon=folium.Icon(color="red")
        ).add_to(m)
        
        return m.get_root().render()
    except Exception as e:
        print(f"Error in generate_folium_map_route: {e}")
        return f"<div>Error rendering Folium map: {safe_str(e)}</div>"

# === Dash App Setup ===

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Animated Flight Route Map (Folium)"),
    html.Div([
        html.Label("Select Map Type:"),
        dcc.RadioItems(
            id="map-type",
            options=[
                {"label": "Folium", "value": "folium"},
                {"label": "Plotly", "value": "plotly"}
            ],
            value="folium",
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ], style={"marginBottom": "20px"}),
    html.Div(id="map-container"),
    dcc.Interval(id="interval-component", interval=2000, n_intervals=0),
    html.Div(id="route-info", style={"marginTop": "20px", "fontSize": "16px"})
], style={"padding": "20px", "fontFamily": "Arial", "maxWidth": "1200px", "margin": "0 auto"})

# Callback to update the map content based on selected map type and animation step.
@app.callback(
    [Output("map-container", "children"),
     Output("route-info", "children")],
    [Input("interval-component", "n_intervals"),
     Input("map-type", "value")]
)
def update_map(n_intervals, map_type):
    # Increase step count with each interval.
    step = n_intervals + 1
    total_points = len(route_data["points"])
    if step > total_points:
        step = total_points

    progress_percent = (step / total_points) * 100
    info_text = f"Flight Progress: {progress_percent:.1f}% (Step {step} of {total_points})"

    if map_type == "folium":
        # Generate Folium map showing the route up to current step.
        map_html = generate_folium_map_route(route_data, step)
        map_component = html.Iframe(srcDoc=map_html,
                                    style={'width': '100%', 'height': '600px', 'border': 'none'})
    else:
        # Use Plotly map if selected.
        lats = [pt["latitude"] for pt in route_data["points"][:step]]
        lons = [pt["longitude"] for pt in route_data["points"][:step]]
        start = route_data["start"]
        destination = route_data["destination"]
        route_trace = go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="lines+markers",
            line=dict(width=3, color="blue"),
            marker=dict(size=8, color="yellow"),
            name="Flight Path"
        )
        start_trace = go.Scattermapbox(
            lat=[start["latitude"]],
            lon=[start["longitude"]],
            mode="markers",
            marker=dict(size=12, color="green"),
            name="Start"
        )
        dest_trace = go.Scattermapbox(
            lat=[destination["latitude"]],
            lon=[destination["longitude"]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Destination"
        )
        all_lats = lats + [destination["latitude"]]
        all_lons = lons + [destination["longitude"]]
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        fig = go.Figure(data=[route_trace, start_trace, dest_trace])
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center={"lat": center_lat, "lon": center_lon},
                zoom=9
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        map_component = dcc.Graph(figure=fig, style={"width": "100%", "height": "600px"})

    return map_component, info_text

if __name__ == '__main__':
    app.run(debug=True)
