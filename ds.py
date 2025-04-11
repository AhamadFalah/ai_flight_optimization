import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
import numpy as np
import branca
import math
import random
import requests
import json
import os
from folium.plugins import HeatMap
from datetime import datetime

# OpenWeather API configuration
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1cee8056d94879a5018a7f0d339dad98"  # Your API key

# OpenSky API URL
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# Haversine formula: returns distance in km between two lat/lon pairs.
def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Global simulation variables
# Heathrow airport coordinates
airport_coord = [51.4700, -0.4543]
# Starting flight position (simulate a position some distance away)
flight_coord = [51.60, -0.60]
# Flight data from API (simulated)
flight_data = {
    "callsign": "AB123",
    "origin_country": "Ireland",
    "speed": 250,  # km/h, example
    "altitude": 3200  # example altitude in meters
}

# Safe string conversion function
def safe_str(value):
    """Safely convert any value to string without error."""
    if value is None:
        return "N/A"
    try:
        return str(value)
    except:
        return "Error"

# Function to get real-time weather data
def get_weather_data():
    """Fetches current weather data from the OpenWeather API."""
    params_weather = {
        "lat": airport_coord[0],
        "lon": airport_coord[1],
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(OPENWEATHER_URL, params=params_weather)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Weather API Request Failed: {e}")
        return None

# Function to get flight data
def get_flight_data():
    """Fetches real-time flight data from the OpenSky Network."""
    params_flight = {
        "lamin": 51.0,
        "lomin": -0.6,
        "lamax": 51.6,
        "lomax": 0.2
    }
    
    try:
        response = requests.get(OPENSKY_URL, params=params_flight)
        response.raise_for_status()
        data = response.json()
        states = data.get("states", [])
        return states if states else []
    except requests.exceptions.RequestException as e:
        print(f"Flight API Request Failed: {e}")
        return []

# Function to generate simulated turbulence data
def generate_turbulence_data(center_lat, center_lon, radius=0.1):
    """Generates simulated turbulence data points around a center location."""
    turbulence_points = []
    
    # Generate points in a rough circle with random intensity
    for _ in range(40):
        angle = 2 * math.pi * random.random()
        r = radius * math.sqrt(random.random())
        lat = center_lat + r * math.cos(angle)
        lon = center_lon + r * math.sin(angle)
        # Intensity from 0 to 1, higher values mean more turbulence
        intensity = random.uniform(0.1, 1.0)
        turbulence_points.append([lat, lon, intensity])
        
    return turbulence_points

# Function to generate folium map HTML string based on current flight and weather data
def generate_map(flight_coord):
    try:
        # Define bounding box around Heathrow area
        min_lat, max_lat = 51.4, 51.65
        min_lon, max_lon = -0.65, -0.35

        # Get real weather data - wrap in try/except for resilience
        try:
            weather_data = get_weather_data()
        except Exception as e:
            print(f"Error getting weather data for map: {e}")
            weather_data = None
        
        # Create folium map centered around Heathrow
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Create a new feature group for the grid
        grid_layer = folium.FeatureGroup(name='Cost Grid')
        
        # Increase grid resolution: 20 rows x 20 columns
        num_rows = 20
        num_cols = 20
        lat_steps = np.linspace(min_lat, max_lat, num_rows + 1)
        lon_steps = np.linspace(min_lon, max_lon, num_cols + 1)

        # Create a colormap from green (low cost) to red (high cost)
        colormap = branca.colormap.LinearColormap(colors=['green', 'yellow', 'red'], vmin=0, vmax=100)

        # Draw grid cells with simulated cost values
        for i in range(num_rows):
            for j in range(num_cols):
                cell_bounds = [
                    [lat_steps[i], lon_steps[j]],
                    [lat_steps[i+1], lon_steps[j]],
                    [lat_steps[i+1], lon_steps[j+1]],
                    [lat_steps[i], lon_steps[j+1]]
                ]
                # Simulated cost; in production, derive from weather, congestion etc.
                cost = np.random.uniform(0, 100)
                cell_color = colormap(cost)
                # Use static string for popup to avoid any potential split() issues
                popup_text = "Cost: {:.1f}".format(cost)
                folium.Polygon(
                    locations=cell_bounds,
                    color='black',
                    weight=1,
                    fill=True,
                    fill_color=cell_color,
                    fill_opacity=0.6,
                    popup=popup_text
                ).add_to(grid_layer)

        # Add grid layer to the map
        grid_layer.add_to(m)
        
        # Add colormap legend
        colormap.caption = 'Cost (0 = low risk, 100 = high risk)'
        colormap.add_to(m)

        # Add airport marker with a blue plane icon - USING MINIMAL PARAMETERS
        folium.Marker(
            location=airport_coord,
            popup="Airport: Heathrow",
            icon=folium.Icon(color="blue")  # Simplified icon parameters
        ).add_to(m)

        # Prepare flight tooltip with pre-formatted strings to avoid any potential split() issues
        callsign = safe_str(flight_data['callsign'])
        origin = safe_str(flight_data['origin_country'])
        speed = "{:.1f}".format(float(flight_data['speed']))
        altitude = "{:.0f}".format(float(flight_data['altitude']))
        
        # Pre-format the entire string
        popup_text = "Flight: " + callsign + "<br>Origin: " + origin + "<br>Speed: " + speed + " km/h<br>Altitude: " + altitude + " m"
        
        # Add flight marker with minimal parameters
        folium.Marker(
            location=flight_coord,
            popup=popup_text,
            icon=folium.Icon(color="red")  # Simplified icon parameters
        ).add_to(m)

        # Draw direct route (straight line)
        distance = haversine_distance(flight_coord, airport_coord)
        distance_str = "{:.2f}".format(distance)
        tooltip_text = "Direct Route: " + distance_str + " km"
        
        folium.PolyLine(
            locations=[flight_coord, airport_coord],
            color="gray",
            weight=2,
            opacity=0.5,
            tooltip=tooltip_text
        ).add_to(m)

        # Simulate an optimized route that avoids high-risk areas.
        optimized_route = [
            flight_coord,
            [(flight_coord[0]+airport_coord[0])/2 + 0.02, (flight_coord[1]+airport_coord[1])/2 - 0.02],
            [(flight_coord[0]+airport_coord[0])/2 - 0.01, (flight_coord[1]+airport_coord[1])/2 + 0.015],
            airport_coord
        ]
        
        # Calculate distances along the optimized route
        optimized_distance = sum(
            haversine_distance(optimized_route[i], optimized_route[i+1])
            for i in range(len(optimized_route)-1)
        )
        
        # Pre-format tooltip text
        optimized_distance_str = "{:.2f}".format(optimized_distance)
        optimized_tooltip = "Optimized Route: " + optimized_distance_str + " km, Score: 87%"

        folium.PolyLine(
            locations=optimized_route,
            color="blue",
            weight=3,
            opacity=0.8,
            tooltip=optimized_tooltip
        ).add_to(m)

        # Create weather layers with simpler naming to avoid potential issues
        weather_layer = folium.FeatureGroup(name='Weather')
        
        # If we have weather data, add it to the map with maximum safety
        if weather_data:
            try:
                # Extract weather data with maximum safety
                try:
                    # Default to empty dictionary for indices that might not exist
                    weather_item = weather_data.get('weather', [{}])[0] if weather_data.get('weather') else {}
                    weather_desc = safe_str(weather_item.get('description', 'Unknown'))
                except:
                    weather_desc = "Unknown"
                
                try:
                    # Handle potentially missing 'wind' field
                    wind_dict = weather_data.get('wind', {}) if isinstance(weather_data.get('wind'), dict) else {}
                    wind_speed = float(wind_dict.get('speed', 0))
                except:
                    wind_speed = 0.0
                    
                try:
                    # Handle potentially missing 'wind' field
                    wind_dict = weather_data.get('wind', {}) if isinstance(weather_data.get('wind'), dict) else {}
                    wind_dir = float(wind_dict.get('deg', 0))
                except:
                    wind_dir = 0.0
                
                try:
                    # Handle potentially missing 'main' field
                    main_dict = weather_data.get('main', {}) if isinstance(weather_data.get('main'), dict) else {}
                    temp = float(main_dict.get('temp', 0))
                    temp_str = "{:.1f}".format(temp) + "°C"
                except:
                    temp_str = "N/A"
                
                # Pre-format the entire HTML content to avoid dynamic string operations
                weather_html = """
                <div style="font-family: Arial; width: 200px;">
                    <h4>Current Weather at Heathrow</h4>
                    <p><b>Conditions:</b> """ + weather_desc + """</p>
                    <p><b>Wind:</b> """ + "{:.1f}".format(wind_speed) + """ m/s at """ + "{:.0f}".format(wind_dir) + """°</p>
                    <p><b>Temperature:</b> """ + temp_str + """</p>
                    <p><b>Updated:</b> """ + datetime.now().strftime('%H:%M:%S') + """</p>
                </div>
                """
                
                # Add weather marker with minimal parameters
                folium.Marker(
                    location=[airport_coord[0] + 0.02, airport_coord[1] + 0.02],
                    popup=folium.Popup(html=weather_html, max_width=300),
                    icon=folium.Icon(color="lightblue")  # Simplified icon parameters
                ).add_to(weather_layer)
                
            except Exception as e:
                print(f"Error adding weather data to map: {e}")
        
        # Add simplified turbulence representation (avoiding HeatMap which might be causing issues)
        turbulence_layer = folium.FeatureGroup(name='Turbulence')
        
        # Generate some turbulence data points
        turbulence_points = generate_turbulence_data(center_lat, center_lon, radius=0.08)
        
        # Use simple circle markers instead of HeatMap
        for point in turbulence_points:
            intensity = point[2]
            color = 'blue' if intensity < 0.5 else 'red'
            tooltip_text = "Turbulence: {:.2f}".format(intensity)
            
            folium.CircleMarker(
                location=[point[0], point[1]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=intensity * 0.6,
                tooltip=tooltip_text
            ).add_to(turbulence_layer)
        
        # Add simplified wind arrows layer
        wind_layer = folium.FeatureGroup(name='Wind')
        
        # Get wind data from weather or use defaults
        wind_speed = 5.0
        wind_dir = 45.0
        
        if weather_data:
            try:
                wind_dict = weather_data.get('wind', {}) if isinstance(weather_data.get('wind'), dict) else {}
                wind_speed = float(wind_dict.get('speed', 5.0))
            except:
                wind_speed = 5.0
                
            try:
                wind_dict = weather_data.get('wind', {}) if isinstance(weather_data.get('wind'), dict) else {}
                wind_dir = float(wind_dict.get('deg', 45.0))
            except:
                wind_dir = 45.0
        
        # Convert wind direction from meteorological to mathematical
        math_dir = (270 - wind_dir) % 360
        math_rad = math.radians(math_dir)
        
        
        arrow_spacing = 0.08  # Larger spacing for fewer arrows
        
        for lat in np.arange(min_lat, max_lat, arrow_spacing):
            for lon in np.arange(min_lon, max_lon, arrow_spacing):
                # Calculate arrow endpoint
                dx = math.cos(math_rad) * (0.005 * wind_speed)
                dy = math.sin(math_rad) * (0.005 * wind_speed)
                
                # Pre-format tooltip
                tooltip_text = "Wind: {:.1f} m/s at {:.0f}°".format(wind_speed, wind_dir)
                
                # Draw simple line for wind
                folium.PolyLine(
                    locations=[[lat, lon], [lat + dy, lon + dx]],
                    color='purple',
                    weight=1,
                    tooltip=tooltip_text
                ).add_to(wind_layer)
                
                # Add simple marker for arrowhead
                folium.CircleMarker(
                    location=[lat + dy, lon + dx],
                    radius=2,
                    color='purple',
                    fill=True,
                    fill_color='purple'
                ).add_to(wind_layer)
        
        # Add all layers to map
        grid_layer.add_to(m)
        weather_layer.add_to(m)
        turbulence_layer.add_to(m)
        wind_layer.add_to(m)
        
        # Add simplified layer control
        folium.LayerControl().add_to(m)

        # Get the HTML and return it
        try:
            return m.get_root().render()
        except Exception as e:
            print(f"Error rendering map: {e}")
            # Fall back to an extremely basic map if rendering fails
            basic_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            return basic_map.get_root().render()
    
    except Exception as e:
        print(f"Critical error in generate_map: {e}")
        # Return a simple HTML error message
        return """
        <div style="text-align: center; padding: 20px; font-family: Arial;">
            <h2>Map Rendering Error</h2>
            <p>Error details: """ + safe_str(e) + """</p>
            <p>Please check the console for more information.</p>
        </div>
        """

# Dash app for live updating
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Live Flight Path Optimization with Weather Overlays"),
    html.Div([
        html.Div([
            html.H3("Flight Information"),
            html.Div(id='flight-info', className='info-box')
        ], className='info-column'),
        html.Div([
            html.H3("Weather Information"),
            html.Div(id='weather-info', className='info-box')
        ], className='info-column')
    ], className='info-row'),
    # Use two separate intervals to prevent conflicts
    dcc.Interval(id='map-interval', interval=5000, n_intervals=0),
    dcc.Interval(id='info-interval', interval=5000, n_intervals=0),
    html.Div([
        html.Div(id='live-map', style={'width': '100%', 'height': '600px'})
    ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '5px'}),
    html.Div([
        html.H4("Layer Controls:"),
        html.P("Use the layer control in the top right of the map to toggle between:"),
        html.Ul([
            html.Li("Cost Grid - Shows risk/cost areas"),
            html.Li("Weather - General weather information"),
            html.Li("Turbulence - Points showing turbulence intensity"),
            html.Li("Wind - Arrows showing wind direction and speed")
        ])
    ], className='instructions')
], style={'padding': '20px', 'fontFamily': 'Arial'})

# Add some CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Flight Path Optimization Dashboard</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            .info-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .info-column {
                flex: 1;
                margin-right: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .info-box {
                min-height: 100px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .instructions {
                margin-top: 20px;
                padding: 15px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Global variable to simulate flight movement
current_flight_coord = flight_coord.copy()

# Define separate callbacks to prevent errors
@app.callback(
    Output('live-map', 'children'),
    Input('map-interval', 'n_intervals')
)
def update_map(n):
    try:
        global current_flight_coord
        
        # Simulate flight movement: add small random delta to lat and lon
        delta_lat = random.uniform(-0.001, 0.001)
        delta_lon = random.uniform(-0.001, 0.001)
        current_flight_coord[0] += delta_lat
        current_flight_coord[1] += delta_lon
        
        # Generate new map HTML with updated flight position and weather
        try:
            map_html = generate_map(current_flight_coord)
            # Embed the HTML in an IFrame
            return html.Iframe(
                srcDoc=map_html, 
                style={'width': '100%', 'height': '600px', 'border': 'none'}
            )
        except Exception as e:
            print(f"Error generating map: {e}")
            # Provide a fallback when map generation fails
            return html.Div([
                html.P("Map temporarily unavailable. Refreshing..."),
                html.Pre(str(e))
            ], style={'width': '100%', 'height': '600px', 'border': '1px solid red', 'padding': '20px'})
    except Exception as e:
        print(f"Critical error in map update: {e}")
        return html.Div([
            html.P("Dashboard error. Attempting to recover..."),
            html.Pre(str(e))
        ], style={'width': '100%', 'height': '600px', 'border': '1px solid red', 'padding': '20px'})

@app.callback(
    [Output('flight-info', 'children'),
     Output('weather-info', 'children')],
    Input('info-interval', 'n_intervals')
)
def update_dashboard(n):
    try:
        global flight_data
        
        # Update flight_data (in production, fetch from API)
        flight_data["speed"] = flight_data["speed"] + random.uniform(-5, 5)
        flight_data["altitude"] = flight_data["altitude"] + random.uniform(-50, 50)
        
        # Pre-format all values
        callsign = safe_str(flight_data['callsign'])
        origin = safe_str(flight_data['origin_country'])
        try:
            speed = "{:.1f}".format(float(flight_data['speed']))
        except:
            speed = "N/A"
            
        try:
            altitude = "{:.0f}".format(float(flight_data['altitude']))
        except:
            altitude = "N/A"
            
        try:
            distance = "{:.2f}".format(haversine_distance(current_flight_coord, airport_coord))
        except:
            distance = "N/A"
            
        update_time = datetime.now().strftime('%H:%M:%S')
        
        # Generate HTML for flight info panel
        flight_info_html = html.Div([
            html.P(f"Flight: {callsign}"),
            html.P(f"From: {origin}"),
            html.P(f"Speed: {speed} km/h"),
            html.P(f"Altitude: {altitude} m"),
            html.P(f"Distance to Heathrow: {distance} km"),
            html.P(f"Last updated: {update_time}")
        ])
        
        # Get weather data with maximum safety
        try:
            weather_data = get_weather_data()
            if weather_data:
                # Pre-format all values
                try:
                    weather_item = weather_data.get('weather', [{}])[0] if weather_data.get('weather') else {}
                    weather_desc = safe_str(weather_item.get('description', 'Unknown'))
                except:
                    weather_desc = "Unknown"
                
                try:
                    main_dict = weather_data.get('main', {}) if isinstance(weather_data.get('main'), dict) else {}
                    temp = "{:.1f}".format(float(main_dict.get('temp', 0))) + "°C"
                except:
                    temp = "N/A"
                    
                try:
                    main_dict = weather_data.get('main', {}) if isinstance(weather_data.get('main'), dict) else {}
                    humidity = "{:.0f}".format(float(main_dict.get('humidity', 0))) + "%"
                except:
                    humidity = "N/A"
                
                try:
                    wind_dict = weather_data.get('wind', {}) if isinstance(weather_data.get('wind'), dict) else {}
                    wind_speed = "{:.1f}".format(float(wind_dict.get('speed', 0)))
                    wind_dir = "{:.0f}".format(float(wind_dict.get('deg', 0)))
                    wind_str = f"{wind_speed} m/s at {wind_dir}°"
                except:
                    wind_str = "N/A"
                
                try:
                    visibility = "{:,}".format(int(weather_data.get('visibility', 0))) + " m"
                except:
                    visibility = "N/A"
                
                weather_info_html = html.Div([
                    html.P(f"Current conditions: {weather_desc}"),
                    html.P(f"Temperature: {temp}"),
                    html.P(f"Humidity: {humidity}"),
                    html.P(f"Wind: {wind_str}"),
                    html.P(f"Visibility: {visibility}"),
                    html.P(f"Last updated: {update_time}")
                ])
            else:
                weather_info_html = html.Div([
                    html.P("Weather data currently unavailable"),
                    html.P(f"Last attempt: {update_time}")
                ])
        except Exception as e:
            print(f"Error processing weather data: {e}")
            weather_info_html = html.Div([
                html.P("Error processing weather data"),
                html.P(f"Last attempt: {datetime.now().strftime('%H:%M:%S')}")
            ])
        
        # Return the outputs
        return flight_info_html, weather_info_html
        
    except Exception as e:
        print(f"Error in update_dashboard: {e}")
        # Return default content for both panels in case of error
        update_time = datetime.now().strftime('%H:%M:%S')
        default_panel = html.Div([
            html.P("Data temporarily unavailable"),
            html.P(f"Last attempt: {update_time}")
        ])
        
        return default_panel, default_panel

if __name__ == '__main__':
    app.run(debug=True)
