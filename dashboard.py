import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Output, Input
import json
import os

# Optional: paths to your cached data
WEATHER_DATA_PATH = "data/weather.json"
FLIGHT_DATA_PATH = "data/flights.json"

# Load functions
def load_weather():
    if os.path.exists(WEATHER_DATA_PATH):
        with open(WEATHER_DATA_PATH) as f:
            return json.load(f)
    return []

def load_flights():
    if os.path.exists(FLIGHT_DATA_PATH):
        with open(FLIGHT_DATA_PATH) as f:
            return json.load(f).get("states", [])
    return []

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Live AI Flight Dashboard"),
    dl.Map(center=[51.47, -0.45], zoom=9, children=[
        dl.TileLayer(),
        dl.LayerGroup(id="flights"),
        dl.LayerGroup(id="weather")
    ], style={'width': '100%', 'height': '600px'}),
    dcc.Interval(id='interval', interval=5000, n_intervals=0)
])

@app.callback(
    [Output("flights", "children"), Output("weather", "children")],
    Input("interval", "n_intervals")
)
def update_map(n):
    flight_data = load_flights()
    weather_data = load_weather()

    flight_markers = []
    for state in flight_data:
        try:
            lat = state[6]
            lon = state[5]
            if lat and lon:
                flight_markers.append(
                    dl.Marker(position=[lat, lon], children=dl.Tooltip(f"{state[1]}"))
                )
        except:
            continue

    weather_points = []
    for point in weather_data:
        lat = point.get("lat")
        lon = point.get("lon")
        wind = point["data"].get("wind", {}).get("speed", 0)
        color = "green" if wind < 5 else "orange" if wind < 10 else "red"
        weather_points.append(
            dl.Circle(center=[lat, lon], radius=500, color=color, fillOpacity=0.5)
        )

    return flight_markers, weather_points

if __name__ == '__main__':
    app.run(debug=True)

