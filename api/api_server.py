from flask import Flask, jsonify, abort
import json
import os

app = Flask(__name__)

# Paths to the cached data files
FLIGHT_DATA_PATH = "data/flights.json"
WEATHER_DATA_PATH = "data/weather.json"

def load_json_data(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

@app.route('/flights', methods=['GET'])
def get_flights():
    data = load_json_data(FLIGHT_DATA_PATH)
    if data is None:
        abort(404, description="Flight data not found")
    return jsonify(data)

@app.route('/weather', methods=['GET'])
def get_weather():
    data = load_json_data(WEATHER_DATA_PATH)
    if data is None:
        abort(404, description="Weather data not found")
    return jsonify(data)

if __name__ == '__main__':
    # Run the API server in debug mode for development purposes
    app.run(debug=True)
