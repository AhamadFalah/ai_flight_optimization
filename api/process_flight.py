import pandas as pd
import json

FLIGHT_DATA_PATH = "data/flights.json"
PROCESSED_DATA_PATH = "data/processed_flights.csv"

# Load the saved flight data from JSON 
def load_flight_data():
    with open(FLIGHT_DATA_PATH, "r") as f:
        return json.load(f)

# Convert raw JOSN data into a structured data set
def process_flight_data(data):
    columns = ["icao24", "callsign", "origin_country", "longitude", "latitude", 
               "baro_altitude", "on_ground", "velocity", "true_track", "vertical_rate"]
    
    flights = []
    for flight in data:
        flights.append([
            flight[0], flight[1], flight[2], flight[5], flight[6], 
            flight[7], flight[8], flight[9], flight[10], flight[11]
        ])
    
    df = pd.DataFrame(flights, columns=columns)
    return df

def save_processed_data(df):
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Processed flight data saved!")

# Run processing
if __name__ == "__main__":
    raw_data = load_flight_data()
    df = process_flight_data(raw_data)
    save_processed_data(df)
    print(df.head())
