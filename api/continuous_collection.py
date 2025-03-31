import time
from api_ingestion import (
    get_flight_data,
    append_data_to_file,  # Use the append function
    fetch_weather_data_grid,
    FLIGHT_DATA_PATH,
    WEATHER_DATA_PATH
)

if __name__ == "__main__":
    print("Starting continuous data collection every 5 seconds...")
    try:
        while True:
            # Get flight data (returns a dictionary with a timestamp and states)
            flight_data = get_flight_data()
            if flight_data:
                append_data_to_file(FLIGHT_DATA_PATH, flight_data)
                print(f"Flight data appended at {flight_data['timestamp']}")
            else:
                print("No flight data retrieved.")

            # Get grid-based weather data (returns a list of weather data entries)
            weather_grid = fetch_weather_data_grid()
            if weather_grid:
                append_data_to_file(WEATHER_DATA_PATH, weather_grid)
                print("Weather grid data appended.")
            else:
                print("No weather grid data retrieved.")

            time.sleep(5)  # Wait 5 seconds before the next collection cycle
    except KeyboardInterrupt:
        print("Data collection stopped by user.")