import os

print("Running data pipeline...")

# Fetch flight data
os.system("python api/opensky_api.py")

# Process flight data
os.system("python api/process_flight.py")

# Fetch weather data
os.system("python api/openweather_api.py")

# Optimize flight routes
os.system("python models/route_optimizer.py")

print("Pipeline execution completed!")
