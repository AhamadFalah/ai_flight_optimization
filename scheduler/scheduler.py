import os

print("Running data pipeline...")

# Fetch flight data
os.system("python api/opensky_api.py")

# Process flight data
os.system("python data/process_flight.py")

# Fetch weather data
os.system("python api/openweather_api.py")

# Process weather data
os.system("python data/process_weather.py")

# Run Traffic Model
os.system("python models/traffic_model.py")

# Run turbualance Model
os.system("python models/turbulence_model.py")

# Optimize flight routes
os.system("python models/route_optimizer.py")

print("Pipeline execution completed!")
