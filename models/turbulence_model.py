import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

# Load processed flight data
df_flights = pd.read_csv("data/processed_flights.csv")

# Load processed weather data
df_weather = pd.read_csv("data/processed_weather.csv")

# Display missing values before merging
print("\n🛠 Checking for NaN values before merging:")
print("Flight Data Missing Values:\n", df_flights.isnull().sum())
print("Weather Data Missing Values:\n", df_weather.isnull().sum())

# Round flight location data to match weather precision 
df_flights["longitude"] = df_flights["longitude"].round(4)
df_flights["latitude"] = df_flights["latitude"].round(4)

# Merge weather data with flights based on location 
df = pd.merge(df_flights, df_weather, on=["longitude", "latitude"], how="left")

# Display missing values after merging
print("\n Checking for NaN values after merging:")
print(df.isnull().sum())

df = df.fillna({
    "temperature": df["temperature"].mean(),
    "humidity": df["humidity"].mean(),
    "pressure": df["pressure"].mean(),
    "wind_speed": df["wind_speed"].mean(),
    "wind_direction": df["wind_direction"].mean(),
    "cloud_coverage": df["cloud_coverage"].mean(),
    "visibility": df["visibility"].mean()
})

# Confirm all missing values are fixed
print("\n Final NaN check after filling:")
print(df.isnull().sum())

# Feature selection
X = df[["longitude", "latitude", "baro_altitude", "wind_speed", "temperature", "pressure", "humidity", "cloud_coverage"]]
y = df["baro_altitude"]  # Predicting altitude-based turbulence risk

# Ensure dataset is not empty before splitting
if X.isnull().values.any():
    print(" ERROR: X still contains NaN values!")
    exit()

if X.empty or y.empty:
    print(" ERROR: No valid data available for training. Check input files.")
    exit()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Neural Network Model 
model = MLPRegressor(hidden_layer_sizes=(50,50,50), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/turbulence_model.pkl")

print("\nTurbulence Forecasting Model Trained & Saved!")
