import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load processed flight data
df = pd.read_csv("data/processed_flights.csv")

# Prepare dataset
X = df[["longitude", "latitude", "velocity", "baro_altitude"]]
y = df["velocity"]  

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/traffic_model.pkl")

print(" Traffic Congestion Model Trained & Saved!")
