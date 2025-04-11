# turbulence_model.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import json

# === Configuration ===
DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_preprocessed_data():
    # Find the most recent processed flights CSV
    processed_files = sorted([
        f for f in os.listdir(DATA_DIR) 
        if f.startswith("processed_flights_") and f.endswith(".csv")
    ])
    
    if not processed_files:
        raise FileNotFoundError("No processed flight data found. Run preprocess.py first.")
    
    file_path = os.path.join(DATA_DIR, processed_files[-1])
    df = pd.read_csv(file_path)
    
    return df

def prepare_turbulence_features(df):

    # Select relevant features for turbulence prediction
    feature_columns = [
        'velocity', 
        'baro_altitude', 
        'vertical_rate', 
        'temperature', 
        'pressure', 
        'humidity', 
        'wind_speed', 
        'wind_deg'
    ]
    
    # Check for missing values
    print("Missing values:")
    print(df[feature_columns + ['turbulence_score']].isnull().sum())
    
    # Remove rows with missing values
    df_clean = df.dropna(subset=feature_columns + ['turbulence_score'])
    
    # Prepare features and target
    X = df_clean[feature_columns]
    y = df_clean['turbulence_score']
    
    return X, y, feature_columns

def train_turbulence_model():

    # Load and prepare data
    df = load_preprocessed_data()
    X, y, feature_columns = prepare_turbulence_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    
    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n=== Turbulence Prediction Model Performance ===")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Feature importance
    result = permutation_importance(
        model, X_test_scaled, y_test, 
        n_repeats=10, random_state=42
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    # Visualization: Feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Turbulence Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'turbulence_feature_importance.png'))
    
    # Visualization: Prediction vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Turbulence Score')
    plt.ylabel('Predicted Turbulence Score')
    plt.title('Turbulence Prediction: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'turbulence_prediction_scatter.png'))
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, 'turbulence_model.joblib')
    scaler_path = os.path.join(MODEL_DIR, 'turbulence_scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save model metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'features': feature_columns,
        'performance': {
            'mse': mse,
            'r2': r2,
            'mae': mae
        },
        'feature_importance': feature_importance.to_dict(orient='records')
    }
    
    with open(os.path.join(MODEL_DIR, 'turbulence_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n Model saved to {model_path}")
    print(f" Scaler saved to {scaler_path}")
    
    return model, scaler

def predict_turbulence(features):
    # Load model and scaler
    model_path = os.path.join(MODEL_DIR, 'turbulence_model.joblib')
    scaler_path = os.path.join(MODEL_DIR, 'turbulence_scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Turbulence model not trained. Run train_turbulence_model() first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Ensure that the input features are a DataFrame with valid feature names
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features, columns=scaler.feature_names_in_)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict and return the turbulence score (for a single prediction)
    return model.predict(features_scaled)[0]

if __name__ == "__main__":
    # Train the model when the script is run directly
    train_turbulence_model()
