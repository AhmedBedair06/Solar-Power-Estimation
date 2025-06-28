from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import requests

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

def get_open_meteo_forecast(lat, lon):
    # Variables
    daily_vars = [
        "shortwave_radiation_sum",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean"
    ]
    hourly_vars = ["surface_pressure"]

    # Build URLs
    daily_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily={','.join(daily_vars)}"
        "&timezone=auto"
    )

    hourly_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly={','.join(hourly_vars)}"
        "&timezone=auto"
    )

    # Fetch daily + hourly data
    daily_response = requests.get(daily_url)
    hourly_response = requests.get(hourly_url)

    if daily_response.status_code != 200 or hourly_response.status_code != 200:
        print("API error or bad response")
        return None

    daily_data = daily_response.json()['daily']
    hourly_data = hourly_response.json()['hourly']

    # Create daily DataFrame
    daily_df = pd.DataFrame({
        'date': pd.to_datetime(daily_data['time']),
        "daily_globalirrad_kwh": [x / 3.6 for x in daily_data["shortwave_radiation_sum"]],   # Already kWh/m²/day
        'daily_temperature_mean': daily_data['temperature_2m_mean'],       # °C
        'daily_humidity_mean': daily_data['relative_humidity_2m_mean'],    # %
        'daily_windspeed_mean': daily_data['wind_speed_10m_mean']          # m/s
    })

    # Create hourly DataFrame
    hourly_df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly_data['time']),
        'surface_pressure': hourly_data['surface_pressure']  # hPa
    })

    # Extract date and aggregate pressure
    hourly_df['date'] = hourly_df['timestamp'].dt.date
    pressure_df = hourly_df.groupby('date')['surface_pressure'].mean().reset_index()
    pressure_df.rename(columns={'surface_pressure': 'daily_pressure_mean'}, inplace=True)

    # Merge daily_df + pressure_df
    daily_df['date'] = daily_df['date'].dt.date
    merged_df = pd.merge(daily_df, pressure_df, on='date', how='inner')

    # Drop date (not needed for model)
    merged_df.drop(columns='date', inplace=True)

    expected_cols = scaler.feature_names_in_

    merged_df = merged_df[expected_cols]

    return merged_df

@app.get("/predict")
def predict(lat: float = Query(...), lon: float = Query(...), station_max_power: float = Query(...)):
    df = get_open_meteo_forecast(lat, lon)
    if df is None or df.empty:
        return {"error": "Could not retrieve or process forecast data"}

    # Scale and predict
    try:
        X_scaled = scaler.transform(df)
        percentage = model.predict(X_scaled)[0]
        percentage = max(0.0, min(1.0, percentage))  # Clamp to [0, 1]
        predicted_power = percentage * station_max_power
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    return {
        "percentage": round(percentage, 4),
        "predicted_power": round(predicted_power, 2)
    }
