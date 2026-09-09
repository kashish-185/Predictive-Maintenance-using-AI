import joblib
import pandas as pd
import numpy as np

def load_pipeline(model_path, features_path):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

def predict_engine_rul(engine_df, model, feature_cols):
    sensors = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ]
    df = engine_df.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)
    for s in sensors:
        df[f"{s}_rolling_mean_5"] = df.groupby("unit")[s].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f"{s}_rolling_std_5"] = df.groupby("unit")[s].transform(lambda x: x.rolling(5, min_periods=1).std())
        df[f"{s}_trend_5"] = df.groupby("unit")[s].diff(5)
    
    df = df.fillna(0)
    latest = df.groupby("unit").tail(1).reset_index(drop=True)
    X = latest[feature_cols]
    rul_pred = model.predict(X)
    return float(rul_pred[0])
