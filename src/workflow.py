from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define Paths
DATA_PATH = "data/processed/feature_engineered_holdout.csv"
MODELS_DIR = "models"

# Ensure models dir exists
os.makedirs(MODELS_DIR, exist_ok=True)

# --- TASK 1: INGESTION ---
@task(name="Load Data", retries=3, retry_delay_seconds=5)
def load_data(path: str):
    print(f"📥 Loading data from {path}...")
    df = pd.read_csv(path).fillna(0)
    # Create the date column just like in your notebook
    if 'year' in df.columns and 'month' in df.columns:
        df['date'] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    return df

# --- TASK 2: TRAIN REGRESSION ---
@task(name="Train Price Model")
def train_regression(df: pd.DataFrame):
    print("🤖 Training Regression Model...")
    
    # Select features (exclude non-numeric)
    feature_cols = [c for c in df.columns if c not in ["price", "date", "city_full", "id"]]
    X = df[feature_cols]
    y = df["price"]
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    
    path = f"{MODELS_DIR}/regression.joblib"
    joblib.dump(model, path)
    print(f"✅ Saved Regression Model to {path}")
    return path

# --- TASK 3: TRAIN CLUSTERING ---
@task(name="Train Clustering")
def train_clustering(df: pd.DataFrame):
    print("🗺️ Training Clustering Model...")
    
    if 'lat' in df.columns and 'lon' in df.columns:
        features = df[['lat', 'lon', 'price']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_scaled)
        
        # Save both
        joblib.dump(kmeans, f"{MODELS_DIR}/clustering.joblib")
        joblib.dump(scaler, f"{MODELS_DIR}/cluster_scaler.joblib")
        print("✅ Saved Clustering Model")
    else:
        print("⚠️ Skipping Clustering (missing lat/lon)")

# --- MAIN FLOW ---
@flow(name="Housing Training Pipeline", task_runner=SequentialTaskRunner())
def main_pipeline():
    # 1. Load
    data = load_data(DATA_PATH)
    
    # 2. Train Models (Sequential)
    train_regression(data)
    train_clustering(data)
    
    print("🚀 Pipeline Completed Successfully!")

if __name__ == "__main__":
    main_pipeline()