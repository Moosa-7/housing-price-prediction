from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI(title="Housing Intelligence API")

# Load Models Dictionary
models = {}

@app.on_event("startup")
def load_models():
    """Load all models on startup to avoid loading delay per request"""
    model_dir = "models"  # Ensure this matches your container path
    try:
        models["regression"] = joblib.load(f"{model_dir}/regression.joblib")
        models["classification"] = joblib.load(f"{model_dir}/classification.joblib")
        
        # Spatial
        models["clustering"] = joblib.load(f"{model_dir}/clustering.joblib")
        models["scaler"] = joblib.load(f"{model_dir}/cluster_scaler.joblib")
        models["recommender"] = joblib.load(f"{model_dir}/recommender.joblib")
        models["rec_data"] = joblib.load(f"{model_dir}/rec_data.joblib")
        
        # Forecasting
        models["forecaster"] = joblib.load(f"{model_dir}/forecasting.joblib")
        print("✅ All Models Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Warning: Some models failed to load. Error: {e}")

@app.get("/")
def root():
    return {"status": "Active", "features": ["predict", "classify", "segment", "recommend", "forecast"]}

# 1. Regression (Existing)
@app.post("/predict")
def predict_price(data: list[dict]):
    df = pd.DataFrame(data)
    # Ensure you drop columns that weren't used in training if necessary
    # For simplicity, assuming input matches training features exactly
    pred = models["regression"].predict(df)
    return {"predictions": pred.tolist()}

# 2. Classification (New)
@app.post("/classify")
def classify_tier(data: list[dict]):
    df = pd.DataFrame(data)
    pred = models["classification"].predict(df)
    return {"tiers": pred.tolist()}

# 3. Clustering (New)
@app.post("/segment")
def segment_neighborhood(data: list[dict]):
    df = pd.DataFrame(data)
    # Transform data using the saved scaler
    X_scaled = models["scaler"].transform(df[['lat', 'lon', 'price']])
    clusters = models["clustering"].predict(X_scaled)
    return {"clusters": clusters.tolist()}

# 4. Recommendation (New)
@app.post("/recommend")
def get_recommendations(data: list[dict]):
    # Expects 1 row of input features
    df = pd.DataFrame(data)
    # Find nearest neighbors indices
    distances, indices = models["recommender"].kneighbors(df)
    
    # Retrieve the actual house data from the saved dataset
    # indices[0] contains the index numbers of the 5 closest houses
    similar_houses = models["rec_data"].iloc[indices[0]]
    
    return similar_houses.fillna(0).to_dict(orient="records")

# 5. Forecasting (New)
@app.get("/forecast/{months}")
def forecast_market(months: int):
    try:
        # Forecast future steps
        pred = models["forecaster"].forecast(steps=months)
        
        # Format for JSON (Date -> Price)
        results = [{"date": str(d.date()), "price": round(v, 2)} 
                   for d, v in zip(pred.index, pred.values)]
        return {"forecast": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))