import streamlit as st  # MUST BE FIRST IMPORT
import pandas as pd
import requests
import plotly.express as px
import boto3
import os
import numpy as np
from pathlib import Path

# ============================
# Config & Setup
# ============================
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# DEBUG: Show where we are trying to connect
default_api_url = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

with st.sidebar:
    st.header("⚙️ Configuration")
    # This input allows you to fix the connection without redeploying code!
    API_URL = st.text_input("API URL", value=default_api_url)
    
    st.info(f"Connecting to: `{API_URL}`")
    
    if "127.0.0.1" in API_URL or "localhost" in API_URL:
        st.warning("⚠️ 'localhost' will not work on AWS! Use the API Service's Public IP.")

S3_BUCKET = os.getenv("S3_BUCKET", "housing-regression-data")
REGION = os.getenv("AWS_REGION", "eu-west-2")

# Define where local data should live
LOCAL_DATA_DIR = Path("data/processed")
HOLDOUT_ENGINEERED_PATH = LOCAL_DATA_DIR / "feature_engineered_holdout.csv"
HOLDOUT_META_PATH = LOCAL_DATA_DIR / "cleaning_holdout.csv"

def load_from_s3_if_missing(key, local_path):
    """Download from S3 ONLY if file is missing locally and we have credentials."""
    local_path = Path(local_path)
    if local_path.exists():
        return str(local_path)

    try:
        s3 = boto3.client("s3", region_name=REGION)
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
    except Exception as e:
        # Fail silently here to avoid UI clutter; main loader handles errors
        print(f"S3 Download Error: {e}")

    return str(local_path)

# Ensure paths exist
path_fe = load_from_s3_if_missing("processed/feature_engineered_holdout.csv", HOLDOUT_ENGINEERED_PATH)
path_meta = load_from_s3_if_missing("processed/cleaning_holdout.csv", HOLDOUT_META_PATH)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    try:
        if not os.path.exists(path_fe) or not os.path.exists(path_meta):
             st.error("Data files not found. Check S3 credentials or local paths.")
             return pd.DataFrame(), pd.DataFrame()

        fe = pd.read_csv(path_fe)
        meta = pd.read_csv(path_meta)

        if "date" in meta.columns:
            meta["date"] = pd.to_datetime(meta["date"])
        
        meta = meta[["date", "city_full"]]

        # Alignment
        if len(fe) != len(meta):
            min_len = min(len(fe), len(meta))
            fe = fe.iloc[:min_len].copy()
            meta = meta.iloc[:min_len].copy()

        disp = pd.DataFrame(index=fe.index)
        disp["date"] = meta["date"]
        disp["region"] = meta["city_full"]
        disp["year"] = disp["date"].dt.year
        disp["month"] = disp["date"].dt.month

        if "price" in fe.columns:
            disp["actual_price"] = fe["price"]

        return fe, disp

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

fe_df, disp_df = load_data()

if fe_df.empty:
    st.stop()

# ============================
# UI Main Content
# ============================
st.title("🏠 Housing Price Prediction Explorer")

years = sorted(disp_df["year"].dropna().unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

c1, c2, c3 = st.columns(3)
with c1: year = st.selectbox("Year", years)
with c2: month = st.selectbox("Month", months)
with c3: region = st.selectbox("Region", regions)

if st.button("Predict Prices 🚀", type="primary"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)
    
    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found.")
    else:
        # Prepare Payload
        payload = fe_df.loc[idx].fillna(0).to_dict(orient="records")
        
        try:
            with st.spinner("Asking API for predictions..."):
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
            preds = data.get("predictions", [])
            
            # Align lengths
            preds = preds[:len(idx)]
            
            # Create View
            view = disp_df.loc[idx, ["date", "region"]].copy()
            if "actual_price" in disp_df.columns:
                view["actual_price"] = disp_df.loc[idx, "actual_price"]
            
            view["prediction"] = preds
            view = view.sort_values("date")

            # Metrics
            if "actual_price" in view.columns:
                mae = (view["prediction"] - view["actual_price"]).abs().mean()
                st.metric("MAE (Mean Error)", f"${mae:,.0f}")
                
            st.dataframe(view, use_container_width=True)
            
        except Exception as e:
            st.error(f"API Error: {e}")
            st.error("Check the Sidebar Configuration to ensure API URL is correct.")