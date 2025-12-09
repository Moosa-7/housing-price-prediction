import streamlit as st  # MUST BE FIRST IMPORT
import pandas as pd
import requests
import plotly.express as px
import boto3
import os
import numpy as np 
from pathlib import Path

# Debug message 
st.write("DEBUG: App is starting...")

# ============================
# Config & Local Paths
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
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
        st.info(f"📥 Downloading {key} from S3...")
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
        st.success(f"Downloaded {key}")
    except Exception as e:
        st.warning(f"Could not download {key} from S3. Using local mode. Error: {e}")
    
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
        fe = pd.read_csv(path_fe)
        meta = pd.read_csv(path_meta)
        
        # FIX: Ensure 'date' is datetime before using .dt accessor
        if "date" in meta.columns:
            meta["date"] = pd.to_datetime(meta["date"])
        
        meta = meta[["date", "city_full"]]

        if len(fe) != len(meta):
            st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
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

    except FileNotFoundError as e:
        st.error(f"❌ Missing data files. Please run the notebooks first. Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

fe_df, disp_df = load_data()

if fe_df.empty or disp_df.empty:
    st.error("CRITICAL ERROR: DataFrames are empty. Application stopped.")
    st.stop()

# ============================
# UI
# ============================
st.title("🏠 Housing Price Prediction — Holdout Explorer")

years = sorted(disp_df["year"].dropna().unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")

        # Payload for Table
        payload = fe_df.loc[idx].fillna(0).to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            # === FIX 1: Truncate Table Predictions ===
            if len(preds) != len(idx):
                preds = preds[:len(idx)] # Chop off extras

            view = disp_df.loc[idx, ["date", "region"]].copy()
            if "actual_price" in disp_df.columns:
                view["actual_price"] = disp_df.loc[idx, "actual_price"]
            elif actuals:
                view["actual_price"] = actuals

            view["prediction"] = preds
            view = view.sort_values("date")

            if "actual_price" in view.columns:
                mae = (view["prediction"] - view["actual_price"]).abs().mean()
                rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
                avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

                st.subheader("Predictions vs Actuals")
                st.dataframe(view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True), use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("MAE", f"${mae:,.0f}")
                with c2: st.metric("RMSE", f"${rmse:,.0f}")
                with c3: st.metric("Avg % Error", f"{avg_pct_error:.2f}%")
            else:
                st.subheader("Predictions")
                st.dataframe(view[["date", "region", "prediction"]], use_container_width=True)

            # ============================
            # Yearly Trend Chart (FIXED)
            # ============================
            try:
                if region == "All":
                    yearly_mask = (disp_df["year"] == year)
                else:
                    yearly_mask = (disp_df["year"] == year) & (disp_df["region"] == region)
                
                yearly_data = disp_df[yearly_mask].copy()
                
                # Payload for Chart
                payload_year = fe_df.loc[yearly_mask.index].fillna(0).to_dict(orient="records")
                resp_year = requests.post(API_URL, json=payload_year, timeout=60)
                yearly_preds = resp_year.json().get("predictions", [])
                
                # === FIX 2: Truncate Chart Predictions (The Magic Fix) ===
                if len(yearly_preds) != len(yearly_data):
                    st.warning(f"Note: API returned {len(yearly_preds)} rows, expected {len(yearly_data)}. Auto-trimming.")
                    yearly_preds = yearly_preds[:len(yearly_data)]
                # =========================================================

                yearly_data["prediction"] = pd.Series(yearly_preds, index=yearly_data.index).astype(float)

                monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

                fig = px.line(
                    monthly_avg, x="month", y=["actual_price", "prediction"], markers=True,
                    labels={"value": "Price ($)", "month": "Month"},
                    title=f"Yearly Trend — {year} ({region})"
                )
                fig.add_vrect(x0=month - 0.5, x1=month + 0.5, fillcolor="red", opacity=0.1, layer="below", line_width=0)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as chart_err:
                st.warning(f"Could not load trend chart: {chart_err}")

        except Exception as e:
            st.error(f"API call failed: {e}")

else:
    st.info("Choose filters and click **Show Predictions** to compute.")