"""
Inference pipeline for Housing Regression MLE.

- Takes input data (Raw OR Processed).
- Applies preprocessing only if needed.
- Aligns features with training.
- Returns predictions.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"] 
else:
    TRAIN_FEATURE_COLUMNS = None


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    
    # === FIX: Robust Detection of Processed Data ===
    # If the data has 'zipcode_freq' OR 'city_full_encoded', it is definitely processed.
    # If it has 'lat' and 'lng', it is also processed.
    is_processed = False
    if "zipcode_freq" in input_df.columns or "city_full_encoded" in input_df.columns:
        is_processed = True
    elif "lat" in input_df.columns and "lng" in input_df.columns:
        is_processed = True

    if is_processed:
        print("ℹ️ Input appears already preprocessed. Skipping merge step.")
        df = input_df.copy()
    else:
        print("⚙️ Running clean_and_merge on raw input...")
        df = clean_and_merge(input_df)
        df = drop_duplicates(df)
        df = remove_outliers(df)
    
    # Step 2: Feature engineering (Only run if needed)
    if "date" in df.columns and "year" not in df.columns:
        df = add_date_features(df)

    # Step 3: Encodings ----------------
    # Frequency encoding (zipcode)
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    # Target encoding (city_full → city_full_encoded)
    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        try:
            df["city_full_encoded"] = target_encoder.transform(df["city_full"])
        except:
            df["city_full_encoded"] = 0
        df = df.drop(columns=["city_full"], errors="ignore")

    # Drop leakage columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Step 4: Separate actuals
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Step 5: Align columns
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Step 6: Load model & predict
    model = load(model_path)

    # Align column names if mismatch occurs
    if "city_encoded" in df.columns and "city_full_encoded" not in df.columns:
        df = df.rename(columns={"city_encoded": "city_full_encoded"})
    if "city_full_encoded" in df.columns and "city_encoded" not in df.columns:
         if hasattr(model, "feature_names") and "city_encoded" in model.feature_names:
             df = df.rename(columns={"city_full_encoded": "city_encoded"})

    preds = model.predict(df)

    # Step 7: Build output
    out = df.copy()
    out["predicted_price"] = preds
    
    # FIX: Safety check for length mismatch
    # If y_true got doubled or modified, ignore it to prevent crash
    if y_true is not None and len(y_true) == len(out):
        out["actual_price"] = y_true

    return out

# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    
    # Optional arguments to override defaults
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--freq_encoder", type=str, default=str(DEFAULT_FREQ_ENCODER), help="Path to frequency encoder pickle")
    parser.add_argument("--target_encoder", type=str, default=str(DEFAULT_TARGET_ENCODER), help="Path to target encoder pickle")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")