"""
Train an Isolation Forest anomaly‑detection model on FX exposure data,
save the fitted pipeline to disk, and output a scored CSV.

Usage:
    python train_fx_anomaly_model.py \
        --input synthetic_fx_exposure_data.csv \
        --model_path fx_anomaly_pipeline.pkl \
        --output fx_exposure_with_anomaly_flags.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(num_cols, cat_cols):
    """Return a preprocessing + IsolationForest pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    iso_model = IsolationForest(
        n_estimators=300,
        contamination=0.02,   # tweak if you expect a different anomaly rate
        random_state=42,
    )

    return Pipeline(steps=[("prep", preprocess), ("clf", iso_model)])


def engineer_features(df):
    """Add engineered columns needed for the model."""
    df = df.copy()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
    df["Due_Date"] = pd.to_datetime(df["Due_Date"])
    df["Days_To_Due"] = (df["Due_Date"] - df["Transaction_Date"]).dt.days
    df["Log_Amount_USD"] = np.log1p(df["Amount_USD"])
    return df


def main(args):
    df = pd.read_csv(args.input)
    df = engineer_features(df)

    num_cols = ["Log_Amount_USD", "FX_Rate", "Days_To_Due"]
    cat_cols = ["Entity", "Type", "Business_Unit", "Currency", "GL_Code"]

    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(df)

    # Score & flag
    df["anomaly_score"] = -pipe.decision_function(df)    # higher => stranger
    df["is_anomaly"] = pipe.predict(df) == -1

    df.to_csv(args.output, index=False)
    print(f"[✓] Scored data written to {args.output}")

    joblib.dump(pipe, args.model_path)
    print(f"[✓] Trained model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to training CSV")
    parser.add_argument("--model_path", default="fx_anomaly_pipeline.pkl", help="Output pickle path")
    parser.add_argument("--output", default="fx_exposure_with_anomaly_flags.csv", help="Scored CSV path")
    main(parser.parse_args())
