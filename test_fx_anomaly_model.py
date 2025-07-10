"""
Load a saved Isolation Forest pipeline and score NEW FX exposure data.

Usage:
    python test_fx_anomaly_model.py \
        --model_path fx_anomaly_pipeline.pkl \
        --input new_fx_data.csv \
        --output new_fx_data_scored.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd


def engineer_features(df):
    """Replicate training‑time feature engineering."""
    df = df.copy()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
    df["Due_Date"] = pd.to_datetime(df["Due_Date"])
    df["Days_To_Due"] = (df["Due_Date"] - df["Transaction_Date"]).dt.days
    df["Log_Amount_USD"] = np.log1p(df["Amount_USD"])
    return df


def main(args):
    # Load model
    pipe = joblib.load(args.model_path)
    print(f"[✓] Loaded model from {args.model_path}")

    # Load new data
    df_new = pd.read_csv(args.input)
    df_new = engineer_features(df_new)

    # Score
    df_new["anomaly_score"] = -pipe.decision_function(df_new)
    df_new["is_anomaly"] = pipe.predict(df_new) == -1

    df_new.to_csv(args.output, index=False)
    print(f"[✓] Scored data written to {args.output}")

    # Quick peek at the 10 strangest records
    print("\nTop 10 most anomalous rows:")
    print(
        df_new.sort_values("anomaly_score", ascending=False)
        .head(10)[
            [
                "Invoice_ID",
                "Entity",
                "Type",
                "Currency",
                "Amount_FCY",
                "Amount_USD",
                "anomaly_score",
            ]
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved pickle file")
    parser.add_argument("--input", required=True, help="CSV of new/unseen data")
    parser.add_argument("--output", default="scored_output.csv", help="Output CSV with anomaly flags")
    main(parser.parse_args())
