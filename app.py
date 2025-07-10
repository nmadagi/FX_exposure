# fx_anomaly_streamlit_app.py
# Streamlit UI to upload an ERP CSV, flag anomalies with a pre‑trained Isolation Forest pipeline,
# and let the user download the scored file.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate training‑time feature engineering."""
    df = df.copy()
    # Ensure expected columns exist
    date_cols = ["Transaction_Date", "Due_Date"]
    for col in date_cols:
        if col not in df.columns:
            st.error(f"Missing required date column: {col}")
            st.stop()

    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Due_Date"] = pd.to_datetime(df["Due_Date"], errors="coerce")

    if df[["Transaction_Date", "Due_Date"]].isna().any().any():
        st.warning("Some dates could not be parsed and will be set to NaT → rows may score higher as anomalies.")

    df["Days_To_Due"] = (df["Due_Date"] - df["Transaction_Date"]).dt.days
    if "Amount_USD" not in df.columns:
        st.error("Missing required monetary column: Amount_USD")
        st.stop()

    df["Log_Amount_USD"] = np.log1p(df["Amount_USD"].clip(lower=0))
    return df


def score_anomalies(model, df_fe: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with anomaly_score and is_anomaly flag."""
    df_scored = df_fe.copy()
    df_scored["anomaly_score"] = -model.decision_function(df_fe)
    df_scored["is_anomaly"] = model.predict(df_fe) == -1
    return df_scored


def to_csv_download(df: pd.DataFrame) -> BytesIO:
    """Convert DataFrame to CSV BytesIO for download button."""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

# ------------------------------------------------------
# Streamlit layout
# ------------------------------------------------------

st.set_page_config(page_title="FX Exposure Anomaly Detector", layout="wide")

st.title("📊 FX Exposure Anomaly Detector")
st.markdown(
    "Upload a monthly ERP FX exposure extract (CSV). The app will flag unusual invoices, intercompany entries, or payables/receivables using a pre‑trained Isolation Forest model."
)

# Sidebar for model upload / selection
with st.sidebar:
    st.header("Model")
    default_path = "fx_anomaly_pipeline.pkl"
    model_file = st.file_uploader(
        "Upload trained model (*.pkl) or leave empty to use default pipeline.",
        type=["pkl"],
        key="model_uploader",
    )

    if model_file is not None:
        # User‑supplied model
        model = joblib.load(model_file)
        st.success("Custom model loaded ✅")
    else:
        try:
            model = joblib.load(default_path)
            st.info("Using default model fx_anomaly_pipeline.pkl")
        except FileNotFoundError:
            st.error("Default model not found. Please upload a model .pkl file in the sidebar.")
            st.stop()

st.divider()

uploaded_file = st.file_uploader("Upload ERP CSV (e.g., june2025_erp_extract.csv)", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Reading CSV…"):
        df_raw = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    # Feature engineering
    with st.spinner("Engineering features & scoring anomalies…"):
        df_fe = engineer_features(df_raw)
        df_scored = score_anomalies(model, df_fe)

    # Show anomaly summary
    total_rows = len(df_scored)
    anomalies = df_scored["is_anomaly"].sum()
    st.metric("Total rows", total_rows)
    st.metric("Anomalies detected", anomalies)

    # Display top anomalies table
    st.subheader("🚩 Top 25 Most Anomalous Rows")
    top_n = df_scored.sort_values("anomaly_score", ascending=False).head(25)
    st.dataframe(top_n[
        [
            "Invoice_ID",
            "Entity",
            "Type",
            "Currency",
            "Amount_FCY",
            "Amount_USD",
            "Days_To_Due",
            "anomaly_score",
        ]
    ], use_container_width=True)

    # Download full scored CSV
    csv_bytes = to_csv_download(df_scored)
    st.download_button(
        label="📥 Download full scored CSV",
        data=csv_bytes,
        file_name="scored_" + uploaded_file.name,
        mime="text/csv",
    )

else:
    st.info("Awaiting ERP CSV upload…")
