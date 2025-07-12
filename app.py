# app.py – Enhanced FX Exposure Anomaly Detector with Business‑Friendly Risk Levels
# ---------------------------------------------------------------
# Streamlit UI: Upload ERP CSV → Score with Isolation Forest → Display risk buckets and allow download.
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate training‑time feature engineering (date deltas, log amounts)."""
    df = df.copy()
    required_cols = ["Transaction_Date", "Due_Date", "Amount_USD"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Due_Date"] = pd.to_datetime(df["Due_Date"], errors="coerce")
    df["Days_To_Due"] = (df["Due_Date"] - df["Transaction_Date"]).dt.days
    df["Log_Amount_USD"] = np.log1p(df["Amount_USD"].clip(lower=0))
    return df


def score_anomalies(model, df_fe: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with raw score, percentile, risk bucket, and business flag."""
    scored = df_fe.copy()
    scored["anomaly_score"] = -model.decision_function(df_fe)
    scored["anomaly_percentile"] = scored["anomaly_score"].rank(pct=True) * 100
    scored["anomaly_percentile"] = scored["anomaly_percentile"].round(2)

    def map_risk(p):
        if p >= 98:
            return "🔴 High Risk"
        elif p >= 90:
            return "🟠 Moderate Risk"
        elif p >= 75:
            return "🟡 Low Risk"
        else:
            return "🟢 Normal"

    scored["risk_level"] = scored["anomaly_percentile"].apply(map_risk)

    def business_msg(risk):
        return {
            "🔴 High Risk": "Potential FX exposure issue – investigate immediately",
            "🟠 Moderate Risk": "Review for misclassification or unusual terms",
            "🟡 Low Risk": "Minor outlier – monitor",
            "🟢 Normal": "No action needed",
        }[risk]

    scored["business_flag"] = scored["risk_level"].apply(business_msg)
    scored["is_anomaly"] = scored["risk_level"].isin(["🔴 High Risk", "🟠 Moderate Risk"])
    return scored


def to_csv_download(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------

st.set_page_config(page_title="FX Anomaly Detector", layout="wide")
st.title("📊 FX Exposure Anomaly Detector")
st.write("Upload an ERP FX exposure CSV. The app flags anomalies and maps them to business‑friendly risk levels.")

# Sidebar – load model
with st.sidebar:
    st.header("Model")
    mdl_file = st.file_uploader("Upload .pkl model (optional)", type=["pkl"])
    default_model = "fx_anomaly_pipeline.pkl"
    if mdl_file:
        model = joblib.load(mdl_file)
        st.success("Custom model loaded ✅")
    else:
        try:
            model = joblib.load(default_model)
            st.info("Using default model fx_anomaly_pipeline.pkl")
        except FileNotFoundError:
            st.error("Default model not found. Upload one on the sidebar.")
            st.stop()

# Main – upload ERP CSV
csv_file = st.file_uploader("Upload ERP CSV", type=["csv"])
if not csv_file:
    st.info("Awaiting ERP CSV upload…")
    st.stop()

# Read and preview
raw_df = pd.read_csv(csv_file)
st.subheader("Raw Data Preview")
st.dataframe(raw_df.head(), use_container_width=True)

# Engineer, score, and enrich
with st.spinner("Scoring anomalies…"):
    fe_df = engineer_features(raw_df)
    scored_df = score_anomalies(model, fe_df)

# KPIs
total = len(scored_df)
high = (scored_df["risk_level"] == "🔴 High Risk").sum()
moderate = (scored_df["risk_level"] == "🟠 Moderate Risk").sum()
st.metric("Total Records", total)
st.metric("High Risk", high)
st.metric("Moderate Risk", moderate)

# Show top anomalies
st.subheader("Top 25 Anomalies (by percentile)")
cols_show = [
    "Invoice_ID",
    "Entity",
    "Type",
    "Currency",
    "Amount_USD",
    "Days_To_Due",
    "anomaly_percentile",
    "risk_level",
    "business_flag",
]

st.dataframe(
    scored_df.sort_values("anomaly_percentile", ascending=False).head(25)[cols_show],
    use_container_width=True,
)

# Download button
csv_download = to_csv_download(scored_df)
st.download_button(
    label="📥 Download scored CSV",
    data=csv_download,
    file_name="scored_" + csv_file.name,
    mime="text/csv",
)
