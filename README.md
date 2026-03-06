# 💱 FX Exposure Analytics & Anomaly Detection

> A Python + Streamlit application for analyzing foreign exchange (FX) exposure, detecting anomalies using machine learning, and generating risk-based insights from ERP-style transaction data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit) ![ML](https://img.shields.io/badge/ML-Anomaly%20Detection-orange)

---

## 📌 Overview

This project simulates an **enterprise FX exposure management workflow** — the kind used in corporate treasury and finance operations to monitor currency risk across business units and geographies.

It includes:
- Synthetic FX exposure data generation (multi-currency, multi-entity)
- **ML-based anomaly detection** using Isolation Forest
- An interactive **Streamlit dashboard** for drill-down analysis
- Scored output datasets for downstream reporting

---

## 🗂️ Project Structure

```
FX_exposure/
├── app.py                              # Streamlit dashboard
├── train_fx_anomaly_model.py            # Train Isolation Forest anomaly model
├── test_fx_anomaly_model.py             # Unit tests for the anomaly model
├── fx_anomaly_pipeline.pkl             # Serialized trained ML pipeline
├── synthetic_fx_exposure_data.csv      # Synthetic FX exposure dataset
├── fx_exposure_with_anomaly_flags.csv  # Dataset with anomaly labels applied
├── june2025_erp_extract.csv            # Sample ERP extract (synthetic)
├── june2025_scored.csv                 # Scored output with anomaly flags
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Key Features

| Feature | Description |
|---|---|
| 📊 Multi-Currency Exposure | Track exposure across USD, EUR, GBP, JPY, and more |
| 🤖 Anomaly Detection | Isolation Forest ML model flags unusual FX transactions |
| 🏢 Entity-Level Drill-Down | Breakdown by legal entity, business unit, or region |
| 📉 Risk Scoring | Every transaction gets a scored anomaly probability |
| 📊 Streamlit Dashboard | Interactive charts, filters, and exposure summaries |
| 🧪 ERP-Style Data | Mimics real SAP/Oracle ERP extracts for realism |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nmadagi/FX_exposure.git
cd FX_exposure
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the anomaly model (optional — pre-trained pkl included)
```bash
python train_fx_anomaly_model.py
```

### 4. Run the Streamlit dashboard
```bash
streamlit run app.py
```

---

## 🤖 ML Model: Isolation Forest

The anomaly detection pipeline uses **Isolation Forest**, an unsupervised algorithm well-suited for detecting outliers in high-dimensional financial data.

- Features include: notional amount, currency pair, entity, settlement date, transaction type
- Outputs an **anomaly score** and a binary **flag** (normal / anomalous)
- Trained pipeline serialized to `fx_anomaly_pipeline.pkl` for reuse

---

## 📦 Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive dashboard
- **Scikit-learn** — Isolation Forest anomaly detection
- **Pandas / NumPy** — Data wrangling
- **Plotly** — Visualizations
- **Pickle** — Model serialization

---

## ⚠️ Disclaimer

All data in this repository is **fully synthetic**. No real company, bank, or client data is used.

---

## 👤 Author

**Nitin Madagi** | [GitHub](https://github.com/nmadagi) | [Portfolio](https://nmadagi.github.io/portfolio)
