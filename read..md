# 🎮 GameZone: Enterprise Data Science Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20|%20ScikitLearn%20|%20Statsmodels-orange)

## **📖 Project Overview**
This project is an end-to-end **Business Intelligence & Predictive Modeling Pipeline** designed for the "GameZone" e-commerce dataset. It automates the transition from raw transaction data to actionable strategic insights using advanced Machine Learning techniques.

The system is architected into three domain-specific modules:
1. **👷 Data Engineering:** Cleans, validates, and transforms raw data using an automated ETL process.
2. **📊 Data Analysis:** Generates retrospective business reports (Revenue, Ops efficiency, Marketing segmentation).
3. **🧪 Data Science:** Deploys 5 advanced Machine Learning models for Churn, Forecasting, Segmentation, Inventory, and Anomaly Detection.

---

## **📂 Project Structure**

```text
GameZone_Project/
│
├── 📄 main.py                  # The Master Orchestrator (Configures paths & runs the sequence)
├── 📄 data_engineer.py         # ETL Pipeline (Cleaning, Hashing, Validation)
├── 📄 data_analyst.py          # Descriptive Analytics (Excel Reporting)
├── 📄 data_scientist.py        # Predictive Modeling (AI, ML & Visualization)
├── 📄 requirements.txt         # List of Python dependencies
├── 📓 analysis.ipynb           # Initial Exploratory Data Analysis (EDA) & Logic Checks
│
├── 📂 Input/
│   └── gamezone-orders-data (1).xlsx  # RAW DATA (Place file here)
│
└── 📂 Outputs/                 # (Generated Automatically)
    ├── gold_master_data.xlsx          # Cleaned, production-ready data
    ├── data_quality_review.xlsx       # Rejected data (Invalid countries/dates)
    ├── business_insights.xlsx         # Multi-tab Business Report
    ├── model_churn_risk.xlsx          # List of high-risk customers
    ├── model_inventory_opt.xlsx       # Safety stock recommendations
    ├── model_forecast.xlsx            # 12-week revenue predictions
    ├── model_anomalies.xlsx           # Detected data irregularities
    └── *.png                          # Visual Charts (5 professional plots)
