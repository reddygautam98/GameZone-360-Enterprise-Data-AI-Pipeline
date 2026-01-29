# main.py
import os
from data_engineer import DataEngineeringPipeline
from data_analyst import BusinessAnalyst
from data_scientist import DataScientist

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Update this to match your exact file name
RAW_DATA_FILE = r"C:\Users\reddy\Downloads\gamezone-orders-data (1).xlsx"

# 2. Define Output Names
CLEAN_DATA_FILE = "gold_master_data.xlsx"
REVIEW_DATA_FILE = "data_quality_review.xlsx"
REPORT_FILE = "business_insights.xlsx"


def main():
    print("🚀 STARTING GAMEZONE DATA PROJECT...\n")

    # Check if input file exists
    if not os.path.exists(RAW_DATA_FILE):
        print(f"❌ CRITICAL ERROR: Input file '{RAW_DATA_FILE}' not found.")
        print("   Please make sure the Excel file is in this folder.")
        return

    # STEP 1: DATA ENGINEERING
    # Cleans data, fixes columns, creates features, removes bad rows
    engineer = DataEngineeringPipeline(RAW_DATA_FILE, CLEAN_DATA_FILE, REVIEW_DATA_FILE)
    engineer.run_pipeline()

    # STEP 2: DATA ANALYSIS
    # Generating reports for business stakeholders
    analyst = BusinessAnalyst(CLEAN_DATA_FILE, REPORT_FILE)
    analyst.generate_report()

    # STEP 3: DATA SCIENCE
    # Predictive modeling for Strategy team
    scientist = DataScientist(CLEAN_DATA_FILE)
    scientist.run_advanced_models()

    print("✅ PROJECT COMPLETE.")
    print(f"1. Clean Data: {CLEAN_DATA_FILE}")
    print(f"2. Errors:     {REVIEW_DATA_FILE}")
    print(f"3. Report:     {REPORT_FILE}")


if __name__ == "__main__":
    main()
