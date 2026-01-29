# data_engineer.py
import pandas as pd
import numpy as np
import hashlib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DataEngineeringPipeline:
    def __init__(self, input_file, clean_output, review_output):
        self.input_file = input_file
        self.clean_output = clean_output
        self.review_output = review_output
        self.df = None

    def run_pipeline(self):
        print("👷 [ENGINEER] Starting Pipeline...")
        self._extract()
        self._transform_clean()
        self._transform_features()
        self._load()
        print("👷 [ENGINEER] Pipeline Complete.\n")

    def _extract(self):
        # Handle both Excel and CSV formats
        try:
            self.df = pd.read_excel(self.input_file)
        except:
            try:
                self.df = pd.read_excel(self.input_file, sheet_name=0)
            except:
                self.df = pd.read_csv(self.input_file)

        # CRITICAL: Standardize column headers (Your data has 'PURCHASE_TS', this makes it 'purchase_ts')
        self.df.columns = self.df.columns.str.lower().str.strip().str.replace(" ", "_")
        print(f"   - Loaded {len(self.df)} raw rows.")

    def _transform_clean(self):
        # 1. Drop Junk Columns (detected in your analysis.ipynb as Unnamed: 12, 13)
        drop_cols = [c for c in self.df.columns if "unnamed" in c]
        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True)
            print(f"   - Dropped junk columns: {drop_cols}")

        # 2. Fix Data Types
        date_cols = ["purchase_ts", "ship_ts", "refund_ts"]
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        if "usd_price" in self.df.columns:
            self.df["usd_price"] = pd.to_numeric(self.df["usd_price"], errors="coerce")

    def _transform_features(self):
        # 1. Feature: Is Refunded?
        if "refund_ts" in self.df.columns:
            self.df["is_refunded"] = self.df["refund_ts"].notnull().astype(int)

        # 2. Feature: Shipping Speed
        if "ship_ts" in self.df.columns and "purchase_ts" in self.df.columns:
            self.df["days_to_ship"] = (
                self.df["ship_ts"] - self.df["purchase_ts"]
            ).dt.days

        # 3. Feature: User Anonymization (SHA-256)
        if "user_id" in self.df.columns:
            self.df["user_hash"] = (
                self.df["user_id"]
                .astype(str)
                .apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
            )

    def _load(self):
        # LOGIC: Filter "Bad Data"
        # 1. Invalid Country Codes (Regex A-Z)
        mask_country = ~self.df["country_code"].astype(str).str.match(r"^[A-Z]{2}$")

        # 2. Logic Error: Ship Date before Purchase Date
        mask_dates = self.df["ship_ts"] < self.df["purchase_ts"]

        # 3. Critical Missing Data
        mask_missing = (
            self.df["purchase_ts"].isnull()
            | (self.df["usd_price"] <= 0)
            | self.df["usd_price"].isnull()
        )

        # Combine Error Masks
        total_bad_mask = mask_country | mask_dates | mask_missing

        clean_df = self.df[~total_bad_mask]
        bad_df = self.df[total_bad_mask]

        # Save Files
        clean_df.to_excel(self.clean_output, index=False)

        # Save Bad Data with reason (Optional enhancement)
        bad_df.to_excel(self.review_output, index=False)

        print(f"   - Saved {len(clean_df)} clean rows to {self.clean_output}")
        print(f"   - Isolated {len(bad_df)} error rows to {self.review_output}")
