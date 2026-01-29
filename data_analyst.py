# data_analyst.py
import pandas as pd
import datetime as dt


class BusinessAnalyst:
    def __init__(self, input_file, report_file):
        self.input_file = input_file
        self.report_file = report_file
        self.df = None

    def generate_report(self):
        print("📊 [ANALYST] Generating Business Report...")
        self.df = pd.read_excel(self.input_file)

        # Create Excel Writer with multiple sheets
        with pd.ExcelWriter(self.report_file, engine="openpyxl") as writer:
            self._executive_summary().to_excel(
                writer, sheet_name="Executive_Summary", index=False
            )
            self._ops_metrics().to_excel(
                writer, sheet_name="Operations_Speed", index=False
            )
            self._marketing_segments().to_excel(
                writer, sheet_name="Marketing_RFM", index=False
            )

        print(f"📊 [ANALYST] Report Saved: {self.report_file}\n")

    def _executive_summary(self):
        # Group by Month
        self.df["month"] = pd.to_datetime(self.df["purchase_ts"]).dt.to_period("M")

        summary = (
            self.df.groupby("month")
            .agg(
                Revenue=("usd_price", "sum"),
                Total_Orders=("order_id", "count"),
                Refund_Loss=(
                    "usd_price",
                    lambda x: x[self.df.loc[x.index, "is_refunded"] == 1].sum(),
                ),
            )
            .reset_index()
        )

        summary["Refund_Rate_Pct"] = (
            summary["Refund_Loss"] / summary["Revenue"] * 100
        ).round(2)
        summary = summary.sort_values("month")
        # Convert period back to string for Excel compatibility
        summary["month"] = summary["month"].astype(str)
        return summary

    def _ops_metrics(self):
        # Analyze Shipping Speed vs Refunds
        # We categorize speed into buckets
        self.df["speed_bucket"] = pd.cut(
            self.df["days_to_ship"],
            bins=[-1, 0, 2, 5, 100],
            labels=["Same Day", "1-2 Days", "3-5 Days", "Late (>5 Days)"],
        )

        ops = (
            self.df.groupby("speed_bucket")
            .agg(Order_Count=("order_id", "count"), Refund_Rate=("is_refunded", "mean"))
            .reset_index()
        )

        ops["Refund_Rate"] = (ops["Refund_Rate"] * 100).round(2)
        return ops

    def _marketing_segments(self):
        # RFM Analysis
        current_date = pd.to_datetime(self.df["purchase_ts"]).max() + dt.timedelta(
            days=1
        )

        rfm = (
            self.df.groupby("user_id")
            .agg(
                Recency=("purchase_ts", lambda x: (current_date - x.max()).days),
                Frequency=("order_id", "count"),
                Monetary=("usd_price", "sum"),
            )
            .reset_index()
        )

        # Assign Segments
        # VIP = Top 20% Spenders
        # At Risk = Haven't bought in 100+ days
        monetary_threshold = rfm["Monetary"].quantile(0.8)

        def assign_segment(row):
            if row["Monetary"] >= monetary_threshold:
                return "VIP"
            if row["Recency"] > 100:
                return "At Risk"
            return "Standard"

        rfm["Segment"] = rfm.apply(assign_segment, axis=1)

        # Return count of users per segment
        return (
            rfm["Segment"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Segment", "Segment": "User_Count"})
        )
