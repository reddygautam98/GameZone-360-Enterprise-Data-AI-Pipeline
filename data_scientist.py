# data_scientist.py
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced Models
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings
warnings.filterwarnings("ignore")

# Set professional chart style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class DataScientist:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None

    def run_advanced_models(self):
        print("🧪 [SCIENTIST] Running Advanced Models & Visualizations...")
        self.df = pd.read_excel(self.input_file)

        # Run specific modules
        self._run_churn_prediction()
        self._run_customer_segmentation()  # <--- NEW MODULE
        self._run_inventory_optimization()
        self._run_forecasting()
        self._run_anomaly_detection()  # <--- NEW MODULE

        print("🧪 [SCIENTIST] Modeling & Charting Complete.\n")

    def _run_churn_prediction(self):
        # 1. Feature Engineering
        current_date = pd.to_datetime(self.df["purchase_ts"]).max()
        user_features = (
            self.df.groupby("user_id")
            .agg(
                recency=("purchase_ts", lambda x: (current_date - x.max()).days),
                frequency=("order_id", "count"),
                monetary=("usd_price", "sum"),
            )
            .reset_index()
        )

        # 2. Define Target
        user_features["is_churned"] = (user_features["recency"] > 120).astype(int)

        # 3. Train Model
        X = user_features[["recency", "frequency", "monetary"]]
        y = user_features["is_churned"]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # 4. Predict Risk Score
        user_features["Churn_Risk_Score"] = (model.predict_proba(X)[:, 1] * 100).round(
            1
        )

        # --- 📊 CHART 1: CHURN DISTRIBUTION ---
        try:
            plt.figure()
            sns.histplot(
                user_features["Churn_Risk_Score"], bins=20, kde=True, color="red"
            )
            plt.title("Distribution of Customer Churn Risk Scores", fontsize=16)
            plt.xlabel("Churn Probability (%)")
            plt.ylabel("Number of Users")
            plt.axvline(
                70, color="black", linestyle="--", label="High Risk Threshold (70%)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig("chart_churn_distribution.png")
            plt.close()
            print("   - Chart Saved: chart_churn_distribution.png")
        except Exception as e:
            print(f"   ⚠️ Chart Error: {e}")

        # Save Data
        high_risk = user_features[user_features["Churn_Risk_Score"] > 70]
        high_risk.to_excel("model_churn_risk.xlsx", index=False)

    def _run_customer_segmentation(self):
        """
        NEW MODULE: Uses K-Means Clustering to group users into 'Personas'.
        (High Spenders, Loyalists, Occasional, etc.)
        """
        try:
            # Prepare RFM Data
            current_date = pd.to_datetime(self.df["purchase_ts"]).max()
            rfm = (
                self.df.groupby("user_id")
                .agg(
                    Recency=("purchase_ts", lambda x: (current_date - x.max()).days),
                    Frequency=("order_id", "count"),
                    Monetary=("usd_price", "sum"),
                )
                .reset_index()
            )

            # Scale Data (Crucial for K-Means)
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

            # Train K-Means (Assuming 4 distinct groups)
            kmeans = KMeans(n_clusters=4, random_state=42)
            rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

            # --- 📊 CHART 2: CUSTOMER SEGMENTS ---
            plt.figure()
            sns.scatterplot(
                data=rfm,
                x="Frequency",
                y="Monetary",
                hue="Cluster",
                palette="deep",
                s=100,
                alpha=0.7,
            )
            plt.title("Customer Segmentation: Value vs. Frequency", fontsize=16)
            plt.xlabel("Number of Orders")
            plt.ylabel("Total Spend ($)")
            plt.yscale("log")  # Log scale handles "Whale" customers better
            plt.tight_layout()
            plt.savefig("chart_customer_segments.png")
            plt.close()
            print("   - Chart Saved: chart_customer_segments.png")

            rfm.to_excel("model_customer_segments.xlsx", index=False)

        except Exception as e:
            print(f"   ⚠️ Segmentation Error: {e}")

    def _run_inventory_optimization(self):
        # Group by Product
        daily_sales = (
            self.df.groupby(["product_name", "purchase_ts"])
            .agg(qty=("order_id", "count"))
            .reset_index()
        )

        stats = (
            daily_sales.groupby("product_name")
            .agg(
                avg_daily_demand=("qty", "mean"),
                max_daily_demand=("qty", "max"),
                std_dev_demand=("qty", "std"),
            )
            .reset_index()
            .fillna(0)
        )

        # Lead Time inputs
        max_lead_time = 7
        avg_lead_time = 3

        # Safety Stock Formula
        stats["Recommended_Safety_Stock"] = (
            (stats["max_daily_demand"] * max_lead_time)
            - (stats["avg_daily_demand"] * avg_lead_time)
        ).round(0)

        # --- 📊 CHART 3: INVENTORY LEVELS ---
        try:
            top_items = stats.sort_values(
                "Recommended_Safety_Stock", ascending=False
            ).head(10)
            plt.figure()
            plot_data = top_items.melt(
                id_vars="product_name",
                value_vars=["avg_daily_demand", "Recommended_Safety_Stock"],
                var_name="Metric",
                value_name="Units",
            )

            sns.barplot(
                data=plot_data,
                x="Units",
                y="product_name",
                hue="Metric",
                palette="viridis",
            )
            plt.title(
                "Top 10 Products: Avg Demand vs. Required Safety Stock", fontsize=16
            )
            plt.tight_layout()
            plt.savefig("chart_inventory_levels.png")
            plt.close()
            print("   - Chart Saved: chart_inventory_levels.png")
        except Exception as e:
            print(f"   ⚠️ Chart Error: {e}")

        stats.to_excel("model_inventory_opt.xlsx", index=False)

    def _run_forecasting(self):
        # Time Series: Weekly Revenue
        ts = self.df.set_index("purchase_ts").resample("W")["usd_price"].sum().fillna(0)

        try:
            model = ExponentialSmoothing(ts, seasonal="add", seasonal_periods=52).fit()
            forecast = model.forecast(12)  # Next 12 weeks

            # --- 📊 CHART 4: FORECAST TRAJECTORY ---
            plt.figure()
            last_6_months = ts.tail(26)
            plt.plot(
                last_6_months.index,
                last_6_months.values,
                label="Historical Revenue",
                marker="o",
            )
            plt.plot(
                forecast.index,
                forecast.values,
                label="AI Forecast (Next 12 Wks)",
                color="green",
                linestyle="--",
                marker="x",
            )
            plt.title("Revenue Forecast: Next Quarter Trajectory", fontsize=16)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("chart_revenue_forecast.png")
            plt.close()
            print("   - Chart Saved: chart_revenue_forecast.png")

            # Save Data
            forecast_df = pd.DataFrame(
                {"Week": forecast.index, "Predicted_Revenue": forecast.values}
            )
            forecast_df.to_excel("model_forecast.xlsx", index=False)

        except Exception as e:
            print(f"   - Forecasting skipped: {e}")

    def _run_anomaly_detection(self):
        """
        NEW MODULE: Uses Isolation Forest to find revenue anomalies (Fraud/System Failure).
        """
        try:
            # Aggregate to daily revenue
            daily_rev = self.df.groupby("purchase_ts")["usd_price"].sum().reset_index()
            daily_rev.columns = ["Date", "Revenue"]

            # Train Isolation Forest (Contamination = 5% of data might be outliers)
            iso = IsolationForest(contamination=0.05, random_state=42)
            daily_rev["anomaly"] = iso.fit_predict(daily_rev[["Revenue"]])

            # Filter Anomalies (-1 means anomaly)
            anomalies = daily_rev[daily_rev["anomaly"] == -1]

            # --- 📊 CHART 5: ANOMALY DETECTION ---
            plt.figure()
            plt.plot(
                daily_rev["Date"],
                daily_rev["Revenue"],
                label="Normal Revenue",
                color="blue",
                alpha=0.6,
            )
            plt.scatter(
                anomalies["Date"],
                anomalies["Revenue"],
                color="red",
                label="Anomaly detected",
                s=50,
                zorder=5,
            )

            plt.title("Anomaly Detection: Unusual Revenue Spikes/Drops", fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.savefig("chart_anomalies.png")
            plt.close()
            print("   - Chart Saved: chart_anomalies.png")

            anomalies.to_excel("model_anomalies.xlsx", index=False)

        except Exception as e:
            print(f"   ⚠️ Anomaly Detection Error: {e}")
