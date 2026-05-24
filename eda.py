# ==========================================
# Exploratory Data Analysis for data15k
# ==========================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import mysql.connector
except ImportError:
    mysql = None


TARGET_COL = "label"
LEAKAGE_COLUMNS = ["future_return_5m", "is_pump", "is_pump_target"]
EDA_FEATURES = [
    "rsi",
    "vol_ratio",
    "price_zscore",
    "volatility_5m",
    "ema_diff",
    "high_low_pct",
    "return_1m",
    "return_15m",
]


def load_data():
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_database = os.getenv("MYSQL_DATABASE", "data15k")
    mysql_table = os.getenv("MYSQL_TABLE", "data15k")

    if mysql_host and mysql is not None:
        conn = mysql.connector.connect(
            host=mysql_host,
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=mysql_database,
        )
        try:
            return pd.read_sql(f"SELECT * FROM {mysql_table}", conn), f"MySQL table {mysql_table}"
        finally:
            conn.close()

    return pd.read_csv("data15k.csv"), "CSV data15k.csv"


def main():
    df, source_name = load_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    available_features = [col for col in EDA_FEATURES if col in df.columns]
    clean_df = df.dropna(subset=available_features + [TARGET_COL]).copy()

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print("Source:", source_name)
    print("Raw shape:", df.shape)
    print("Clean EDA shape:", clean_df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Inf values replaced:", int(inf_count))
    print("Time range:", clean_df["timestamp"].min(), "to", clean_df["timestamp"].max())

    print("\n" + "=" * 50)
    print("TARGET DISTRIBUTION")
    print("=" * 50)
    print(clean_df[TARGET_COL].value_counts().sort_index())
    print(clean_df[TARGET_COL].value_counts(normalize=True).sort_index().round(4))

    print("\n" + "=" * 50)
    print("FEATURES USED IN EDA")
    print("=" * 50)
    print(available_features)

    corr_cols = available_features + [TARGET_COL]
    plt.figure(figsize=(12, 8))
    corr = clean_df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix Without Leakage Columns")
    plt.tight_layout()
    plt.show()

    pump_df = clean_df[clean_df[TARGET_COL] == 1]
    normal_df = clean_df[clean_df[TARGET_COL] == 0]

    for col in ["rsi", "vol_ratio", "price_zscore"]:
        if col not in clean_df.columns:
            continue

        plt.figure(figsize=(10, 5))
        plt.hist(normal_df[col], bins=50, alpha=0.5, label="Normal")
        plt.hist(pump_df[col], bins=50, alpha=0.5, label="Pump")
        plt.title(f"{col} Distribution by Target")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    for col in ["rsi", "vol_ratio", "price_zscore", "volatility_5m"]:
        if col not in clean_df.columns:
            continue

        plt.figure(figsize=(8, 4))
        sns.boxplot(data=clean_df, x=TARGET_COL, y=col)
        plt.title(f"{col} vs Target Label")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
