# ==========================================
# Build modeling dataset for crypto pump detection
# ==========================================

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DATASET = BASE_DIR / "pump_dump_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data15k.csv"
TARGET_COL = "label"
PUMP_THRESHOLD = 0.05
FUTURE_STEPS = 5
RANDOM_STATE = 42
MAX_ROWS = 15000


def load_source_data():
    if RAW_DATASET.exists():
        return pd.read_csv(RAW_DATASET)

    raw_files = sorted((BASE_DIR / "data").glob("*_raw.csv"))
    if not raw_files:
        raise FileNotFoundError("No pump_dump_dataset.csv or data/*_raw.csv files found.")

    frames = []
    for path in raw_files:
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            frame["symbol"] = path.stem.replace("_raw", "")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def add_technical_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])

    sort_cols = ["symbol", "timestamp"] if "symbol" in df.columns else ["timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    group_key = "symbol" if "symbol" in df.columns else None

    def by_symbol(series):
        return df.groupby(group_key)[series] if group_key else None

    close_group = by_symbol("close")
    volume_group = by_symbol("volume")

    if close_group is not None:
        df["return_1m"] = close_group.pct_change(1)
        df["return_5m"] = close_group.pct_change(5)
        df["return_15m"] = close_group.pct_change(15)
        df["return_30m"] = close_group.pct_change(30)
        df["return_1h"] = close_group.pct_change(60)
        df["vol_return_1m"] = volume_group.pct_change(1)
        df["vol_return_5m"] = volume_group.pct_change(5)
        df["vol_ratio"] = df["volume"] / volume_group.transform(lambda x: x.rolling(20, min_periods=5).mean())
        rolling_mean = close_group.transform(lambda x: x.rolling(20, min_periods=5).mean())
        rolling_std = close_group.transform(lambda x: x.rolling(20, min_periods=5).std())
        df["volatility_5m"] = close_group.pct_change().groupby(df["symbol"]).transform(lambda x: x.rolling(5, min_periods=3).std())
        ema_fast = close_group.transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema_slow = close_group.transform(lambda x: x.ewm(span=26, adjust=False).mean())
    else:
        df["return_1m"] = df["close"].pct_change(1)
        df["return_5m"] = df["close"].pct_change(5)
        df["return_15m"] = df["close"].pct_change(15)
        df["return_30m"] = df["close"].pct_change(30)
        df["return_1h"] = df["close"].pct_change(60)
        df["vol_return_1m"] = df["volume"].pct_change(1)
        df["vol_return_5m"] = df["volume"].pct_change(5)
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=5).mean()
        rolling_mean = df["close"].rolling(20, min_periods=5).mean()
        rolling_std = df["close"].rolling(20, min_periods=5).std()
        df["volatility_5m"] = df["close"].pct_change().rolling(5, min_periods=3).std()
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()

    delta = df.groupby("symbol")["close"].diff() if group_key else df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    if group_key:
        avg_gain = gain.groupby(df["symbol"]).transform(lambda x: x.rolling(14, min_periods=5).mean())
        avg_loss = loss.groupby(df["symbol"]).transform(lambda x: x.rolling(14, min_periods=5).mean())
    else:
        avg_gain = gain.rolling(14, min_periods=5).mean()
        avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["macd"] = ema_fast - ema_slow
    if group_key:
        df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    else:
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    df["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    df["momentum_5m"] = df["close"] - (close_group.shift(5) if close_group is not None else df["close"].shift(5))
    df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
    df["price_zscore"] = (df["close"] - rolling_mean) / rolling_std
    df["ema_diff"] = (ema_fast - ema_slow) / df["close"]

    return df


def create_target(df):
    df = df.copy()
    if "symbol" in df.columns:
        future_price = df.groupby("symbol")["close"].shift(-FUTURE_STEPS)
    else:
        future_price = df["close"].shift(-FUTURE_STEPS)

    df["future_return_5m"] = (future_price - df["close"]) / df["close"]
    df[TARGET_COL] = (df["future_return_5m"] > PUMP_THRESHOLD).astype(int)
    return df.drop(columns=["is_pump", "is_pump_target"], errors="ignore")


def clean_dataset(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df.dropna().drop_duplicates()


def sample_dataset(df):
    df = df.sort_values("timestamp")

    if len(df) <= MAX_ROWS:
        return df.reset_index(drop=True)

    positive_df = df[df[TARGET_COL] == 1]
    negative_df = df[df[TARGET_COL] == 0]
    negative_n = max(0, MAX_ROWS - len(positive_df))

    sampled_negative = negative_df.sample(
        n=min(negative_n, len(negative_df)),
        random_state=RANDOM_STATE,
    )
    sampled = pd.concat([positive_df, sampled_negative], ignore_index=True)
    return sampled.sort_values("timestamp").reset_index(drop=True)


def main():
    df = load_source_data()
    df = add_technical_features(df)
    df = create_target(df)
    final_df = clean_dataset(sample_dataset(df))

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", final_df.shape)
    print("Target distribution:")
    print(final_df[TARGET_COL].value_counts().sort_index())


if __name__ == "__main__":
    main()
