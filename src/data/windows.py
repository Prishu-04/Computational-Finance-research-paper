import numpy as np
import pandas as pd

FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "vwap", "india_vix",
    "log_return_1", "volatility", "sma_6", "sma_12", "rsi", "macd_hist",
    "minutes_since_open", "session_position",
]

def make_windows(df: pd.DataFrame, context_bars: int, horizon_bars: int):
    X, y_return, meta = [], [], []
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        n = len(g)
        for i in range(context_bars, n - horizon_bars):
            ctx = g.iloc[i - context_bars:i]          # strictly PAST bars, up to and including t
            target_row = g.iloc[i + horizon_bars]       # bar at t + horizon
            anchor_close = g.iloc[i]["close"]            # price AT decision time t
            ret = np.log(target_row["close"] / anchor_close)
            X.append(ctx[FEATURE_COLS].values)
            y_return.append(ret)
            meta.append({"date": day, "decision_ts": g.iloc[i]["timestamp"],
                         "target_ts": target_row["timestamp"]})
    X = np.array(X)
    y_return = np.array(y_return)
    meta = pd.DataFrame(meta)
    return X, y_return, meta


def fit_direction_threshold(y_return_train: np.ndarray, quantile: float = 0.33) -> float:
    return float(np.quantile(np.abs(y_return_train), 1 - 2 * quantile))


def to_direction(y_return: np.ndarray, tau: float) -> np.ndarray:
    direction = np.where(y_return > tau, "UP", np.where(y_return < -tau, "DOWN", "NEUTRAL"))
    return direction


def walk_forward_split(meta: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    days = sorted(meta["date"].unique())
    n_days = len(days)
    n_train = int(n_days * train_frac)
    n_val = int(n_days * val_frac)
    train_days = set(days[:n_train])
    val_days = set(days[n_train:n_train + n_val])
    test_days = set(days[n_train + n_val:])

    train_mask = meta["date"].isin(train_days).values
    val_mask = meta["date"].isin(val_days).values
    test_mask = meta["date"].isin(test_days).values
    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    df = pd.read_csv("data/processed/featured_data.csv", parse_dates=["timestamp", "decision_ts", "target_ts"] if False else ["timestamp"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    CONTEXT_BARS = 9   # 45 min of 5-min bars
    HORIZON_BARS = 12  # 60 min ahead

    X, y_return, meta = make_windows(df, CONTEXT_BARS, HORIZON_BARS)
    train_mask, val_mask, test_mask = walk_forward_split(meta)

    tau = fit_direction_threshold(y_return[train_mask])
    y_direction = to_direction(y_return, tau)

    print("X shape:", X.shape)
    print("Train/Val/Test days:", train_mask.sum(), val_mask.sum(), test_mask.sum())
    print("tau:", round(tau, 6))
    print("Direction distribution (train):",
          pd.Series(y_direction[train_mask]).value_counts(normalize=True).round(3).to_dict())

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y_return.npy", y_return)
    np.save("data/processed/y_direction.npy", y_direction)
    np.save("data/processed/train_mask.npy", train_mask)
    np.save("data/processed/val_mask.npy", val_mask)
    np.save("data/processed/test_mask.npy", test_mask)
    meta.to_csv("data/processed/window_meta.csv", index=False)