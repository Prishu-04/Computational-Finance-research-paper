### Rolling intraday forecasting windows + walk-forward chronological splitting.
Implements design:
  - context window: [t - context_len, t]   (what the model sees)
  - target window:  (t, t + horizon]       (what it must predict)
  - both windows must stay WITHIN the same trading day (no overnight leakage).
  - direction threshold tau is fit on TRAIN ONLY, then applied everywhere.
---
### Features
```Python
FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "vwap", "india_vix",
    "log_return_1", "volatility", "sma_6", "sma_12", "rsi", "macd_hist",
    "minutes_since_open", "session_position",
]
```
---
### Making of Window
    context_bars: number of 5-min bars in the input context (e.g. 9 bars = 45 min)
    horizon_bars: number of 5-min bars ahead to predict (e.g. 12 bars = 1 hour)
    Returns X (n_samples, context_bars, n_features), y_return, y_direction,
    and an index of (day, target_timestamp) for traceability.
```Python
def make_windows(df: pd.DataFrame, context_bars: int, horizon_bars: int):
    X, y_return, meta = [], [], []
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        n = len(g)
        for i in range(context_bars, n - horizon_bars):
            ctx = g.iloc[i - context_bars:i]          # strictly PAST bars, up to and including t
            target_row = g.iloc[i + horizon_bars]       # bar at t + horizon
            anchor_close = g.iloc[i]["close"]            # price AT decision time t
            ret = np.log(target_row["close"] / anchor_close)
            X.append(ctx[FEATURE_COLS].values)
            y_return.append(ret)
            meta.append({"date": day, "decision_ts": g.iloc[i]["timestamp"],
                         "target_ts": target_row["timestamp"]})
    X = np.array(X)
    y_return = np.array(y_return)
    meta = pd.DataFrame(meta)
    return X, y_return, meta
```
---
### Direction Threshold Fitting
tau chosen so that roughly `quantile` of TRAIN returns fall below -tau  (DOWN) and above +tau (UP), leaving the rest NEUTRAL. Fit on train only, per Section 5 -- never let test data influence tau.
```Python
def fit_direction_threshold(y_return_train: np.ndarray, quantile: float = 0.33) -> float:
    return float(np.quantile(np.abs(y_return_train), 1 - 2 * quantile))
```
---
### Direction
```Python
def to_direction(y_return: np.ndarray, tau: float) -> np.ndarray:
    direction = np.where(y_return > tau, "UP", np.where(y_return < -tau, "DOWN", "NEUTRAL"))
    return direction
```
---
### Walk Forward Split
Chronological split by unique trading day, NOT by row -- ensures no within-day leakage across the split boundary. Returns boolean masks aligned to `meta`'s row order.
```Python
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
```
---
### Output 
![[Pasted image 20260918123158.png]]