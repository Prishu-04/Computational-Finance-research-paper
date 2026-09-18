# CLeaning + market-session filtering.
### Pipeline Stage 1:
```
Raw OHLCV -> Cleaning -> Trading-session filtering -> Missing-value handling
```
---
### Load Raw Data
```Python
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
```
---
### Filter Market Session
Note : """Keep only bars inside the official trading session, per trading day."""
```Python
def filter_market_session(df: pd.DataFrame) -> pd.DataFrame:
    t = df["timestamp"].dt.time
    mask = (t >= pd.Timestamp(market_op).time()) & (t <= pd.Timestamp(market_cl).time())
    return df.loc[mask].reset_index(drop=True)
```
---
### Handling Missing
"""
    Forward-fill price gaps WITHIN a trading day only (never across days —
    that would leak the previous day's close into a new session's open).
    Volume gaps are filled with 0 (no trades), not fill.
"""
```Python
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    price_cols = ["open", "high", "low", "close", "vwap"]
    df[price_cols] = df.groupby("date")[price_cols].ffill()
    df["volume"] = df["volume"].fillna(0)
    if "india_vix" in df.columns:
        df["india_vix"] = df.groupby("date")["india_vix"].ffill()
    before = len(df)
    df = df.dropna(subset=price_cols).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[clean] dropped {before - after} rows with unrecoverable NaNs "
              f"(e.g. first bar of a day with no prior value to ffill from)")
    return df
```
---
### Run Cleaning data
```Python
def run_cleaning(raw_path: str) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = filter_market_session(df)
    df = handle_missing(df)
    return df
```
---
### Output 
![[Pasted image 20260918122853.png]]