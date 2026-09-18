# Feature Extraction
```
Feature engineering. EVERY feature here is computed using only data at or
before its own timestamp (rolling/expanding windows, no centered windows,
no .shift(-1) anywhere) -- this is what "no future information leakage"
means in practice.
```
---
### Add return
```python
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return_1"] = df.groupby("date")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return df
```
---
### Add Volatility
Note:
"""Rolling realized volatility of 1-bar log returns, within-day only."""

```Python
def add_volatility(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
        df = df.copy()
    df["volatility"] = df.groupby("date")["log_return_1"].transform(
        lambda s: s.rolling(window, min_periods=2).std()
    )
    return df
```
---
### Add Moving Averages
```Python
def add_moving_averages(df: pd.DataFrame, windows=(6, 12)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df.groupby("date")["close"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
    return df
```
---
### Add RSI
```Python
def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    def _rsi(close):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    df["rsi"] = df.groupby("date")["close"].transform(_rsi)
    return df
```
---
### Add Macd
```Python
def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    df = df.copy()
    def _macd(close):
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
        return macd_line - signal_line
    df["macd_hist"] = df.groupby("date")["close"].transform(_macd)
    return df
```
---
### Add Time Feature
Note :
"""Time-of-day / session-position features -- Candidate A from the roadmap."""
```Python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minutes_since_open = (
        (df["timestamp"] - df["timestamp"].dt.normalize()
         - pd.Timedelta(hours=9, minutes=15)).dt.total_seconds() / 60
    )
    session_len_min = (15 * 60 + 30) - (9 * 60 + 15)  # 375 min
    df["minutes_since_open"] = minutes_since_open
    df["session_position"] = (minutes_since_open / session_len_min).clip(0, 1)
    return df
```
---
### Final Dataframe Setup 
```Python
def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_returns(df)
    df = add_volatility(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_time_features(df)
    return df
```
![[Pasted image 20260918121612.png]]

---
### Output
![[Pasted image 20260918123110.png|700]]