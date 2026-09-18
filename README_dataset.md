# Dataset — Phase 1 Deliverable

## File: `raw_data_SYNTHETIC.csv`
**This is synthetic, not real market data.** It's a schema-correct placeholder so you can
start Day 7–9 of your plan (preprocessing, feature engineering, window generation,
leakage-safe splitting, baseline models) immediately, without waiting on a download.
It reproduces the regime pattern from your original scenario — mild uptrend 9:15–12:00,
choppy 12:00–13:00, volatile 13:00–14:00, mild downtrend into close — across 60 trading
days at 5-minute resolution (9:15 AM–3:30 PM IST).

Columns: `date, timestamp, open, high, low, close, volume, vwap, india_vix`

**Do not report results computed on this file as your findings** — it's for pipeline
development only. Every number in it is randomly generated.

## Getting the real data
1. **Primary (recommended):** Kaggle — "NSE Nifty 50 Index Minute Data (2015–2026)"
   https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data
   - Free, weekly-updated, 1-minute OHLC. Resample to 5-min yourself (`resample('5min').agg(...)`)
     so you control exactly how bars are built — better for your leakage-safety story.
2. **Alternative (free, multi-timeframe):** GitHub `debaonline4u/NSE-Data`
   https://github.com/debaonline4u/NSE-Data
   - Already has 1/3/5/10/15/30/60-min bars for NIFTY 50 and Next 50, explicitly licensed
     for research/backtesting.
3. **If you need India VIX / Bank NIFTY as context features:** download these separately
   from NSE's historical data section or the same Kaggle uploader's other datasets, and
   join on timestamp.

## Swapping in real data
Keep the exact same column names (`timestamp, open, high, low, close, volume, ...`) and
drop the real file in as `data/raw/raw_data.csv` per the repo structure ChatGPT gave you.
No other code should need to change if you build your pipeline against this schema now.
