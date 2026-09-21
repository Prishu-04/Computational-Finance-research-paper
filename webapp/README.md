# Intraday Direction Ladder — website

A small Flask backend that serves the dashboard's data live from whatever is
currently in `data/processed/` and `results/tables/`, plus a page to upload a
new dataset and run the whole pipeline from the browser.

## Setup (one time)

```bash
cd Computational-Finance-research-paper
pip install -r requirements.txt          # pandas, numpy, scikit-learn, xgboost, torch, ...
pip install -r webapp/requirements.txt   # Flask
```

## Run

```bash
python webapp/app.py
```

Then open:
- `http://localhost:5000/` — the dashboard (prediction simulator + model ladder)
- `http://localhost:5000/pipeline` — upload a dataset and run the pipeline

If `data/processed/` and `results/tables/` already have output in them (e.g.
you unzipped this project as-is), the dashboard works immediately. If they're
empty, the dashboard shows an empty-state screen with a link to `/pipeline`.

## How it stays in sync with the pipeline

`webapp/app.py` doesn't have its own copy of the data-loading logic — it
imports `build_market_json` / `build_metrics_json` / `ensure_full_ladder_table`
directly from `src/visualization/generate_dashboard.py`. Every `GET
/api/market` and `GET /api/metrics` call re-reads the CSV/`.npy` files on
disk and rebuilds the JSON from scratch, so:

- Uploading a new dataset and clicking **Run pipeline** on `/pipeline` re-runs
  Stages 2–6 (clean → features → windows → leakage gate → train Tier 1 →
  train Tiers 2-5) as real subprocesses, streams their stdout into a log
  panel, and stops immediately if the leakage gate fails.
- The moment training finishes, reloading `/` shows the new data — there is
  no intermediate HTML file to regenerate.
- The standalone `python src/visualization/generate_dashboard.py` script
  still works if you just want a single static HTML file to email or archive
  instead of running a server.

## Endpoints

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Dashboard page |
| `/pipeline` | GET | Upload + run control page |
| `/api/market` | GET | Live JSON of every trading day's bars + prediction windows |
| `/api/metrics` | GET | Live JSON of the full model ladder |
| `/api/meta` | GET | Last-modified timestamps for the raw data and results |
| `/api/upload` | POST | Multipart CSV upload → `data/raw/raw_data.csv` (old file backed up, not deleted) |
| `/api/run` | POST | Starts the pipeline in a background thread (409 if one is already running) |
| `/api/status` | GET | Poll target: per-stage status + running log, for the `/pipeline` page |

## Notes

- This runs Flask's development server (`app.run(...)`), which is fine for
  local/single-user use. For anything multi-user or internet-facing, put it
  behind a production WSGI server (gunicorn/waitress) and a reverse proxy.
- Only one pipeline run is allowed at a time; a second `POST /api/run` while
  one is in progress returns `409`.
- Uploaded CSVs must have at least `timestamp, open, high, low, close,
  volume` columns; `vwap` and `india_vix` are optional, matching the shape
  of the original `data/raw/raw_data.csv`.
