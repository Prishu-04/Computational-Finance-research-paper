# Optimized Moirai — Intraday Financial Forecasting

A five-tier model ladder for 60-minute-ahead direction forecasting on 5-minute
intraday bars — **Naive → Logistic Regression → Random Forest → XGBoost → SVR
→ LSTM → GRU → Transformer → PatchTST → CARD → Moirai → Optimized Moirai** —
plus a website that turns the pipeline's output into an interactive
prediction simulator and model comparison dashboard.

This README covers two things: **how the pipeline's data flows from raw CSV
to trained models**, and **how the website sits on top of it**.

---

## 1. Data flow — raw CSV to trained models

Everything downstream is deterministic given `data/raw/raw_data.csv` and a
fixed seed (`SEED = 42`). Each stage reads the previous stage's output from
disk and writes its own; nothing is passed in memory between stages, which
is what lets the website re-run any stage independently and lets
`generate_dashboard.py` / `webapp/app.py` rebuild dashboard data from disk at
any time.

```mermaid
flowchart TD
    A["Stage 1 — Raw data\ndata/raw/raw_data.csv\n5-min OHLCV + vwap + india_vix, 60 days"]

    A --> B["Stage 2 — Clean\nsrc/preprocessing/Datacleaning.py\nload_raw → filter_market_session (09:15–15:30) → handle_missing"]
    B --> B1[("data/processed/clean_data.csv")]

    B1 --> C["Stage 3 — Feature engineering\nsrc/features/features.py\nadd_returns → add_volatility → add_moving_averages →\nadd_rsi → add_macd → add_time_features"]
    C --> C1[("data/processed/featured_data.csv\n17 columns")]

    C1 --> D["Stage 4 — Window generation\nsrc/data/windows.py\nmake_windows: 45-min context, 60-min horizon, per day\nfit_direction_threshold(train only) → to_direction\nwalk_forward_split: 70/15/15 by calendar day"]
    D --> D1[("X.npy, y_return.npy, y_direction.npy\ntrain/val/test_mask.npy, window_meta.csv")]

    D1 --> E["Stage 5 — Leakage gate\nsrc/evaluation/leakage_tests.py\nchronological order · target-after-decision ·\nno cross-day windows · train-only-τ check"]
    E -- "any assert fails → STOP" --> X["❌ nothing downstream is trusted"]
    E -- "all pass" --> F

    F["Stage 6a — Train Tier 1 (classical)\nsrc/models/run_classical_baselines.py\nNaive → LogReg → Random Forest → XGBoost → SVR"]
    G["Stage 6b — Train Tiers 2–5 (deep learning)\nsrc/models/run_deep_learning_tiers.py\nLSTM, GRU, Transformer, PatchTST, CARD,\nMoirai, Moirai+FinEncoder, Optimized Moirai (+Regime)"]
    D1 --> F
    D1 --> G

    F --> H1[("results/tables/tier1_classical_ml_results.csv")]
    G --> H2[("results/tables/tier2_5_deep_learning_results.csv")]
    F -.-> L[("experiments/experiment_log.csv\none row per model, EXP-00N")]
    G -.-> L

    H1 --> I["Stage 7 — Merge ladder\ngenerate_dashboard.py: ensure_full_ladder_table()\n(built automatically if missing/stale — no manual step)"]
    H2 --> I
    I --> I1[("results/tables/full_model_ladder_results.csv\ntier-tagged, every metric")]

    C1 --> J["build_market_json()\nper-day bars + decision/target windows"]
    D1 --> J
    I1 --> K["build_metrics_json()"]

    J --> M["Stage 8 — Website\nwebapp/app.py serves this live via\nGET /api/market and GET /api/metrics"]
    K --> M
    M --> N["Browser dashboard\nPrediction simulator · Model ladder · Data & pipeline"]
```

**The one rule enforced at every stage:** anything computed from training
data (scaler stats, the direction threshold τ, model weights) is fit once on
`train` and only ever *applied* — never refit — to `val`/`test`. Stage 5
exists specifically to catch violations of this before any model sees the
data.

---

## 2. The website — how a browser click turns into a pipeline run

The website (`webapp/`) doesn't add a second copy of any pipeline logic. It
imports `build_market_json` / `build_metrics_json` / `ensure_full_ladder_table`
directly from `src/visualization/generate_dashboard.py`, and it runs Stages
2–6 as real subprocesses — the same commands you'd type in a terminal.

```mermaid
sequenceDiagram
    participant U as Browser (you)
    participant F as Flask backend (webapp/app.py)
    participant P as Pipeline scripts (src/...)
    participant D as Disk (data/, results/)

    U->>F: GET /
    F->>D: read data/processed/*, results/tables/*
    D-->>F: (empty on first run)
    F-->>U: dashboard.html — lands on "Data & pipeline" tab

    U->>F: POST /api/upload (raw_data.csv)
    F->>D: write data/raw/raw_data.csv (old file backed up)
    F-->>U: {ok: true, columns: [...]}

    U->>F: POST /api/run
    F->>P: Datacleaning.py
    P->>D: clean_data.csv
    F->>P: features.py
    P->>D: featured_data.csv
    F->>P: windows.py
    P->>D: X.npy, masks, window_meta.csv
    F->>P: leakage_tests.py
    Note over F,P: stops here and reports the error if any assert fails
    F->>P: run_classical_baselines.py
    P->>D: tier1_classical_ml_results.csv
    F->>P: run_deep_learning_tiers.py
    P->>D: tier2_5_deep_learning_results.csv
    F->>D: ensure_full_ladder_table() → full_model_ladder_results.csv

    loop every 900ms
        U->>F: GET /api/status
        F-->>U: stage checklist + live log lines
    end

    U->>F: (auto) reload page
    F->>D: read data/processed/*, results/tables/* (now populated)
    F-->>U: GET /api/market, GET /api/metrics
    U->>U: renders Prediction simulator + Model ladder tabs
```

---

## 3. Directory structure

```
Computational-Finance-research-paper/
├── data/
│   ├── raw/raw_data.csv                 # Stage 1 — only real input; replace to use new data
│   └── processed/                       # Stages 2–4 output (regenerated by the pipeline)
├── src/
│   ├── preprocessing/Datacleaning.py    # Stage 2
│   ├── features/features.py             # Stage 3
│   ├── data/windows.py                  # Stage 4
│   ├── evaluation/leakage_tests.py      # Stage 5 (hard gate)
│   ├── models/
│   │   ├── run_classical_baselines.py   # Stage 6a — Tier 1
│   │   ├── run_deep_learning_tiers.py   # Stage 6b — Tiers 2–5
│   │   ├── architectures.py             # LSTM/GRU/Transformer/PatchTST/CARD/Moirai classes
│   │   └── common.py                    # shared train/evaluate loop for Tiers 2–5
│   └── visualization/
│       ├── generate_dashboard.py        # Stage 7/8 — merges ladder table, builds dashboard JSON
│       └── dashboard_template.html      # used by generate_dashboard.py for a static export
├── results/
│   ├── tables/                          # per-tier + merged model ladder CSVs
│   └── dashboard/                       # static HTML export (optional, see below)
├── experiments/experiment_log.csv       # one row per trained model, auto-incrementing EXP-00N
└── webapp/                              # the website
    ├── app.py                           # Flask backend — see section 2
    ├── templates/dashboard.html         # the entire frontend: 3 tabs, one page
    └── requirements.txt                 # just Flask
```

---

## 4. How to run

### Option A — the website (recommended)

```bash
cd Computational-Finance-research-paper
pip install -r requirements.txt          # pandas, numpy, scikit-learn, xgboost, torch
pip install -r webapp/requirements.txt   # Flask
python webapp/app.py
```

Open **http://localhost:5000/**. Three tabs, one page:

- **Prediction simulator** — step or play through any of the 60 trading
  days, watch a chosen model call the next 60 minutes at every 5-minute
  decision point, and see it resolve correct/wrong against the real label.
- **Model ladder** — direction accuracy, F1, MASE, latency, and CRPS/coverage
  for the Moirai family, across all 13 models.
- **Data & pipeline** — upload a new `raw_data.csv` and run Stages 2–6 from
  the browser, with a live stage checklist and log console. The page reloads
  itself when the run finishes.

If `data/processed/` and `results/tables/` are already populated (true for
this zip as delivered), the first two tabs work immediately. If they're
empty, the site lands on the "Data & pipeline" tab automatically.

### Option B — command line, stage by stage

```bash
cd Computational-Finance-research-paper
python src/preprocessing/Datacleaning.py
python src/features/features.py
python src/data/windows.py
python src/evaluation/leakage_tests.py         # stop if this fails
python src/models/run_classical_baselines.py
python src/models/run_deep_learning_tiers.py
python src/visualization/generate_dashboard.py # writes results/dashboard/intraday_direction_ladder.html
```

The last command produces a single self-contained HTML file you can open
directly, email, or archive — no server needed. It's a snapshot at the time
you ran it; the website (Option A) always reflects whatever's on disk right
now.

---

## 5. About the dataset

`data/raw/raw_data.csv` is **synthetic**, not real market data — see
`README_dataset.md` for what it reproduces and where to get real NSE Nifty-50
minute data instead. No code changes are needed to swap it in: keep the same
column names (`timestamp, open, high, low, close, volume, ...`) and either
replace the file directly or upload it through the "Data & pipeline" tab.
