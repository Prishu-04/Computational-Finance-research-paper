# Merge notes

This project was assembled from two independently built versions of the
same "Optimized Moirai" pipeline + dashboard:

- **c4** — built by Claude (this assistant) in an earlier session.
- **c2** — built by ChatGPT, a newer iteration of an earlier `c1` (c1 was a
  strict subset of c2 — same files, minus saved checkpoints and the future
  forecast module — so c1 contributed nothing that c2 didn't already have,
  and isn't part of this merge).

## Why c2 was used as the base

`src/models/common.py`, `architectures.py`, `src/features/features.py`,
`src/data/windows.py`, `src/preprocessing/Datacleaning.py`,
`src/evaluation/leakage_tests.py` and `metrics.py` were **byte-identical**
between the two versions — both were built on the same original pipeline,
so there was nothing to reconcile there.

Where they diverged was in how each solved "predict on any date, including
today/tomorrow" (the feature requested in the earlier session):

- **c4 (Claude)** retrained all 13 models from scratch in a batch script,
  ran real inference on every historical window, and separately pre-baked
  a small "yesterday .. next week" JSON via bootstrap-resampled synthetic
  continuation. No model weights were saved, so every regeneration meant
  retraining, and re-training is stochastic enough that results don't
  reproduce run to run — a real limitation, flagged honestly in that
  session.
- **c2 (ChatGPT)** saves real trained model **artifacts** during the normal
  training stages (`results/models/classical/*.joblib`,
  `results/models/deep/*.pt`), persists a full per-window **prediction
  trace** (`results/predictions/prediction_trace.csv`) as part of the same
  training run rather than a separate stage, and computes future-date
  scenarios **on demand** from the saved weights
  (`src/evaluation/future_forecast.py`) for any weekday from 2025-01-01
  to 2026-12-31 — reproducible, no retraining needed, and a wider date
  range than c4's fixed one-week window.

c2's approach is the more complete and better-engineered solution to the
same problem, so it's the base of this merged project. c4's parallel
`model_zoo.py` / `generate_predictions.py` / `forecast_future.py` and their
output files were **not** carried over — they'd be redundant with what c2
already does, and keeping both would leave two competing, inconsistent
sources of "the real prediction" in one project.

## What was ported over from c4

Two small, non-conflicting improvements, applied directly to c2's
`webapp/app.py`:

1. **`WEBAPP_PORT` / `WEBAPP_HOST` / `WEBAPP_DISPLAY_HOST` env vars** —
   lets you run the site on a different port/host without editing code.
   Defaults unchanged (`localhost:5000`).
2. **A recursive `_sanitize_nans()` safety net** on every endpoint that
   computes data fresh (`/api/market`, `/api/metrics`, `/api/predictions`,
   `/api/future_forecast`). c2's prediction-trace path already survives a
   CSV round-trip that happens to normalize stray `NaN`s to `None` before
   they're serialized (verified — no live bug found there), but
   `future_forecast.py` computes some values directly from live model
   output with no such round-trip, so this closes that path defensively.
   Tested against all 13 models — all return strictly valid JSON.

## Verification performed after merging

- All template JavaScript (`dashboard.html`, `pipeline.html`) syntax-checked
  with Node.
- Flask app boots; `/`, `/pipeline`, `/api/market`, `/api/metrics`,
  `/api/future_dates`, `/api/meta` all return `200` with valid JSON.
- `/api/future_forecast` tested against **all 13 models** — all return
  `200` with strictly valid JSON, zero stray `NaN`/`Infinity` tokens.
- `/api/predictions?day=...&model=...` spot-checked across model tiers —
  valid JSON, no NaN leakage.

## Bug found and fixed after merging

While verifying with a real headless browser (not just curl), switching to
**Future scenario mode** showed a literal `undefined` in the clock readout
(`undefined 2025-01-02`) and threw hundreds of `NaN` SVG attribute errors —
the chart was silently broken.

Root cause: `build_market_json()` (historical days) returns each bar as a
positional **array** `[t, open, high, low, close, volume]`, and the
dashboard's `mapBars()` reads bars positionally (`b[0]`, `b[1]`, ...) to
match. `future_forecast.py`'s `chart` list returned bars as **named-key
objects** (`{"t": ..., "open": ...}`) instead — a data-contract mismatch
between the two data sources feeding the same rendering code. Every field
read off a future-mode bar came back `undefined`, which cascaded into NaN
prices throughout the chart.

Fixed by changing `future_forecast.py`'s chart output to the same
positional-array format as the historical path. Re-verified with a real
Chromium browser (Playwright) across **all 13 models**, the full 60-day
seed dropdown, and multiple future dates (including today) — zero console
errors, zero `NaN`, zero `undefined` after the fix.



Everything else — the candlestick/volume chart, the metric-deviation
z-score heatmap, the model ladder table, the pipeline runner UI, the saved
checkpoints, the prediction-trace persistence logic — is c2 as built,
unchanged.
