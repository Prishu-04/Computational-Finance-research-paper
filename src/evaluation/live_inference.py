"""Rolling / live inference for REAL (not recursively-generated) market data.

This is the deploy-ready counterpart to future_forecast.py's scenario
generator. The rule that distinguishes this module from that one:

    ONLY REAL, OBSERVED BARS EVER BECOME MODEL CONTEXT.

No prediction is ever fed back in as a synthetic future bar. A forecast is
made the moment enough real context exists (>= CONTEXT_BARS real bars so
far today), then it sits "pending" until the real bar at its target
timestamp actually arrives, at which point it is resolved against the true
outcome. This is the mode you would actually run once real 2025/26 (or any
live) data starts printing -- future_forecast.py's recursive scenario mode
is for exploring "what might a day look like" *before* real data exists;
this module is for genuinely forecasting the next unseen 60-minute window
from data that is actually happening.

Two ways to use it:

  LiveForecaster   -- a small stateful class: ingest_bar() with each new
                       real 5-min bar as it prints, get back any freshly
                       resolved predictions (with their realized error).
                       This is what you'd wire up to a real market-data feed
                       (or to a websocket handler, a cron job polling a
                       broker API, etc).

  simulate_live()  -- replays data/raw/raw_data.csv bar-by-bar through
                       LiveForecaster as if it were arriving live, purely as
                       a validation harness: it proves the streaming code
                       path never touches a self-generated bar, and gives
                       you an honest accuracy number to compare against the
                       offline walk-forward test-set metrics in
                       results/tables/full_model_ladder_results.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.windows import fit_direction_threshold, to_direction
from src.evaluation.future_forecast import _load_model, _predict_classical, _predict_deep
from src.features.features import add_all_features
from src.models.common import FEATURE_COLS

CONTEXT_BARS = 9
HORIZON_BARS = 12
RAW_COLS = ["date", "timestamp", "open", "high", "low", "close", "volume", "vwap", "india_vix"]


def _load_tau(project_root: Path) -> float:
    """Recompute tau exactly the way the offline pipeline does (Stage 4,
    src/data/windows.py: fit on the TRAIN split only). This keeps the
    realized UP/DOWN/NEUTRAL labels used to score live predictions
    consistent with how every model in the ladder was trained and
    evaluated -- the whole point of Stage 5's leakage gate."""
    proc = Path(project_root) / "data" / "processed"
    y_return = np.load(proc / "y_return.npy")
    train_mask = np.load(proc / "train_mask.npy")
    return fit_direction_threshold(y_return[train_mask])


class LiveForecaster:
    """Stateful rolling forecaster. Feed it real bars one at a time via
    ingest_bar(); it predicts as soon as it has enough real context, and
    resolves each pending prediction against the real bar that later lands
    on its target timestamp."""

    def __init__(self, project_root: Path, model_name: str, max_buffer_bars: int = 200):
        self.project_root = Path(project_root)
        self.model_name = model_name
        self.kind, self.bundle = _load_model(self.project_root, model_name)
        self.tau = _load_tau(self.project_root)
        self.max_buffer_bars = max_buffer_bars
        self.buffer = pd.DataFrame(columns=RAW_COLS)
        self.pending: dict[str, dict] = {}   # target_ts (iso) -> prediction record
        self.resolved: list[dict] = []

    def ingest_bar(self, bar: dict) -> list[dict]:
        """bar: a REAL, observed 5-min OHLCV(+vwap, india_vix) bar, as a
        dict with keys matching RAW_COLS. Returns any predictions resolved
        by this bar (i.e. this bar's timestamp equals a pending
        prediction's target_ts)."""
        row = {c: bar.get(c) for c in RAW_COLS}
        self.buffer.loc[len(self.buffer)] = row
        if len(self.buffer) > self.max_buffer_bars:
            self.buffer = self.buffer.iloc[-self.max_buffer_bars:].reset_index(drop=True)

        ts = pd.Timestamp(row["timestamp"])
        close = float(row["close"])

        newly_resolved = self._resolve_due(ts, close)
        self._maybe_predict(ts)
        return newly_resolved

    def _maybe_predict(self, ts: pd.Timestamp):
        buf_ts = pd.to_datetime(self.buffer["timestamp"])
        today = self.buffer[buf_ts.dt.date == ts.date()].reset_index(drop=True)
        i = len(today) - 1  # index of the bar just ingested, within today
        if i < CONTEXT_BARS:
            return  # not enough real context yet today -- do nothing

        featured = add_all_features(today.assign(timestamp=pd.to_datetime(today["timestamp"])))
        x_seq = (
            featured[FEATURE_COLS]
            .replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
            .iloc[i - CONTEXT_BARS:i].values.astype(float)
        )
        if self.kind == "deep":
            pred = _predict_deep(self.bundle, x_seq)
        else:
            flat = x_seq.reshape(1, -1)
            pred = (_predict_classical(self.bundle, self.bundle["scaler"].transform(flat))
                    if "scaler" in self.bundle else _predict_classical(self.bundle, flat))

        anchor_close = float(today.iloc[i]["close"])
        target_ts = ts + pd.Timedelta(minutes=5 * HORIZON_BARS)
        record = {
            "decision_ts": ts.isoformat(), "target_ts": target_ts.isoformat(),
            "anchor_price": anchor_close,
            "pred_return": pred["pred_return"], "pred_direction": pred["pred_direction"],
            "confidence": pred["confidence"],
        }
        self.pending[target_ts.isoformat()] = record

    def _resolve_due(self, ts: pd.Timestamp, close: float) -> list[dict]:
        key = ts.isoformat()
        if key not in self.pending:
            return []
        rec = self.pending.pop(key)
        realized_return = float(np.log(close / rec["anchor_price"]))
        realized_direction = str(to_direction(np.array([realized_return]), self.tau)[0])
        rec.update({
            "realized_return": realized_return,
            "realized_direction": realized_direction,
            "return_error": realized_return - rec["pred_return"],
            "direction_correct": bool(realized_direction == rec["pred_direction"]),
        })
        self.resolved.append(rec)
        return [rec]

    def trace_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.resolved)

    def accuracy_summary(self) -> dict:
        if not self.resolved:
            return {"n_resolved": 0, "n_pending": len(self.pending)}
        df = self.trace_df()
        return {
            "n_resolved": len(df),
            "n_pending": len(self.pending),
            "direction_accuracy": float(df["direction_correct"].mean()),
            "mae_return": float(df["return_error"].abs().mean()),
        }


def simulate_live(project_root: Path, model_name: str, days: list[str] | None = None) -> dict:
    """Replay data/raw/raw_data.csv bar-by-bar through LiveForecaster, as a
    validation harness. Proves the streaming code path only ever consumes
    real bars (never a self-generated one), and reports the resulting
    accuracy so it can be sanity-checked against
    results/tables/full_model_ladder_results.csv."""
    project_root = Path(project_root)
    raw = pd.read_csv(project_root / "data" / "raw" / "raw_data.csv", parse_dates=["timestamp"])
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    if days:
        raw = raw[raw["date"].isin(days)]
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    forecaster = LiveForecaster(project_root, model_name)
    for _, row in raw.iterrows():
        forecaster.ingest_bar(row.to_dict())

    return {
        "model": model_name,
        "n_bars_replayed": int(len(raw)),
        "n_days_replayed": int(raw["date"].nunique()) if len(raw) else 0,
        "summary": forecaster.accuracy_summary(),
        "trace": forecaster.resolved,
    }
