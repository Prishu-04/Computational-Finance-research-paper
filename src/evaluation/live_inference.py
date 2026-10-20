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
    proc = Path(project_root) / "data" / "processed"
    y_return = np.load(proc / "y_return.npy")
    train_mask = np.load(proc / "train_mask.npy")
    return fit_direction_threshold(y_return[train_mask])


class LiveForecaster:
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
