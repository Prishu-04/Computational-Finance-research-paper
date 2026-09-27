"""Future-date ML scenario forecasting for the dashboard.

The repository's market file ends in 2024, so 2025/2026 cannot be treated as
observed market data. This module exposes two future-scenario modes:

  forecast_future()          -- single scenario, seeded from ONE chosen
                                 historical day (original behaviour, kept
                                 for backward compatibility with the
                                 dashboard's single-path chart).

  forecast_future_ensemble() -- seeds a scenario from EVERY historical day
                                 in data/raw/raw_data.csv (all 60, by
                                 default), runs the same recursive
                                 generator once per seed day, and reports
                                 the median path plus a 10/90 percentile
                                 band at each decision checkpoint. This is
                                 the "use the complete dataset, not one
                                 arbitrary day" mode.

  build_session_profile()    -- per-5-minute-clock-slot statistics (return,
                                 volume, VIX mean/std) computed across every
                                 historical day. Used by the ensemble mode
                                 and available standalone for anyone who
                                 wants the raw distributional profile.

Both future modes are still just that -- scenarios. A selected future
weekday's first 45 minutes are seeded from history (one day, or all of
them), then each later 5-minute bar is recursively generated using the
selected trained model's own rolling 60-minute forecast. The dashboard/API
clearly labels these paths as scenarios, not realized prices.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.features import add_all_features
from src.models.architectures import (
    RecurrentModel, TransformerModel, PatchTSTModel, CARDModel,
    MoiraiModel, OptimizedMoiraiModel,
)
from src.models.common import FEATURE_COLS, DIR_LABEL_MAP

CONTEXT_BARS = 9
HORIZON_BARS = 12
SESSION_TIMES = pd.date_range("09:15", "15:30", freq="5min").strftime("%H:%M").tolist()
DECISION_TIMES = SESSION_TIMES[CONTEXT_BARS: len(SESSION_TIMES) - HORIZON_BARS]
MODEL_FILES = {
    "Naive": "Naive.joblib",
    "Logistic Regression": "Logistic_Regression.joblib",
    "Random Forest": "Random_Forest.joblib",
    "XGBoost": "XGBoost.joblib",
    "SVR": "SVR.joblib",
    "LSTM": "LSTM.pt",
    "GRU": "GRU.pt",
    "Transformer": "Transformer.pt",
    "PatchTST": "PatchTST.pt",
    "CARD": "CARD.pt",
    "Moirai (reproduced, zero-shot-arch)": "Moirai_reproduced.pt",
    "Moirai + Financial Encoder": "Moirai_Financial_Encoder.pt",
    "Optimized Moirai (+Regime)": "Optimized_Moirai_Regime.pt",
}


def available_future_dates(start="2025-01-01", end="2026-12-31") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _load_raw(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "raw" / "raw_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing historical market data: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("timestamp").reset_index(drop=True)


def _choose_seed_day(raw: pd.DataFrame, seed_day: str | None) -> str:
    days = sorted(raw["date"].unique())
    if seed_day in days:
        return str(seed_day)
    return str(days[-1])


def _future_ts(date: str) -> list[pd.Timestamp]:
    base = pd.Timestamp(date).normalize() + pd.Timedelta(hours=9, minutes=15)
    return [base + pd.Timedelta(minutes=5 * i) for i in range(len(SESSION_TIMES))]


def _load_model(project_root: Path, model_name: str):
    """Load a trained artifact ONCE. Callers that need to run many scenarios
    against the same model (e.g. the ensemble mode below) should call this
    once and reuse the returned (kind, bundle) rather than calling it per
    seed day -- reloading a torch checkpoint 60 times is pure waste."""
    model_dir = project_root / "results" / "models"
    if model_name in {"Naive", "Logistic Regression", "Random Forest", "XGBoost", "SVR"}:
        path = model_dir / "classical" / MODEL_FILES[model_name]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained future artifact: {path}. Re-run the pipeline with the updated files."
            )
        return "classical", joblib.load(path)

    path = model_dir / "deep" / MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained future artifact: {path}. Re-run the pipeline with the updated files."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ctors = {
        "LSTM": lambda: RecurrentModel(cell="lstm"),
        "GRU": lambda: RecurrentModel(cell="gru"),
        "Transformer": lambda: TransformerModel(),
        "PatchTST": lambda: PatchTSTModel(),
        "CARD": lambda: CARDModel(),
        "Moirai (reproduced, zero-shot-arch)": lambda: MoiraiModel(),
        "Moirai + Financial Encoder": lambda: MoiraiModel(use_financial_encoder=True),
        "Optimized Moirai (+Regime)": lambda: OptimizedMoiraiModel(),
    }
    model = ctors[model_name]()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return "deep", {
        "model": model,
        "mean": np.asarray(payload["mean"], dtype=float),
        "std": np.asarray(payload["std"], dtype=float),
    }


def _predict_classical(bundle, x_scaled):
    kind = bundle["kind"]
    if kind == "naive":
        return {"pred_return": 0.0, "pred_direction": "NEUTRAL", "prob_down": 0.0, "prob_neutral": 1.0, "prob_up": 0.0, "confidence": 1.0}

    inv = {v: k for k, v in DIR_LABEL_MAP.items()}
    if kind == "lr":
        probs = bundle["cls"].predict_proba(x_scaled)[0]
        typical = float(bundle["typical_move"])
        pred = float((probs[2] - probs[0]) * typical)
        return {"pred_return": pred, "pred_direction": inv[int(np.argmax(probs))],
                "prob_down": float(probs[0]), "prob_neutral": float(probs[1]), "prob_up": float(probs[2]),
                "confidence": float(probs.max())}
    if kind in {"rf", "xgb"}:
        pred = float(bundle["reg"].predict(x_scaled)[0])
        probs = bundle["cls"].predict_proba(x_scaled)[0]
        return {"pred_return": pred, "pred_direction": inv[int(np.argmax(probs))],
                "prob_down": float(probs[0]), "prob_neutral": float(probs[1]), "prob_up": float(probs[2]),
                "confidence": float(probs.max())}

    pred = float(bundle["reg"].predict(x_scaled)[0])
    tau = float(bundle["tau"])
    direction = "UP" if pred > tau else ("DOWN" if pred < -tau else "NEUTRAL")
    return {"pred_return": pred, "pred_direction": direction,
            "prob_down": None, "prob_neutral": None, "prob_up": None, "confidence": None}


def _predict_deep(bundle, seq):
    model = bundle["model"]
    scaled = (seq - bundle["mean"]) / bundle["std"]
    with torch.no_grad():
        out = model(torch.tensor(scaled[None], dtype=torch.float32))
        probs = torch.softmax(out["cls"], dim=1).numpy()[0]
        inv = {v: k for k, v in DIR_LABEL_MAP.items()}
        result = {
            "pred_return": float(out["reg"].numpy()[0]),
            "pred_direction": inv[int(np.argmax(probs))],
            "prob_down": float(probs[0]), "prob_neutral": float(probs[1]),
            "prob_up": float(probs[2]), "confidence": float(probs.max()),
        }
        if "quantiles" in out:
            q = np.sort(out["quantiles"].numpy()[0])
            result["q10"], result["q50"], result["q90"] = map(float, q[:3])
        return result


def _profile_lookup(seed_raw: pd.DataFrame, clock: str, column: str, fallback: float) -> float:
    s = seed_raw[seed_raw["timestamp"].dt.strftime("%H:%M") == clock]
    if not s.empty and pd.notna(s.iloc[0][column]):
        return float(s.iloc[0][column])
    return fallback


def _build_seed(raw: pd.DataFrame, seed_day: str, future_date: str, target_base: float | None):
    seed_raw = raw[raw["date"] == seed_day].sort_values("timestamp").reset_index(drop=True)
    if len(seed_raw) < CONTEXT_BARS + 1:
        raise ValueError(f"Seed day {seed_day} does not contain enough bars.")
    ts = _future_ts(future_date)
    seed = seed_raw.iloc[:CONTEXT_BARS + 1].copy()
    base = float(target_base) if target_base and target_base > 0 else float(raw.sort_values("timestamp")["close"].iloc[-1])
    scale = base / float(seed["close"].iloc[-1])
    rows = []
    for j, (_, r) in enumerate(seed.iterrows()):
        rows.append({
            "date": future_date, "timestamp": ts[j],
            "open": float(r["open"] * scale), "high": float(r["high"] * scale),
            "low": float(r["low"] * scale), "close": float(r["close"] * scale),
            "volume": float(r["volume"]), "vwap": float(r["vwap"] * scale),
            "india_vix": float(r["india_vix"]),
        })
    return pd.DataFrame(rows), seed_raw, base


def _validate_future_date(future_date: str, model_name: str):
    date_ts = pd.Timestamp(future_date)
    if date_ts < pd.Timestamp("2025-01-01") or date_ts > pd.Timestamp("2026-12-31"):
        raise ValueError("Future date must be between 2025-01-01 and 2026-12-31.")
    if date_ts.weekday() >= 5:
        raise ValueError("Choose a weekday for the future scenario.")
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")


def _forecast_core(raw: pd.DataFrame, kind: str, bundle, future_date: str, seed_day: str,
                    seed_close: float | None, end_time: str = "15:30") -> dict:
    """The actual recursive-scenario generator. Assumes raw + the model are
    already loaded (see forecast_future / forecast_future_ensemble for the
    two ways callers get here) so it can be called many times cheaply."""
    synthetic, seed_raw, base_close = _build_seed(raw, seed_day, future_date, seed_close)
    timestamps = _future_ts(future_date)
    predictions = []

    # The first decision (10:00) sees the seed day's 09:15-09:55 pattern.
    # Later decision bars are generated recursively from the previous model call.
    for i in range(CONTEXT_BARS, len(timestamps) - HORIZON_BARS):
        # Generate the current decision bar if it does not exist yet.
        if len(synthetic) <= i:
            prev_close = float(synthetic["close"].iloc[-1])
            feat_prev = add_all_features(synthetic.copy())
            vol = pd.to_numeric(feat_prev["volatility"], errors="coerce").iloc[-1]
            vol = 0.001 if not np.isfinite(vol) or vol <= 0 else min(float(vol), 0.01)
            # Use prior forecast as the 5-minute recursion driver.
            prev_pred = predictions[-1]["pred_return"] if predictions else 0.0
            step_ret = float(np.clip(prev_pred / HORIZON_BARS, -0.004, 0.004))
            close = prev_close * math.exp(step_ret)
            spread = max(prev_close * vol * 0.65, abs(close - prev_close) * 0.55)
            high = max(prev_close, close) + spread
            low = min(prev_close, close) - spread
            clock = timestamps[i].strftime("%H:%M")
            synthetic.loc[len(synthetic)] = {
                "date": future_date, "timestamp": timestamps[i],
                "open": prev_close, "high": high, "low": low, "close": close,
                "volume": _profile_lookup(seed_raw, clock, "volume", float(synthetic["volume"].iloc[-1])),
                "vwap": (high + low + close) / 3.0,
                "india_vix": _profile_lookup(seed_raw, clock, "india_vix", float(synthetic["india_vix"].iloc[-1])),
            }

        features = add_all_features(synthetic.copy())
        x_seq = features[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).iloc[i - CONTEXT_BARS:i].values.astype(float)
        if kind == "deep":
            pred = _predict_deep(bundle, x_seq)
        else:
            flat = x_seq.reshape(1, -1)
            pred = _predict_classical(bundle, bundle["scaler"].transform(flat)) if "scaler" in bundle else _predict_classical(bundle, flat)

        anchor = float(synthetic["close"].iloc[i])
        target_idx = i + HORIZON_BARS
        target_price = anchor * math.exp(float(pred["pred_return"]))
        predictions.append({
            "decision_ts": timestamps[i].isoformat(),
            "target_ts": timestamps[target_idx].isoformat(),
            "pred_return": float(pred["pred_return"]),
            "pred_direction": pred["pred_direction"],
            "pred_prob_down": pred["prob_down"], "pred_prob_neutral": pred["prob_neutral"],
            "pred_prob_up": pred["prob_up"], "confidence": pred["confidence"],
            "q10": pred.get("q10"), "q50": pred.get("q50", pred["pred_return"]), "q90": pred.get("q90"),
            "anchor_price": anchor, "target_price": float(target_price),
            "downside": bool(pred["pred_return"] < 0),
        })

    # Extend the scenario to the closing bell. These final bars are generated
    # from the last rolling forecast because there is no new observed target
    # after the final 14:30 decision point in the 60-minute setup.
    while len(synthetic) < len(timestamps):
        prev_close = float(synthetic["close"].iloc[-1])
        feat_prev = add_all_features(synthetic.copy())
        vol = pd.to_numeric(feat_prev["volatility"], errors="coerce").iloc[-1]
        vol = 0.001 if not np.isfinite(vol) or vol <= 0 else min(float(vol), 0.01)
        driver = predictions[-1]["pred_return"] if predictions else 0.0
        close = prev_close * math.exp(float(np.clip(driver / HORIZON_BARS, -0.004, 0.004)))
        spread = max(prev_close * vol * 0.65, abs(close - prev_close) * 0.55)
        high, low = max(prev_close, close) + spread, min(prev_close, close) - spread
        clock = timestamps[len(synthetic)].strftime("%H:%M")
        synthetic.loc[len(synthetic)] = {
            "date": future_date, "timestamp": timestamps[len(synthetic)],
            "open": prev_close, "high": high, "low": low, "close": close,
            "volume": _profile_lookup(seed_raw, clock, "volume", float(synthetic["volume"].iloc[-1])),
            "vwap": (high + low + close) / 3.0,
            "india_vix": _profile_lookup(seed_raw, clock, "india_vix", float(synthetic["india_vix"].iloc[-1])),
        }

    end_idx = SESSION_TIMES.index(end_time) if end_time in SESSION_TIMES else len(SESSION_TIMES) - 1
    chart = []
    for _, r in synthetic.iloc[: end_idx + 1].iterrows():
        # Array form [t, open, high, low, close, volume] -- matches the
        # historical bars contract from build_market_json() exactly, since
        # the dashboard's mapBars() does positional b[0]/b[1]/... access on
        # every bar regardless of which mode (historical vs future) it came
        # from. A dict here silently produced `undefined` for every field.
        chart.append([
            r["timestamp"].strftime("%H:%M"),
            round(float(r["open"]), 4), round(float(r["high"]), 4),
            round(float(r["low"]), 4), round(float(r["close"]), 4),
            round(float(r["volume"]), 4),
        ])

    selected = [p for p in predictions if p["decision_ts"][11:16] in DECISION_TIMES]
    return {
        "mode": "future_scenario",
        "future_date": future_date,
        "seed_day": seed_day,
        "base_close": base_close,
        "chart": chart,
        "predictions": selected,
        "note": "ML scenario projection using historical seed context and recursive 60-minute forecasts. Future OHLCV/VIX are unknown until observed.",
    }


def forecast_future(project_root: Path, future_date: str, model_name: str,
                    seed_day: str | None = None, seed_close: float | None = None,
                    end_time: str = "15:30") -> dict:
    """Single scenario, seeded from ONE historical day (default: the most
    recent day in the file). Kept exactly as before -- this is what the
    dashboard's single-path chart calls."""
    _validate_future_date(future_date, model_name)
    raw = _load_raw(project_root)
    seed_day = _choose_seed_day(raw, seed_day)
    kind, bundle = _load_model(project_root, model_name)
    result = _forecast_core(raw, kind, bundle, future_date, seed_day, seed_close, end_time)
    result["model"] = model_name
    return result


# --------------------------------------------------------------------------
# Complete-dataset mode: build a per-clock-slot statistical profile across
# EVERY historical day, and run the scenario generator once per seed day so
# the "future" forecast reflects the dataset's actual distribution instead
# of one arbitrary day's realization.
# --------------------------------------------------------------------------

def build_session_profile(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-5-minute-clock-slot statistics aggregated across ALL historical
    days: mean/std of the log return into that bar, and mean/std of volume
    and india_vix at that bar. One row per clock slot (09:15 ... 15:30)."""
    raw = raw.copy()
    raw["clock"] = raw["timestamp"].dt.strftime("%H:%M")
    raw["log_ret"] = raw.groupby("date")["close"].transform(lambda s: np.log(s / s.shift(1)))
    prof = raw.groupby("clock").agg(
        n_days=("date", "nunique"),
        ret_mean=("log_ret", "mean"), ret_std=("log_ret", "std"),
        vol_mean=("volume", "mean"), vol_std=("volume", "std"),
        vix_mean=("india_vix", "mean"), vix_std=("india_vix", "std"),
    ).reset_index()
    return prof


def forecast_future_ensemble(project_root: Path, future_date: str, model_name: str,
                              seed_days: list[str] | None = None, n_paths: int | None = None,
                              seed_close: float | None = None) -> dict:
    """Run the recursive scenario generator once per historical seed day
    (every day in data/raw/raw_data.csv by default -- pass n_paths to use a
    subset, e.g. the most recent N) and aggregate into a median path plus a
    10/90 percentile band at each decision checkpoint. This is the
    "use the complete dataset, not one arbitrary day" mode: instead of
    trusting one historical morning's pattern, it asks "if I'd seeded this
    future day from EVERY historical morning I have, what's the spread of
    plausible 60-minute-ahead forecasts at each checkpoint?"

    Also returns the individual per-seed-day paths (for anyone who wants to
    plot the full fan of scenarios, not just the summary band) and the
    dataset-wide session profile from build_session_profile() for context.
    """
    _validate_future_date(future_date, model_name)
    raw = _load_raw(project_root)
    kind, bundle = _load_model(project_root, model_name)

    all_days = sorted(raw["date"].unique())
    if seed_days:
        chosen_days = [d for d in seed_days if d in all_days]
    elif n_paths:
        chosen_days = all_days[-n_paths:]
    else:
        chosen_days = all_days

    runs = []
    failures = []
    for seed_day in chosen_days:
        try:
            r = _forecast_core(raw, kind, bundle, future_date, seed_day, seed_close)
            runs.append((seed_day, r))
        except Exception as exc:  # a malformed/short seed day shouldn't kill the whole ensemble
            failures.append({"seed_day": seed_day, "error": str(exc)})

    if not runs:
        raise ValueError("No seed day produced a valid scenario.")

    # Long-format table: one row per (seed_day, decision checkpoint)
    rows = []
    for seed_day, r in runs:
        for p in r["predictions"]:
            rows.append({
                "seed_day": seed_day,
                "clock": p["decision_ts"][11:16],
                "pred_return": p["pred_return"],
                "pred_direction": p["pred_direction"],
                "anchor_price": p["anchor_price"],
                "target_price": p["target_price"],
            })
    long_df = pd.DataFrame(rows)

    def _pct(s, q):
        return float(np.nanpercentile(s, q))

    summary = (
        long_df.groupby("clock")
        .agg(
            n_seeds=("pred_return", "size"),
            ret_p10=("pred_return", lambda s: _pct(s, 10)),
            ret_p50=("pred_return", lambda s: _pct(s, 50)),
            ret_p90=("pred_return", lambda s: _pct(s, 90)),
            pct_up=("pred_direction", lambda s: float((s == "UP").mean())),
            pct_down=("pred_direction", lambda s: float((s == "DOWN").mean())),
            pct_neutral=("pred_direction", lambda s: float((s == "NEUTRAL").mean())),
        )
        .reset_index()
        .sort_values("clock")
    )
    # Majority-vote direction per checkpoint, from the vote shares above.
    def _consensus(row):
        shares = {"UP": row["pct_up"], "DOWN": row["pct_down"], "NEUTRAL": row["pct_neutral"]}
        return max(shares, key=shares.get)
    summary["consensus_direction"] = summary.apply(_consensus, axis=1)

    base_close = runs[0][1]["base_close"]
    checkpoints = [
        {
            "decision_time": row["clock"],
            "n_seeds": int(row["n_seeds"]),
            "return_p10": row["ret_p10"], "return_p50": row["ret_p50"], "return_p90": row["ret_p90"],
            "target_price_p10": base_close * math.exp(row["ret_p10"]),
            "target_price_p50": base_close * math.exp(row["ret_p50"]),
            "target_price_p90": base_close * math.exp(row["ret_p90"]),
            "pct_up": row["pct_up"], "pct_down": row["pct_down"], "pct_neutral": row["pct_neutral"],
            "consensus_direction": row["consensus_direction"],
        }
        for _, row in summary.iterrows()
    ]

    return {
        "mode": "future_scenario_ensemble",
        "future_date": future_date,
        "model": model_name,
        "base_close": base_close,
        "n_seed_days_used": len(runs),
        "n_seed_days_total": len(all_days),
        "seed_days_failed": failures,
        "session_profile": build_session_profile(raw).to_dict(orient="records"),
        "checkpoints": checkpoints,
        "paths": [
            {"seed_day": seed_day, "predictions": r["predictions"]} for seed_day, r in runs
        ],
        "note": (
            "ML scenario projection built from EVERY historical trading day as a "
            "separate seed (not one arbitrary day). p10/p50/p90 are percentiles of "
            "the predicted 60-minute return across all seed-day scenarios at each "
            "decision checkpoint. Still not observed market data."
        ),
    }
