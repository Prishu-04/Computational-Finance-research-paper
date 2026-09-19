#!/usr/bin/env python3
"""
Stage 8 — Dashboard generation.

Reads the same files Stage 4 (windows), Stage 6 (training) and Stage 7
(evaluate & log) already produce, and rebuilds the Intraday Direction Ladder
dashboard from whatever is currently on disk. Nothing about the dashboard is
hand-edited — rerun this after any change to raw_data.csv, the feature set,
the window config, or the model ladder, and every chart, the day picker and
the model picker all update to match.

Usage (from the project root, after Stage 7 has produced results/tables/):
    python src/visualization/generate_dashboard.py

Optional flags:
    --project-root   defaults to the parent of this script's src/ dir
    --template       defaults to dashboard_template.html next to this script
    --output         defaults to results/dashboard/intraday_direction_ladder.html
    --hero-model     substring used to highlight "this work" in the ladder
                      (default: "Optimized Moirai")

What it expects to find (all produced by earlier stages, per the project's
own flow of control):
    data/processed/featured_data.csv     (Stage 3)
    data/processed/window_meta.csv       (Stage 4)
    data/processed/y_return.npy          (Stage 4)
    data/processed/y_direction.npy       (Stage 4)
    data/processed/train_mask.npy        (Stage 4)
    data/processed/val_mask.npy          (Stage 4)
    data/processed/test_mask.npy         (Stage 4)
    results/tables/full_model_ladder_results.csv   (Stage 7)

If the dataset changes shape (more/fewer days, a different context/horizon,
a different model roster), this script does not need editing — it derives
everything from the files above at run time.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_market_json(data_dir: Path) -> dict:
    """One entry per trading day: the full intraday bar series plus every
    (decision, target) window and its ground-truth label, so the simulator
    can play back any day exactly as Stage 4 defined it."""
    featured_path = data_dir / "featured_data.csv"
    window_meta_path = data_dir / "window_meta.csv"
    for p in (featured_path, window_meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    fd = pd.read_csv(featured_path)
    wm = pd.read_csv(window_meta_path)

    required_fd_cols = {"date", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required_fd_cols - set(fd.columns)
    if missing:
        raise ValueError(f"featured_data.csv is missing columns: {sorted(missing)}")

    y_return = np.load(data_dir / "y_return.npy")
    y_direction = np.load(data_dir / "y_direction.npy", allow_pickle=True)

    n = len(wm)
    if len(y_return) != n or len(y_direction) != n:
        raise ValueError(
            f"window_meta.csv has {n} rows but y_return/y_direction have "
            f"{len(y_return)}/{len(y_direction)}. Re-run Stage 4 fully before "
            f"regenerating the dashboard."
        )

    wm = wm.copy()
    wm["y_return"] = y_return
    wm["y_direction"] = y_direction
    wm["split"] = "train"
    for mask_name, split_name in (("val_mask.npy", "val"), ("test_mask.npy", "test")):
        mask_path = data_dir / mask_name
        if mask_path.exists():
            mask = np.load(mask_path)
            if len(mask) != n:
                raise ValueError(f"{mask_name} length {len(mask)} != window count {n}")
            wm.loc[mask, "split"] = split_name

    out_days = {}
    for day, day_bars in fd.groupby("date"):
        bars = day_bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        bars["t"] = bars["timestamp"].astype(str).str.slice(11, 16)
        bars_list = bars[["t", "open", "high", "low", "close", "volume"]].round(4).values.tolist()

        wins = wm[wm["date"] == day].copy()
        if wins.empty:
            # a day with bars but no complete windows (e.g. too near session
            # close for the horizon) still gets a chart, just no markers
            out_days[str(day)] = {"split": "train", "bars": bars_list, "windows": []}
            continue
        wins["dt"] = wins["decision_ts"].astype(str).str.slice(11, 16)
        wins["tt"] = wins["target_ts"].astype(str).str.slice(11, 16)
        wins_list = wins[["dt", "tt", "y_direction", "y_return", "split"]].values.tolist()
        out_days[str(day)] = {
            "split": wins["split"].iloc[0],
            "bars": bars_list,
            "windows": wins_list,
        }

    return out_days


# Which tier each model belongs to, used only to assemble
# full_model_ladder_results.csv if that merged file doesn't exist yet
# (run_classical_baselines.py and run_deep_learning_tiers.py each write
# their own tier table; nothing in the pipeline merges them automatically).
TIER_MAP = {
    "Naive": "Tier 1: Classical ML",
    "Logistic Regression": "Tier 1: Classical ML",
    "Random Forest": "Tier 1: Classical ML",
    "XGBoost": "Tier 1: Classical ML",
    "SVR": "Tier 1: Classical ML",
    "LSTM": "Tier 2: Deep Learning",
    "GRU": "Tier 2: Deep Learning",
    "Transformer": "Tier 3: Transformer family",
    "PatchTST": "Tier 3: Transformer family",
    "CARD": "Tier 3: Transformer family",
    "Moirai (reproduced, zero-shot-arch)": "Tier 4-5: Moirai family",
    "Moirai + Financial Encoder": "Tier 4-5: Moirai family",
    "Optimized Moirai (+Regime)": "Tier 4-5: Moirai family",
}


def ensure_full_ladder_table(results_dir: Path) -> Path:
    """full_model_ladder_results.csv isn't written by any stage script —
    it's the union of tier1_classical_ml_results.csv and
    tier2_5_deep_learning_results.csv. Build it if it's missing or stale
    relative to those two, so a fresh pipeline run doesn't require a manual
    merge step before the dashboard can be generated."""
    tables_dir = results_dir / "tables"
    full_path = tables_dir / "full_model_ladder_results.csv"
    tier1_path = tables_dir / "tier1_classical_ml_results.csv"
    tier25_path = tables_dir / "tier2_5_deep_learning_results.csv"

    have_tier1 = tier1_path.exists()
    have_tier25 = tier25_path.exists()
    if not have_tier1 and not have_tier25:
        return full_path  # nothing to build from; let the caller raise a clear error

    newest_source_mtime = max(
        (p.stat().st_mtime for p in (tier1_path, tier25_path) if p.exists()), default=0
    )
    if full_path.exists() and full_path.stat().st_mtime >= newest_source_mtime:
        return full_path  # already up to date

    frames = []
    if have_tier1:
        frames.append(pd.read_csv(tier1_path))
    if have_tier25:
        frames.append(pd.read_csv(tier25_path))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("CRPS", "coverage_80pct"):
        if col not in merged.columns:
            merged[col] = np.nan

    unknown = sorted(set(merged["model"]) - set(TIER_MAP))
    if unknown:
        print(
            f"[dashboard] WARNING: no tier mapping for {unknown} — add them to "
            f"TIER_MAP in generate_dashboard.py. Tagging as 'Unclassified' for now."
        )
    merged["tier"] = merged["model"].map(TIER_MAP).fillna("Unclassified")

    ordered_cols = [
        "tier", "model", "MAE", "RMSE", "MASE", "direction_accuracy",
        "F1_macro", "CRPS", "coverage_80pct", "inference_ms",
    ]
    merged = merged[ordered_cols]
    tables_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(full_path, index=False)
    print(f"[dashboard] rebuilt {full_path} from tier1 + tier2-5 result tables")
    return full_path


def build_metrics_json(results_dir: Path) -> list:
    """Every row of the full model ladder, tier-tagged, NaN-safe."""
    path = ensure_full_ladder_table(results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path} (and no tier1/tier2-5 result "
            f"tables to build it from — run Stage 6 first)"
        )

    df = pd.read_csv(path)
    required = {"tier", "model", "direction_accuracy", "F1_macro", "inference_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"full_model_ladder_results.csv is missing columns: {sorted(missing)}")

    records = df.to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, float):
                r[k] = None if math.isnan(v) else round(v, 6)
    return records


def render(template_path: Path, market: dict, metrics: list, output_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    if "__MARKET_DATA__" not in template or "__MODELS_METRICS__" not in template:
        raise ValueError(
            "Template is missing the __MARKET_DATA__ / __MODELS_METRICS__ "
            "placeholders — did dashboard_template.html get edited by hand?"
        )
    html = template.replace("__MARKET_DATA__", json.dumps(market)).replace(
        "__MODELS_METRICS__", json.dumps(metrics)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _guess_project_root(script_dir: Path) -> Path:
    """Walk upward from cwd, then from the script's own location, looking for
    a directory that has both data/ and results/ — works whether this script
    lives in src/visualization/ or is run from anywhere else in the repo."""
    for start in (Path.cwd(), script_dir):
        for candidate in (start, *start.parents):
            if (candidate / "data").is_dir() and (candidate / "results").is_dir():
                return candidate
    return Path.cwd()


def main():
    script_dir = Path(__file__).resolve().parent
    default_root = _guess_project_root(script_dir)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--template", type=Path, default=script_dir / "dashboard_template.html")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hero-model", type=str, default="Optimized Moirai")
    args = parser.parse_args()

    root = args.project_root
    data_dir = root / "data" / "processed"
    results_dir = root / "results"
    output_path = args.output or (results_dir / "dashboard" / "intraday_direction_ladder.html")

    print(f"[dashboard] project root : {root}")
    print(f"[dashboard] reading      : {data_dir}")
    print(f"[dashboard] reading      : {results_dir / 'tables'}")

    market = build_market_json(data_dir)
    metrics = build_metrics_json(results_dir)

    n_days = len(market)
    n_models = len(metrics)
    print(f"[dashboard] {n_days} trading day(s), {n_models} model row(s) in the ladder")

    render(args.template, market, metrics, output_path)
    print(f"[dashboard] wrote        : {output_path}")


if __name__ == "__main__":
    main()
