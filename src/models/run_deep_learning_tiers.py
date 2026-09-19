"""
Trains and evaluates Tiers 2-5:
  Tier 2: LSTM, GRU
  Tier 3: Transformer, PatchTST, CARD
  Tier 4: Moirai (reproduced architecture, trained from scratch)
  Tier 5: Optimized Moirai (+ ablation: Moirai+FinancialEncoder only)

Same data, same splits, same loss weighting, same epoch/patience budget
for every model.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.models.common import (
    set_seed, fit_standardizer, apply_standardizer, make_loaders,
    train_model, evaluate_model, DIR_LABEL_MAP,
)
from src.models.architectures import (
    RecurrentModel, TransformerModel, PatchTSTModel, CARDModel,
    MoiraiModel, OptimizedMoiraiModel,
)

SEED = 42
CONTEXT_BARS, HORIZON_BARS = 9, 12
EPOCHS = 25
PATIENCE = 5


def load_data():
    X = np.load("data/processed/X.npy")
    y_return = np.load("data/processed/y_return.npy")
    y_direction = np.load("data/processed/y_direction.npy")
    train_mask = np.load("data/processed/train_mask.npy")
    val_mask = np.load("data/processed/val_mask.npy")
    test_mask = np.load("data/processed/test_mask.npy")

    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        X = np.nan_to_num(X, nan=0.0)
        print(f"[data] imputed {n_nan} NaN values (day-start warm-up bars) with 0")

    y_dir_enc = np.array([DIR_LABEL_MAP[v] for v in y_direction])
    return X, y_return, y_dir_enc, train_mask, val_mask, test_mask


def run():
    set_seed(SEED)
    X, y_return, y_dir_enc, train_mask, val_mask, test_mask = load_data()

    mean, std = fit_standardizer(X[train_mask])
    X = apply_standardizer(X, mean, std)

    train_loader, val_loader, test_loader = make_loaders(
        X, y_return, y_dir_enc, train_mask, val_mask, test_mask
    )
    train_naive_errors = y_return[train_mask] - 0.0

    model_specs = [
        ("LSTM", lambda: RecurrentModel(cell="lstm"), False),
        ("GRU", lambda: RecurrentModel(cell="gru"), False),
        ("Transformer", lambda: TransformerModel(), False),
        ("PatchTST", lambda: PatchTSTModel(), False),
        ("CARD", lambda: CARDModel(), False),
        ("Moirai (reproduced, zero-shot-arch)", lambda: MoiraiModel(), True),
        ("Moirai + Financial Encoder", lambda: MoiraiModel(use_financial_encoder=True), True),
        ("Optimized Moirai (+Regime)", lambda: OptimizedMoiraiModel(), True),
    ]

    results = []
    for name, ctor, probabilistic in model_specs:
        set_seed(SEED)
        print(f"\n=== Training {name} ===")
        model = ctor()
        model, best_val = train_model(
            model, train_loader, val_loader, epochs=EPOCHS,
            probabilistic=probabilistic, patience=PATIENCE,
        )
        metrics = evaluate_model(model, test_loader, train_naive_errors, probabilistic=probabilistic)
        metrics["model"] = name
        metrics["best_val_loss"] = best_val
        results.append(metrics)
        print(f"  best_val_loss={best_val:.5f}  "
              f"MAE={metrics['MAE']:.5f}  dir_acc={metrics['direction_accuracy']:.3f}  "
              f"F1={metrics['F1_macro']:.3f}" +
              (f"  CRPS={metrics['CRPS']:.5f}  coverage80={metrics['coverage_80pct']:.3f}"
               if probabilistic else ""))

    df = pd.DataFrame(results)[
        ["model", "MAE", "RMSE", "MASE", "direction_accuracy", "F1_macro",
         "CRPS", "coverage_80pct", "inference_ms"]
    ]
    return df


def log_experiments(df: pd.DataFrame):
    log_path = "experiments/experiment_log.csv"
    log = pd.read_csv(log_path)
    next_id = len(log) + 1
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "experiment_id": f"EXP-{next_id:03d}", "model": r["model"],
            "dataset": "SYNTHETIC_NIFTY_5min", "frequency": "5min",
            "context_min": CONTEXT_BARS * 5, "horizon_min": HORIZON_BARS * 5,
            "features": "OHLCV+VIX+technical+time", "target": "return+direction",
            "seed": SEED, "MAE": r["MAE"], "RMSE": r["RMSE"], "MASE": r["MASE"],
            "direction_accuracy": r["direction_accuracy"], "F1": r["F1_macro"],
            "CRPS": r["CRPS"], "inference_ms": r["inference_ms"],
            "notes": "Tier 2-5 deep learning (harness: src/models/common.py)",
        })
        next_id += 1
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    log.to_csv(log_path, index=False)


if __name__ == "__main__":
    df = run()
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    print("\n\n=== Tier 2-5 Results ===")
    print(df.to_string(index=False))
    df.to_csv("results/tables/tier2_5_deep_learning_results.csv", index=False)
    log_experiments(df)
    print("\nSaved: results/tables/tier2_5_deep_learning_results.csv")
    print("Logged to: experiments/experiment_log.csv")
