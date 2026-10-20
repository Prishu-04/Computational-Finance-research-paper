import sys
sys.path.insert(0, ".")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.common import (
    set_seed, fit_standardizer, apply_standardizer, make_loaders,
    train_model, evaluate_model, DIR_LABEL_MAP,
)
from src.evaluation.prediction_trace import replace_trace
from src.models.architectures import (
    RecurrentModel, TransformerModel, PatchTSTModel, CARDModel,
    MoiraiModel, OptimizedMoiraiModel,
)

SEED = 42
try:
    torch.set_num_threads(4)
except Exception:
    pass
CONTEXT_BARS, HORIZON_BARS = 9, 12
EPOCHS = int(os.getenv("CF_EPOCHS", "25"))
PATIENCE = int(os.getenv("CF_PATIENCE", "5"))


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


def trace_rows_for_model(meta, split_arr, model_name, out, y_return, y_direction, probabilistic):
    logits = out["cls"].detach().cpu().numpy()
    probs = torch.softmax(out["cls"], dim=1).detach().cpu().numpy()
    pred_idx = logits.argmax(axis=1)
    inv = {v: k for k, v in DIR_LABEL_MAP.items()}
    pred_dir = np.array([inv[int(v)] for v in pred_idx])
    pred_ret = out["reg"].detach().cpu().numpy()
    if probabilistic:
        q = out["quantiles"].detach().cpu().numpy()
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
    else:
        q10 = q90 = np.full(len(pred_ret), np.nan)
        q50 = pred_ret
    rows = []
    for i in range(len(meta)):
        rows.append({
            "date": meta.iloc[i]["date"],
            "decision_ts": meta.iloc[i]["decision_ts"],
            "target_ts": meta.iloc[i]["target_ts"],
            "model": model_name,
            "split": split_arr[i],
            "true_return": float(y_return[i]),
            "true_direction": str(y_direction[i]),
            "pred_return": float(pred_ret[i]),
            "pred_direction": str(pred_dir[i]),
            "pred_prob_down": float(probs[i, 0]),
            "pred_prob_neutral": float(probs[i, 1]),
            "pred_prob_up": float(probs[i, 2]),
            "confidence": float(probs[i].max()),
            "q10": float(q10[i]), "q50": float(q50[i]), "q90": float(q90[i]),
        })
    return rows


def predict_full(model, X, batch_size=256):
    model.eval()
    chunks = {"reg": [], "cls": [], "quantiles": []}
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
            out = model(xb)
            for key in chunks:
                if key in out:
                    chunks[key].append(out[key].cpu())
    return {k: torch.cat(v, dim=0) for k, v in chunks.items() if v}


def run():
    set_seed(SEED)
    X, y_return, y_dir_enc, train_mask, val_mask, test_mask = load_data()
    y_direction = np.load("data/processed/y_direction.npy")
    meta = pd.read_csv("data/processed/window_meta.csv")
    split_arr = np.where(train_mask, "train", np.where(val_mask, "val", "test"))

    mean, std = fit_standardizer(X[train_mask])
    X = apply_standardizer(X, mean, std)

    train_loader, val_loader, test_loader = make_loaders(
        X, y_return, y_dir_enc, train_mask, val_mask, test_mask, batch_size=256
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
    trace_rows = []
    model_dir = Path("results/models/deep")
    model_dir.mkdir(parents=True, exist_ok=True)
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
        full_out = predict_full(model, X)
        artifact_map = {
            "LSTM": "LSTM.pt", "GRU": "GRU.pt", "Transformer": "Transformer.pt",
            "PatchTST": "PatchTST.pt", "CARD": "CARD.pt",
            "Moirai (reproduced, zero-shot-arch)": "Moirai_reproduced.pt",
            "Moirai + Financial Encoder": "Moirai_Financial_Encoder.pt",
            "Optimized Moirai (+Regime)": "Optimized_Moirai_Regime.pt",
        }
        artifact_name = artifact_map[name]
        torch.save({"state_dict": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(), "model": name}, model_dir / artifact_name)
        trace_rows.extend(trace_rows_for_model(meta, split_arr, name, full_out, y_return, y_direction, probabilistic))
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
    classical_path = "results/predictions/classical_prediction_trace.csv"
    if Path(classical_path).exists():
        classical = pd.read_csv(classical_path).to_dict(orient="records")
        trace_rows = classical + trace_rows
    replace_trace(trace_rows, "results/predictions/prediction_trace.csv")
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
