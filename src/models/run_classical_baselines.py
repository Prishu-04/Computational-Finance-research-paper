import sys, time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier
import joblib
from pathlib import Path

from src.evaluation.metrics import regression_metrics, classification_metrics
from src.evaluation.prediction_trace import replace_trace

SEED = 42
CONTEXT_BARS, HORIZON_BARS = 9, 12


def load_data():
    X = np.load("data/processed/X.npy")
    y_return = np.load("data/processed/y_return.npy")
    y_direction = np.load("data/processed/y_direction.npy")
    train_mask = np.load("data/processed/train_mask.npy")
    val_mask = np.load("data/processed/val_mask.npy")
    test_mask = np.load("data/processed/test_mask.npy")
    n = X.shape[0]
    X_flat = X.reshape(n, -1)
    n_nan = np.isnan(X_flat).sum()
    if n_nan > 0:
        X_flat = np.nan_to_num(X_flat, nan=0.0)
    return X_flat, y_return, y_direction, train_mask, val_mask, test_mask


def prep_features(X_flat, train_mask, val_mask, test_mask):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_flat[train_mask])
    X_val = scaler.transform(X_flat[val_mask])
    X_test = scaler.transform(X_flat[test_mask])
    return X_train, X_val, X_test


def measure_latency_ms(predict_fn, X_test):
    t0 = time.perf_counter(); predict_fn(X_test); t1 = time.perf_counter()
    return (t1 - t0) / len(X_test) * 1000


def _trace_rows(meta, split, model_name, y_ret, y_dir, pred_ret, pred_dir, probs=None):
    rows = []
    for i in range(len(meta)):
        pr = probs[i] if probs is not None else None
        rows.append({
            "date": meta.iloc[i]["date"],
            "decision_ts": meta.iloc[i]["decision_ts"],
            "target_ts": meta.iloc[i]["target_ts"],
            "model": model_name,
            "split": split[i],
            "true_return": float(y_ret[i]),
            "true_direction": str(y_dir[i]),
            "pred_return": float(pred_ret[i]),
            "pred_direction": str(pred_dir[i]),
            "pred_prob_down": float(pr[0]) if pr is not None else np.nan,
            "pred_prob_neutral": float(pr[1]) if pr is not None else np.nan,
            "pred_prob_up": float(pr[2]) if pr is not None else np.nan,
            "confidence": float(np.max(pr)) if pr is not None else np.nan,
            "q10": np.nan, "q50": float(pred_ret[i]), "q90": np.nan,
        })
    return rows


def run():
    X_flat, y_return, y_direction, train_mask, val_mask, test_mask = load_data()
    X_train, X_val, X_test = prep_features(X_flat, train_mask, val_mask, test_mask)
    artifact_scaler = StandardScaler().fit(X_flat[train_mask])
    y_ret_train, y_ret_test = y_return[train_mask], y_return[test_mask]
    y_dir_train, y_dir_test = y_direction[train_mask], y_direction[test_mask]
    train_naive_errors = y_ret_train - 0.0
    typical_move = float(np.median(np.abs(y_ret_train)))
    tau = fit_tau_from_training("data/processed/y_return.npy", train_mask)
    model_dir = Path("results/models/classical")
    model_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv("data/processed/window_meta.csv")
    split_arr = np.where(train_mask, "train", np.where(val_mask, "val", "test"))
    trace_rows = []
    results = []

    # A single global input matrix keeps the trace aligned 1:1 with window_meta.
    X_all = X_flat

    naive_ret = np.zeros_like(y_return)
    naive_dir = np.full_like(y_direction, "NEUTRAL")
    results.append({"model": "Naive", **regression_metrics(y_ret_test, naive_ret[test_mask], train_naive_errors),
                    **classification_metrics(y_dir_test, naive_dir[test_mask]), "inference_ms": 0.0})
    trace_rows += _trace_rows(meta, split_arr, "Naive", y_return, y_direction, naive_ret, naive_dir,
                              probs=np.tile(np.array([0.0, 1.0, 0.0]), (len(meta), 1)))

    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(X_train, y_dir_train)
    joblib.dump({"kind":"lr", "scaler":artifact_scaler, "cls":lr, "typical_move":typical_move}, model_dir / "Logistic_Regression.joblib")
    lr_pred_all = lr.predict(X_all)
    lr_prob_all = lr.predict_proba(X_all)
    lat = measure_latency_ms(lr.predict, X_test)
    results.append({"model": "Logistic Regression", "MAE": np.nan, "RMSE": np.nan, "MASE": np.nan,
                    **classification_metrics(y_dir_test, lr.predict(X_test)), "inference_ms": lat})
    trace_rows += _trace_rows(meta, split_arr, "Logistic Regression", y_return, y_direction,
                              np.full_like(y_return, np.nan, dtype=float), lr_pred_all, lr_prob_all)

    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_reg.fit(X_train, y_ret_train)
    rf_reg_all = rf_reg.predict(X_all)
    lat = measure_latency_ms(rf_reg.predict, X_test)
    reg_m = regression_metrics(y_ret_test, rf_reg_all[test_mask], train_naive_errors)
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_clf.fit(X_train, y_dir_train)
    joblib.dump({"kind":"rf", "scaler":artifact_scaler, "reg":rf_reg, "cls":rf_clf}, model_dir / "Random_Forest.joblib")
    rf_dir_all = rf_clf.predict(X_all)
    rf_prob_all = rf_clf.predict_proba(X_all)
    results.append({"model": "Random Forest", **reg_m,
                    **classification_metrics(y_dir_test, rf_clf.predict(X_test)), "inference_ms": lat})
    trace_rows += _trace_rows(meta, split_arr, "Random Forest", y_return, y_direction, rf_reg_all, rf_dir_all, rf_prob_all)

    label_map = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}; inv_map = {v: k for k, v in label_map.items()}
    y_dir_train_enc = np.array([label_map[v] for v in y_dir_train])
    xgb_reg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED, n_jobs=-1)
    xgb_reg.fit(X_train, y_ret_train)
    xgb_reg_all = xgb_reg.predict(X_all)
    lat = measure_latency_ms(xgb_reg.predict, X_test)
    reg_m = regression_metrics(y_ret_test, xgb_reg_all[test_mask], train_naive_errors)
    xgb_clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED,
                             n_jobs=-1, eval_metric="mlogloss")
    xgb_clf.fit(X_train, y_dir_train_enc)
    joblib.dump({"kind":"xgb", "scaler":artifact_scaler, "reg":xgb_reg, "cls":xgb_clf}, model_dir / "XGBoost.joblib")
    xgb_idx_all = xgb_clf.predict(X_all).astype(int)
    xgb_dir_all = np.array([inv_map[v] for v in xgb_idx_all])
    xgb_prob_all = xgb_clf.predict_proba(X_all)
    pred_dir = np.array([inv_map[v] for v in xgb_clf.predict(X_test)])
    results.append({"model": "XGBoost", **reg_m,
                    **classification_metrics(y_dir_test, pred_dir), "inference_ms": lat})
    trace_rows += _trace_rows(meta, split_arr, "XGBoost", y_return, y_direction, xgb_reg_all, xgb_dir_all, xgb_prob_all)

    svr = SVR(kernel="rbf", C=1.0, epsilon=0.0005)
    svr.fit(X_train, y_ret_train)
    svr_ret_all = svr.predict(X_all)
    lat = measure_latency_ms(svr.predict, X_test)
    reg_m = regression_metrics(y_ret_test, svr_ret_all[test_mask], train_naive_errors)
    # Keep SVR regression-only in the ladder; derive a directional display label
    # only for the timestamp trace so the forecast trajectory is visible.
    tau = fit_tau_from_training("data/processed/y_return.npy", train_mask)
    svr_dir_all = to_dir_from_return(svr_ret_all, tau)
    joblib.dump({"kind":"svr", "scaler":artifact_scaler, "reg":svr, "tau":tau}, model_dir / "SVR.joblib")
    joblib.dump({"kind":"naive", "typical_move":typical_move, "tau":tau}, model_dir / "Naive.joblib")
    results.append({"model": "SVR", **reg_m, "direction_accuracy": np.nan, "F1_macro": np.nan,
                    "inference_ms": lat})
    trace_rows += _trace_rows(meta, split_arr, "SVR", y_return, y_direction, svr_ret_all, svr_dir_all)

    replace_trace(trace_rows, "results/predictions/classical_prediction_trace.csv")
    return pd.DataFrame(results)[["model", "MAE", "RMSE", "MASE", "direction_accuracy", "F1_macro", "inference_ms"]]


def fit_tau_from_training(path, train_mask):
    y = np.load(path)
    return float(np.quantile(np.abs(y[train_mask]), 1 - 2 * 0.33))


def to_dir_from_return(values, tau):
    return np.where(values > tau, "UP", np.where(values < -tau, "DOWN", "NEUTRAL"))

def log_experiments(df):
    log_path = "experiments/experiment_log.csv"
    log = pd.read_csv(log_path)
    next_id = len(log) + 1
    rows = []
    for _, r in df.iterrows():
        rows.append({"experiment_id": f"EXP-{next_id:03d}", "model": r["model"],
                      "dataset": "SYNTHETIC_NIFTY_5min", "frequency": "5min",
                      "context_min": CONTEXT_BARS * 5, "horizon_min": HORIZON_BARS * 5,
                      "features": "OHLCV+VIX+technical+time", "target": "return+direction", "seed": SEED,
                      "MAE": r["MAE"], "RMSE": r["RMSE"], "MASE": r["MASE"],
                      "direction_accuracy": r["direction_accuracy"], "F1": r["F1_macro"],
                      "CRPS": np.nan, "inference_ms": r["inference_ms"],
                      "notes": "Tier 1 classical ML baseline"})
        next_id += 1
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    log.to_csv(log_path, index=False)


if __name__ == "__main__":
    df = run()
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    print(df.to_string(index=False))
    df.to_csv("results/tables/tier1_classical_ml_results.csv", index=False)
    log_experiments(df)
