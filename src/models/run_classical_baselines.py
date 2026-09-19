import sys, time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier

from src.evaluation.metrics import regression_metrics, classification_metrics

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


def run():
    X_flat, y_return, y_direction, train_mask, val_mask, test_mask = load_data()
    X_train, X_val, X_test = prep_features(X_flat, train_mask, val_mask, test_mask)
    y_ret_train, y_ret_test = y_return[train_mask], y_return[test_mask]
    y_dir_train, y_dir_test = y_direction[train_mask], y_direction[test_mask]
    train_naive_errors = y_ret_train - 0.0
    results = []

    naive_ret = np.zeros_like(y_ret_test)
    naive_dir = np.full_like(y_dir_test, "NEUTRAL")
    results.append({"model": "Naive", **regression_metrics(y_ret_test, naive_ret, train_naive_errors),
                     **classification_metrics(y_dir_test, naive_dir), "inference_ms": 0.0})

    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(X_train, y_dir_train)
    lat = measure_latency_ms(lr.predict, X_test)
    results.append({"model": "Logistic Regression", "MAE": np.nan, "RMSE": np.nan, "MASE": np.nan,
                     **classification_metrics(y_dir_test, lr.predict(X_test)), "inference_ms": lat})

    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_reg.fit(X_train, y_ret_train)
    lat = measure_latency_ms(rf_reg.predict, X_test)
    reg_m = regression_metrics(y_ret_test, rf_reg.predict(X_test), train_naive_errors)
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
    rf_clf.fit(X_train, y_dir_train)
    results.append({"model": "Random Forest", **reg_m,
                     **classification_metrics(y_dir_test, rf_clf.predict(X_test)), "inference_ms": lat})

    label_map = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}; inv_map = {v: k for k, v in label_map.items()}
    y_dir_train_enc = np.array([label_map[v] for v in y_dir_train])
    xgb_reg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED, n_jobs=-1)
    xgb_reg.fit(X_train, y_ret_train)
    lat = measure_latency_ms(xgb_reg.predict, X_test)
    reg_m = regression_metrics(y_ret_test, xgb_reg.predict(X_test), train_naive_errors)
    xgb_clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED,
                             n_jobs=-1, eval_metric="mlogloss")
    xgb_clf.fit(X_train, y_dir_train_enc)
    pred_dir = np.array([inv_map[v] for v in xgb_clf.predict(X_test)])
    results.append({"model": "XGBoost", **reg_m,
                     **classification_metrics(y_dir_test, pred_dir), "inference_ms": lat})

    svr = SVR(kernel="rbf", C=1.0, epsilon=0.0005)
    svr.fit(X_train, y_ret_train)
    lat = measure_latency_ms(svr.predict, X_test)
    reg_m = regression_metrics(y_ret_test, svr.predict(X_test), train_naive_errors)
    results.append({"model": "SVR", **reg_m, "direction_accuracy": np.nan, "F1_macro": np.nan,
                     "inference_ms": lat})

    return pd.DataFrame(results)[["model", "MAE", "RMSE", "MASE", "direction_accuracy", "F1_macro", "inference_ms"]]


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
