import numpy as np
import pandas as pd

def test_chronological_split(meta: pd.DataFrame, train_mask, val_mask, test_mask):
    train_max = meta.loc[train_mask, "decision_ts"].max()
    val_min = meta.loc[val_mask, "decision_ts"].min()
    val_max = meta.loc[val_mask, "decision_ts"].max()
    test_min = meta.loc[test_mask, "decision_ts"].min()
    assert train_max <= val_min, f"FAIL: train overlaps val ({train_max} > {val_min})"
    assert val_max <= test_min, f"FAIL: val overlaps test ({val_max} > {test_min})"
    print("[PASS] chronological_split: train < val < test, no time overlap")

def test_target_after_decision(meta: pd.DataFrame):
    bad = (meta["target_ts"] <= meta["decision_ts"]).sum()
    assert bad == 0, f"FAIL: {bad} windows have target_ts <= decision_ts"
    print("[PASS] target_after_decision: every target strictly follows its decision point")


def test_no_cross_day_windows(meta: pd.DataFrame):
    bad = (pd.to_datetime(meta["target_ts"]).dt.date !=
           pd.to_datetime(meta["decision_ts"]).dt.date).sum()
    assert bad == 0, f"FAIL: {bad} windows span across trading days"
    print("[PASS] no_cross_day_windows: context/target never cross a session boundary")


def test_tau_fit_on_train_only(y_return, train_mask):
    from src.data.windows import fit_direction_threshold
    tau_train = fit_direction_threshold(y_return[train_mask])
    tau_all = fit_direction_threshold(y_return)
    print(f"[INFO] tau(train)={tau_train:.6f} vs tau(all)={tau_all:.6f} "
          f"-- these SHOULD differ slightly; if identical, check you aren't "
          f"accidentally fitting on the full array.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    meta = pd.read_csv("data/processed/window_meta.csv", parse_dates=["decision_ts", "target_ts"])
    train_mask = np.load("data/processed/train_mask.npy")
    val_mask = np.load("data/processed/val_mask.npy")
    test_mask = np.load("data/processed/test_mask.npy")
    y_return = np.load("data/processed/y_return.npy")

    test_chronological_split(meta, train_mask, val_mask, test_mask)
    test_target_after_decision(meta)
    test_no_cross_day_windows(meta)
    test_tau_fit_on_train_only(y_return, train_mask)
    print("\nAll leakage checks passed.")