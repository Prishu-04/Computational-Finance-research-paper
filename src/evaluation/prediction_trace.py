from pathlib import Path
import pandas as pd

TRACE_COLUMNS = [
    "date", "decision_ts", "target_ts", "model", "split",
    "true_return", "true_direction", "pred_return", "pred_direction",
    "pred_prob_down", "pred_prob_neutral", "pred_prob_up", "confidence",
    "q10", "q50", "q90",
]


def append_trace(rows, path: str | Path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for col in TRACE_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[TRACE_COLUMNS]
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def replace_trace(rows, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=TRACE_COLUMNS)
    else:
        for col in TRACE_COLUMNS:
            if col not in frame.columns:
                frame[col] = pd.NA
        frame = frame[TRACE_COLUMNS]
    frame.to_csv(path, index=False)
