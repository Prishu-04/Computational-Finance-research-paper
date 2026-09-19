"""
Shared training + evaluation harness for Tiers 2-5 (LSTM, GRU, Transformer,
PatchTST, CARD, Moirai, Optimized Moirai). Every model uses this SAME
harness -- same optimizer, same epochs budget, same early stopping, same
loss weighting -- so architecture is the only thing that varies between
runs, per the roadmap's fairness rule.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DIR_LABEL_MAP = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}
DIR_INV_MAP = {v: k for k, v in DIR_LABEL_MAP.items()}
QUANTILES = [0.1, 0.5, 0.9]

FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "vwap", "india_vix",
    "log_return_1", "volatility", "sma_6", "sma_12", "rsi", "macd_hist",
    "minutes_since_open", "session_position",
]
SESSION_POSITION_IDX = FEATURE_COLS.index("session_position")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_standardizer(X_train: np.ndarray):
    """Per-feature mean/std computed from TRAIN ONLY, across samples and
    timesteps. Shape (n_features,)."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


class SeqDataset(Dataset):
    def __init__(self, X, y_return, y_dir_enc):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_return = torch.tensor(y_return, dtype=torch.float32)
        self.y_dir = torch.tensor(y_dir_enc, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_return[idx], self.y_dir[idx]


def make_loaders(X, y_return, y_dir_enc, train_mask, val_mask, test_mask, batch_size=64):
    train_ds = SeqDataset(X[train_mask], y_return[train_mask], y_dir_enc[train_mask])
    val_ds = SeqDataset(X[val_mask], y_return[val_mask], y_dir_enc[val_mask])
    test_ds = SeqDataset(X[test_mask], y_return[test_mask], y_dir_enc[test_mask])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=256, shuffle=False),
        DataLoader(test_ds, batch_size=256, shuffle=False),
    )


def pinball_loss_torch(y_true, y_pred_q, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred_q[:, i]
        losses.append(torch.max(q * err, (q - 1) * err))
    return torch.stack(losses, dim=1).mean()


def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3,
                 probabilistic=False, patience=5, verbose_name=""):
    """
    Generic training loop.
      - Non-probabilistic models: loss = MSE(return) + CrossEntropy(direction)
      - Probabilistic models (Moirai/Optimized Moirai): loss =
        pinball_loss(return quantiles) + CrossEntropy(direction)
    Early stopping on validation loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb_ret, yb_dir in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            if probabilistic:
                reg_loss = pinball_loss_torch(yb_ret, out["quantiles"], QUANTILES)
            else:
                reg_loss = mse(out["reg"], yb_ret)
            cls_loss = ce(out["cls"], yb_dir)
            loss = reg_loss + cls_loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb_ret, yb_dir in val_loader:
                out = model(xb)
                if probabilistic:
                    reg_loss = pinball_loss_torch(yb_ret, out["quantiles"], QUANTILES)
                else:
                    reg_loss = mse(out["reg"], yb_ret)
                cls_loss = ce(out["cls"], yb_dir)
                val_losses.append((reg_loss + cls_loss).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate_model(model, test_loader, train_naive_errors, probabilistic=False):
    from src.evaluation.metrics import (
        regression_metrics, classification_metrics, pinball_loss, interval_coverage,
    )

    model.eval()
    y_true_ret, y_pred_ret, y_true_dir, y_pred_dir = [], [], [], []
    y_pred_q = []
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        for xb, yb_ret, yb_dir in test_loader:
            out = model(xb)
            n += len(xb)
            if probabilistic:
                q = out["quantiles"].numpy()
                y_pred_q.append(q)
                y_pred_ret.append(q[:, QUANTILES.index(0.5)])
            else:
                y_pred_ret.append(out["reg"].numpy())
            y_pred_dir.append(out["cls"].argmax(dim=1).numpy())
            y_true_ret.append(yb_ret.numpy())
            y_true_dir.append(yb_dir.numpy())
    t1 = time.perf_counter()

    y_true_ret = np.concatenate(y_true_ret)
    y_pred_ret = np.concatenate(y_pred_ret)
    y_true_dir = np.concatenate(y_true_dir)
    y_pred_dir = np.concatenate(y_pred_dir)

    reg_m = regression_metrics(y_true_ret, y_pred_ret, train_naive_errors)
    clf_m = classification_metrics(
        [DIR_INV_MAP[v] for v in y_true_dir], [DIR_INV_MAP[v] for v in y_pred_dir]
    )
    result = {**reg_m, **clf_m, "inference_ms": (t1 - t0) / n * 1000}

    if probabilistic:
        y_pred_q = np.concatenate(y_pred_q, axis=0)
        result["CRPS"] = pinball_loss(y_true_ret, y_pred_q, QUANTILES)
        result["coverage_80pct"] = interval_coverage(
            y_true_ret, y_pred_q[:, QUANTILES.index(0.1)], y_pred_q[:, QUANTILES.index(0.9)]
        )
    else:
        result["CRPS"] = np.nan
        result["coverage_80pct"] = np.nan

    return result
