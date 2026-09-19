"""
Shared metrics -- used identically for every model tier so comparisons are
fair, per the roadmap's "same dataset, same evaluation, only the
architecture changes" rule.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score


def mase(y_true, y_pred, y_train_naive_errors):
    naive_mae = np.mean(np.abs(y_train_naive_errors))
    if naive_mae == 0:
        return np.nan
    return mean_absolute_error(y_true, y_pred) / naive_mae


def regression_metrics(y_true, y_pred, train_naive_errors):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MASE": mase(y_true, y_pred, train_naive_errors),
    }


def classification_metrics(y_true, y_pred):
    return {
        "direction_accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def pinball_loss(y_true, y_pred_q, quantiles):
    """Average pinball (quantile) loss across a set of predicted quantiles.
    Used as a CRPS approximation for probabilistic models (Moirai/Optimized Moirai)."""
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred_q[:, i]
        losses.append(np.maximum(q * err, (q - 1) * err))
    return float(np.mean(losses))


def interval_coverage(y_true, y_lo, y_hi):
    """Fraction of true values falling inside the predicted [y_lo, y_hi] interval."""
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))
