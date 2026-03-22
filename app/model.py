"""
model.py - LSTM-based price prediction using PyTorch.

Key improvements over v1 (single-feature, price-level prediction):

1. Multi-feature input (6 signals):
       log_return, RSI/100, MA20-distance, relative-volume, BB-%B, MACD/close
2. Predict log returns instead of raw prices:
       The model outputs log(close[t+1]/close[t]), forcing it to learn direction
       rather than just echoing the last price. Predicted price is recovered via
       last_close * exp(predicted_log_return).
3. Bidirectional LSTM:
       Both forward and backward passes over the input window — captures patterns
       that are symmetric in time (e.g. reversal signals). No leakage because the
       entire input window is historical.
4. Gradient clipping — prevents exploding gradients.
5. ReduceLROnPlateau scheduler — adapts learning rate when val loss plateaus.
6. Early stopping with best-weight restoration.
7. StandardScaler (fit on train only) — handles features on different scales
   without the MinMaxScaler distortion caused by extreme outliers.

Split (chronological, no shuffle across boundaries):
    70 % train | 15 % validation | 15 % test
    Sequence assignment is based on the last timestep of each window, so there
    is zero data leakage between splits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from app.model_store import ModelMeta, ModelStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

SEQ_LEN    = 60     # days of history per input window  (was 30)
HIDDEN     = 128    # LSTM hidden units per direction   (was 64)
NUM_LAYERS = 2
DROPOUT    = 0.3    # (was 0.2)
EPOCHS     = 300    # max epochs — early stopping cuts this short
LR         = 5e-4   # (was 1e-3; lower for stability with bidirectional LSTM)
BATCH_SIZE = 64     # (was 32)
PATIENCE   = 20     # early-stopping patience
GRAD_CLIP  = 1.0    # max gradient norm

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

# Feature set (order matters — log_return must be index 0 for target extraction)
N_FEATURES = 6
FEATURE_NAMES = ["log_return", "rsi_norm", "ma20_dist", "vol_ratio", "bb_pct", "macd_norm"]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """
    Bidirectional 2-layer LSTM with a 2-layer prediction head.

    input:  (batch, seq_len, input_size)
    output: (batch, 1)  — predicted scaled log return
    """

    def __init__(
        self,
        input_size:  int   = N_FEATURES,
        hidden_size: int   = HIDDEN,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Bidirectional doubles the output hidden size
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep → prediction


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a (N-1, N_FEATURES) feature matrix from a DataFrame that already has
    MA20, RSI, BB_pct, MACD, and Volume columns applied.

    Row i represents "the state at the close of day i+1" (0-indexed close array).
    Feature 0 is always log_return = log(close[i+1] / close[i]).

    Missing indicator columns fall back to neutral defaults so the function
    degrades gracefully if called with a plain OHLCV DataFrame.
    """
    N     = len(df)
    close = df["Close"].values.astype(np.float64)
    n     = N - 1                                          # feature rows

    # --- Feature 0: log return ---
    log_ret = np.log(close[1:] / np.maximum(close[:-1], 1e-8))   # (n,)

    # --- Feature 1: RSI normalised to [0, 1] ---
    if "RSI" in df.columns:
        rsi_norm = df["RSI"].values[1:].astype(np.float64) / 100.0
        rsi_norm = np.nan_to_num(rsi_norm, nan=0.5)
    else:
        rsi_norm = np.full(n, 0.5)

    # --- Feature 2: % distance of close from MA20 ---
    if "MA20" in df.columns:
        ma20     = df["MA20"].values[1:].astype(np.float64)
        ma20     = np.where(ma20 > 0, ma20, close[1:])    # avoid /0
        ma20_dist = (close[1:] - ma20) / ma20
        ma20_dist = np.clip(np.nan_to_num(ma20_dist, 0.0), -0.5, 0.5)
    else:
        ma20_dist = np.zeros(n)

    # --- Feature 3: relative volume (today / 20-day rolling mean) ---
    if "Volume" in df.columns:
        vol    = df["Volume"].values.astype(np.float64)
        vol_ma = pd.Series(vol).rolling(20, min_periods=1).mean().values
        vol_ratio = vol[1:] / np.where(vol_ma[1:] > 0, vol_ma[1:], 1.0)
        vol_ratio = np.clip(np.nan_to_num(vol_ratio, 1.0), 0.0, 10.0)
    else:
        vol_ratio = np.ones(n)

    # --- Feature 4: Bollinger %B (close position within bands) ---
    if "BB_pct" in df.columns:
        bb_pct = df["BB_pct"].values[1:].astype(np.float64)
        bb_pct = np.clip(np.nan_to_num(bb_pct, 0.5), -0.5, 1.5)
    elif "BB_upper" in df.columns and "BB_lower" in df.columns:
        bb_upper = df["BB_upper"].values[1:].astype(np.float64)
        bb_lower = df["BB_lower"].values[1:].astype(np.float64)
        bb_range = bb_upper - bb_lower
        bb_pct   = np.where(bb_range > 0, (close[1:] - bb_lower) / bb_range, 0.5)
        bb_pct   = np.clip(np.nan_to_num(bb_pct, 0.5), -0.5, 1.5)
    else:
        bb_pct = np.full(n, 0.5)

    # --- Feature 5: MACD normalised by close ---
    if "MACD" in df.columns:
        macd      = df["MACD"].values[1:].astype(np.float64)
        macd_norm = np.where(close[1:] > 0, macd / close[1:], 0.0)
        macd_norm = np.clip(np.nan_to_num(macd_norm, 0.0), -0.1, 0.1)
    else:
        macd_norm = np.zeros(n)

    return np.column_stack(
        [log_ret, rsi_norm, ma20_dist, vol_ratio, bb_pct, macd_norm]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def _build_sequences(
    features_s: np.ndarray,   # (T, N_FEATURES) — already scaled
    targets_s:  np.ndarray,   # (T,) — scaled log returns
    seq_len:    int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    X[k] = features_s[k : k+seq_len]          shape (seq_len, N_FEATURES)
    y[k] = targets_s[k + seq_len - 1]         the log return AFTER the last window step

    Sequence k is "owned by" the split whose boundary contains index k+seq_len-1.
    This gives clean, leak-free splits when sequences are assigned by last index.
    """
    X, y = [], []
    T = len(features_s)
    for k in range(T - seq_len):
        X.append(features_s[k : k + seq_len])
        y.append(targets_s[k + seq_len - 1])
    return (
        torch.FloatTensor(np.array(X)),               # (N, seq_len, N_FEATURES)
        torch.FloatTensor(np.array(y).reshape(-1, 1)), # (N, 1)
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    model:            LSTMPredictor,
    scaler:           StandardScaler,
    X_test:           torch.Tensor,
    y_test_s:         torch.Tensor,    # scaled log returns
    test_last_closes: np.ndarray,      # reference close prices for USD conversion
) -> tuple[float, float, float]:
    """
    Evaluate on the held-out test set.

    Returns:
        (test_mae_usd, test_rmse_usd, directional_accuracy)

    Directional accuracy = fraction of test days where sign(predicted log return)
    matches sign(actual log return).  This is the key trading-relevance metric.
    """
    model.eval()
    with torch.no_grad():
        preds_s = model(X_test).numpy().reshape(-1)
    trues_s = y_test_s.numpy().reshape(-1)

    # Inverse-scale using the log_return column's stats (feature index 0)
    lr_mean = float(scaler.mean_[0])
    lr_std  = float(scaler.scale_[0])
    pred_lr = preds_s * lr_std + lr_mean
    true_lr = trues_s * lr_std + lr_mean

    # Convert log returns → price predictions
    pred_prices = test_last_closes * np.exp(pred_lr)
    true_prices = test_last_closes * np.exp(true_lr)

    errors = pred_prices - true_prices
    mae    = float(np.abs(errors).mean())
    rmse   = float(np.sqrt((errors ** 2).mean()))

    # Directional accuracy on log returns (sign-based)
    dir_acc = float(np.mean(np.sign(pred_lr) == np.sign(true_lr)))

    return round(mae, 4), round(rmse, 4), round(dir_acc, 4)


# ---------------------------------------------------------------------------
# Core training pipeline
# ---------------------------------------------------------------------------

def _train(
    df:     pd.DataFrame,
    ticker: str,
    period: str,
) -> tuple[LSTMPredictor, StandardScaler, ModelMeta, float]:
    """
    Full pipeline: feature engineering → chronological split → scale →
    train with early stopping → evaluate on test set → persist → return.

    Sequence assignment (no leakage):
        A sequence is assigned to the split whose boundary contains its
        LAST timestep index.  This means val/test sequences only receive
        labels that belong to their own time period.

    Returns:
        (model, scaler, meta, predicted_next_day_price)
    """
    close       = df["Close"].values.astype(np.float64)   # (N,)
    raw_features = _build_feature_matrix(df)               # (N-1, N_FEATURES)

    # Align: features[i] → predict targets[i] = log_return at features[i+1][0]
    features = raw_features[:-1]                            # (N-2, N_FEATURES)
    targets  = raw_features[1:, 0]                          # (N-2,) — next-day log return

    T = len(features)   # N-2
    min_rows = SEQ_LEN * 3
    if T < min_rows:
        raise ValueError(
            f"Need at least {min_rows} aligned rows; got {T}. "
            "Use a longer period (e.g., period='2y')."
        )

    n_train = int(T * TRAIN_FRAC)
    n_val   = int(T * VAL_FRAC)

    # --- Scale features (fit on train ONLY) ---
    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features[:n_train])           # fit
    features_scaled = scaler.transform(features)                          # transform all
    targets_scaled  = ((targets - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)

    # --- Build all sequences then split by last-timestep index ---
    X_all, y_all = _build_sequences(features_scaled, targets_scaled, SEQ_LEN)
    # Sequence k: last timestep index = k + SEQ_LEN - 1
    n_split_train    = n_train - SEQ_LEN + 1       # last sequences fully in train
    n_split_val_end  = n_train + n_val - SEQ_LEN + 1

    if n_split_train <= 0:
        raise ValueError("Not enough training data for the given SEQ_LEN.")

    X_train, y_train = X_all[:n_split_train],             y_all[:n_split_train]
    X_val,   y_val   = X_all[n_split_train:n_split_val_end], y_all[n_split_train:n_split_val_end]
    X_test,  y_test  = X_all[n_split_val_end:],           y_all[n_split_val_end:]

    # Reference close prices for USD-based test metrics
    # Test sequence k (global index n_split_val_end + k): last feat = n_split_val_end + k + SEQ_LEN - 1
    # features[i] uses close[i+1] as "current close" → reference = close[n_split_val_end+k+SEQ_LEN]
    n_test_seqs       = len(X_test)
    test_close_offset = n_split_val_end + SEQ_LEN
    test_last_closes  = close[test_close_offset : test_close_offset + n_test_seqs]

    # --- Model + optimiser + scheduler ---
    model     = LSTMPredictor(input_size=N_FEATURES, hidden_size=HIDDEN)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    best_train_loss  = float("inf")
    best_state_dict  = None
    patience_counter = 0
    n_samples        = X_train.shape[0]

    for epoch in range(EPOCHS):
        # --- Train step ---
        model.train()
        perm     = torch.randperm(n_samples)
        X_s, y_s = X_train[perm], y_train[perm]

        epoch_loss = 0.0
        batches    = 0
        for start in range(0, n_samples, BATCH_SIZE):
            xb = X_s[start : start + BATCH_SIZE]
            yb = y_s[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            epoch_loss += loss.item()
            batches    += 1
        train_loss = epoch_loss / max(batches, 1)

        # --- Val step ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(val_loss)

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-7:
            best_val_loss   = val_loss
            best_train_loss = train_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(
                "Early stopping at epoch %d for %s/%s (best val=%.6f)",
                epoch + 1, ticker, period, best_val_loss,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # --- Test evaluation ---
    test_mae, test_rmse, dir_acc = _compute_metrics(
        model, scaler, X_test, y_test, test_last_closes
    )

    # --- Next-day inference ---
    model.eval()
    inf_features = scaler.transform(raw_features[-SEQ_LEN:])   # last SEQ_LEN rows
    with torch.no_grad():
        last_seq  = torch.FloatTensor(inf_features).unsqueeze(0)  # (1, SEQ_LEN, N_FEATURES)
        pred_s    = model(last_seq).item()
    pred_lr       = pred_s * float(scaler.scale_[0]) + float(scaler.mean_[0])
    pred_price    = float(close[-1]) * float(np.exp(pred_lr))

    meta = ModelMeta(
        ticker               = ticker,
        period               = period,
        trained_at           = datetime.now(timezone.utc).isoformat(),
        train_loss           = round(float(best_train_loss), 6),
        val_loss             = round(float(best_val_loss),   6),
        test_mae             = test_mae,
        test_rmse            = test_rmse,
        directional_accuracy = dir_acc,
        n_train              = int(X_train.shape[0]),
        n_val                = int(X_val.shape[0]),
        n_test               = int(X_test.shape[0]),
        input_size           = N_FEATURES,
        hidden_size          = HIDDEN,
    )

    ModelStore.save(ticker, period, model, scaler, meta)
    logger.info(
        "Trained %s/%s — test MAE $%.2f | RMSE $%.2f | dir_acc %.1f%%",
        ticker, period, test_mae, test_rmse, dir_acc * 100,
    )

    return model, scaler, meta, round(pred_price, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lstm_predict(
    df:            pd.DataFrame,
    ticker:        str,
    period:        str  = "1y",
    force_retrain: bool = False,
) -> tuple[float, ModelMeta]:
    """
    Return (predicted_next_day_price, ModelMeta).

    Loads from disk when a fresh model exists; retrains otherwise.
    Architecture mismatches (e.g. after upgrading N_FEATURES) are caught
    and trigger a silent retrain.
    """
    min_rows = SEQ_LEN + 10
    if len(df) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows; got {len(df)}. "
            "Use a longer period (e.g., period='6mo')."
        )

    if not force_retrain and not ModelStore.needs_training(ticker, period):
        try:
            logger.info("Cache hit — loading model for %s/%s", ticker, period)
            model, scaler = ModelStore.load(ticker, period)
            meta          = ModelStore.load_meta(ticker, period)

            raw_features = _build_feature_matrix(df)
            inf_features = scaler.transform(raw_features[-SEQ_LEN:])
            model.eval()
            with torch.no_grad():
                last_seq  = torch.FloatTensor(inf_features).unsqueeze(0)
                pred_s    = model(last_seq).item()
            pred_lr    = pred_s * float(scaler.scale_[0]) + float(scaler.mean_[0])
            pred_price = float(df["Close"].iloc[-1]) * float(np.exp(pred_lr))
            return round(pred_price, 4), meta

        except Exception as exc:
            logger.warning(
                "Cache load failed for %s/%s (%s) — retraining.", ticker, period, exc
            )

    logger.info("Training model for %s/%s", ticker, period)
    _, _, meta, pred_price = _train(df, ticker, period)
    return pred_price, meta


def retrain_ticker(ticker: str, period: str = "1y") -> ModelMeta:
    """Force-retrain a ticker. Used by the APScheduler background job."""
    from app.data       import get_stock_data
    from app.indicators import add_all_indicators

    logger.info("Scheduled retrain: %s/%s", ticker, period)
    df = get_stock_data(ticker, period=period)
    df = add_all_indicators(df)
    _, _, meta, _ = _train(df, ticker, period)
    return meta
