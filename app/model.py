"""
model.py - LSTM-based price prediction using PyTorch.

Architecture:
    - 2-layer LSTM with dropout
    - Single linear head predicting next-day normalised close
    - MinMaxScaler applied per-call (stateless, no persisted weights)

The model trains on the full available history each call. For a production
system you would persist weights and retrain on a schedule; for this MVP
fresh training is intentionally simple and transparent.

Typical training time on CPU: ~3-5 s for 250 epochs on 1 year of daily data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Hyper-parameters (tuned for daily OHLCV, 1-year window)
# ---------------------------------------------------------------------------
SEQ_LEN    = 30    # days of history fed into each LSTM step
HIDDEN     = 64    # LSTM hidden units per layer
NUM_LAYERS = 2
DROPOUT    = 0.2
EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 32    # mini-batch gradient descent


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """Two-layer LSTM that maps a (batch, seq_len, 1) tensor → scalar."""

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = HIDDEN,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep → prediction


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_sequences(
    scaled: np.ndarray,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slide a window over the scaled price series to build (X, y) tensors."""
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i : i + seq_len])
        y.append(scaled[i + seq_len])
    X_t = torch.FloatTensor(np.array(X))          # (N, seq_len, 1)
    y_t = torch.FloatTensor(np.array(y))          # (N, 1)
    return X_t, y_t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lstm_predict(df: pd.DataFrame) -> float:
    """
    Train an LSTM on historical closing prices and predict the next day.

    Args:
        df: DataFrame with a 'Close' column (at least SEQ_LEN + 10 rows).

    Returns:
        Predicted next-day closing price (float, rounded to 4 dp).

    Raises:
        ValueError: If the DataFrame doesn't have enough rows.
    """
    min_rows = SEQ_LEN + 10
    if len(df) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows to train the LSTM; "
            f"got {len(df)}. Use a longer period (e.g., period=6mo)."
        )

    # --- Normalise ---
    prices  = df["Close"].values.reshape(-1, 1).astype(np.float32)
    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaled  = scaler.fit_transform(prices)          # (T, 1)

    # --- Build dataset ---
    X, y = _build_sequences(scaled, SEQ_LEN)

    # --- Model + optimiser ---
    model     = LSTMPredictor()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # --- Mini-batch training ---
    n_samples = X.shape[0]
    model.train()
    for _ in range(EPOCHS):
        # Shuffle each epoch
        perm = torch.randperm(n_samples)
        X_s, y_s = X[perm], y[perm]

        for start in range(0, n_samples, BATCH_SIZE):
            xb = X_s[start : start + BATCH_SIZE]
            yb = y_s[start : start + BATCH_SIZE]

            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        last_seq  = torch.FloatTensor(scaled[-SEQ_LEN:]).unsqueeze(0)   # (1, seq, 1)
        pred_norm = model(last_seq).item()
        # Inverse-transform: clip to [0,1] first to avoid scaler overflow
        pred_norm = float(np.clip(pred_norm, 0.0, 1.0))
        pred_price = scaler.inverse_transform([[pred_norm]])[0][0]

    return round(float(pred_price), 4)
