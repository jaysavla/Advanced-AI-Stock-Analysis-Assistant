"""
model_store.py - Disk-based persistence layer for trained LSTM models.

Directory layout  (MODEL_DIR = ./model_cache/):
    {TICKER}_{period}.pt             — PyTorch state dict
    {TICKER}_{period}_scaler.pkl     — fitted StandardScaler (inverse-transform on inference)
    {TICKER}_{period}_meta.json      — training metadata + evaluation metrics

A model is considered "stale" when it is older than STALE_AFTER_HOURS (default 24 h).
lstm_predict() checks staleness before deciding whether to load or retrain.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from app.model import LSTMPredictor

logger = logging.getLogger(__name__)

MODEL_DIR = Path("model_cache")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STALE_AFTER_HOURS: int = 24


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelMeta:
    ticker:               str
    period:               str
    trained_at:           str    # ISO-8601 UTC timestamp
    train_loss:           float  # MSE on training set (normalised)
    val_loss:             float  # best MSE on validation set (normalised)
    test_mae:             float  # mean absolute error in USD on held-out test set
    test_rmse:            float  # root mean squared error in USD on held-out test set
    directional_accuracy: float  # fraction of days where predicted direction == actual
    n_train:              int    # number of training sequences
    n_val:                int    # number of validation sequences
    n_test:               int    # number of test sequences
    input_size:           int   = 6    # N_FEATURES used to build this model
    hidden_size:          int   = 128  # LSTM hidden units per direction


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------

class ModelStore:
    """Static helpers — read/write model weights, scaler, and metadata to disk."""

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stem(ticker: str, period: str) -> Path:
        return MODEL_DIR / f"{ticker.upper()}_{period}"

    @classmethod
    def _pt_path(cls, ticker: str, period: str) -> Path:
        return cls._stem(ticker, period).with_suffix(".pt")

    @classmethod
    def _scaler_path(cls, ticker: str, period: str) -> Path:
        s = cls._stem(ticker, period)
        return s.parent / (s.name + "_scaler.pkl")

    @classmethod
    def _meta_path(cls, ticker: str, period: str) -> Path:
        return cls._stem(ticker, period).with_suffix(".json")

    # ------------------------------------------------------------------
    # Existence / staleness
    # ------------------------------------------------------------------

    @classmethod
    def exists(cls, ticker: str, period: str) -> bool:
        return (
            cls._pt_path(ticker, period).exists()
            and cls._meta_path(ticker, period).exists()
            and cls._scaler_path(ticker, period).exists()
        )

    @classmethod
    def is_stale(cls, ticker: str, period: str) -> bool:
        meta = cls.load_meta(ticker, period)
        if meta is None:
            return True
        trained_at = datetime.fromisoformat(meta.trained_at)
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600
        return age_hours > STALE_AFTER_HOURS

    @classmethod
    def needs_training(cls, ticker: str, period: str) -> bool:
        return not cls.exists(ticker, period) or cls.is_stale(ticker, period)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @classmethod
    def save(
        cls,
        ticker: str,
        period: str,
        model:  "LSTMPredictor",
        scaler: StandardScaler,
        meta:   ModelMeta,
    ) -> None:
        torch.save(model.state_dict(), cls._pt_path(ticker, period))
        with open(cls._scaler_path(ticker, period), "wb") as f:
            pickle.dump(scaler, f)
        with open(cls._meta_path(ticker, period), "w") as f:
            json.dump(asdict(meta), f, indent=2)
        logger.info("Model saved: %s/%s → %s", ticker, period, cls._stem(ticker, period))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, ticker: str, period: str) -> tuple["LSTMPredictor", StandardScaler]:
        """
        Load persisted model weights + scaler from disk.

        Uses input_size and hidden_size stored in metadata so the model
        architecture is always reconstructed correctly.

        Raises:
            FileNotFoundError: If the model has not been trained yet.
        """
        from app.model import LSTMPredictor, N_FEATURES, HIDDEN

        meta         = cls.load_meta(ticker, period)
        input_size   = meta.input_size  if meta else N_FEATURES
        hidden_size  = meta.hidden_size if meta else HIDDEN

        model = LSTMPredictor(input_size=input_size, hidden_size=hidden_size)
        model.load_state_dict(
            torch.load(cls._pt_path(ticker, period), map_location="cpu", weights_only=True)
        )
        model.eval()

        with open(cls._scaler_path(ticker, period), "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        return model, scaler

    @classmethod
    def load_meta(cls, ticker: str, period: str) -> Optional[ModelMeta]:
        path = cls._meta_path(ticker, period)
        if not path.exists():
            return None
        with open(path) as f:
            return ModelMeta(**json.load(f))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def all_cached(cls) -> list[tuple[str, str]]:
        """Return (ticker, period) for every model currently persisted on disk."""
        results: list[tuple[str, str]] = []
        for meta_path in MODEL_DIR.glob("*.json"):
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                results.append((data["ticker"], data["period"]))
            except Exception:
                pass
        return results
