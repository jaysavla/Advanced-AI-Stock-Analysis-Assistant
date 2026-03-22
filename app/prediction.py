"""
prediction.py - Baseline price prediction logic.

Strategy (no deep learning):
  1. Compute the short-term momentum: average daily change over the last N days.
  2. Blend the last close price with the MA20 to anchor the estimate.
  3. Project one day forward using the blended base + momentum.

This is intentionally simple and transparent — a solid MVP baseline before
any ML model is introduced.
"""

import pandas as pd


def predict_price(df: pd.DataFrame, momentum_window: int = 5) -> float:
    """
    Predict the next closing price using a trend-adjusted moving-average model.

    Requirements:
        df must already contain 'Close' and 'MA20' columns
        (produced by indicators.add_all_indicators).

    Args:
        df:               DataFrame with indicator columns applied.
        momentum_window:  Number of recent days used to estimate trend momentum.

    Returns:
        Predicted next-day closing price (float, rounded to 4 dp).

    Raises:
        ValueError: If required columns are missing or the DataFrame is too short.
    """
    required = {"Close", "MA20"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if len(df) < momentum_window:
        raise ValueError(
            f"DataFrame has fewer rows ({len(df)}) than momentum_window ({momentum_window})."
        )

    latest_close: float = float(df["Close"].iloc[-1])
    latest_ma20: float = float(df["MA20"].iloc[-1])

    # Average daily price change over the momentum window
    recent_closes = df["Close"].iloc[-momentum_window:]
    avg_daily_change: float = float(recent_closes.diff().dropna().mean())

    # Blended base: 70 % last close + 30 % MA20 (anchors to trend)
    blended_base = 0.7 * latest_close + 0.3 * latest_ma20

    predicted = blended_base + avg_daily_change
    return round(predicted, 4)


def trend_direction(df: pd.DataFrame) -> str:
    """
    Derive a simple trend label from the slope of MA20 over the last 5 days.

    Returns:
        "uptrend", "downtrend", or "sideways".
    """
    if "MA20" not in df.columns or len(df) < 5:
        return "sideways"

    ma_recent = df["MA20"].iloc[-5:]
    slope = float(ma_recent.iloc[-1]) - float(ma_recent.iloc[0])
    threshold = float(df["Close"].iloc[-1]) * 0.005  # 0.5 % of price

    if slope > threshold:
        return "uptrend"
    elif slope < -threshold:
        return "downtrend"
    return "sideways"
