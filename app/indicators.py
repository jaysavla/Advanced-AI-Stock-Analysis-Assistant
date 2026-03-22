"""
indicators.py - Technical indicator calculations.

Each function accepts a DataFrame (must contain a 'Close' column) and
returns the same DataFrame with new indicator columns appended.
All computations use pure pandas/numpy to avoid heavy optional deps.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Moving Average
# ---------------------------------------------------------------------------

def add_moving_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Append a simple moving average column (MA{window}) to the DataFrame.

    Args:
        df:     DataFrame with at least a 'Close' column.
        window: Rolling window size in trading days (default 20).

    Returns:
        DataFrame with new column f"MA{window}".
    """
    df = df.copy()
    col = f"MA{window}"
    df[col] = df["Close"].rolling(window=window, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Append a Wilder-smoothed RSI column to the DataFrame.

    Args:
        df:     DataFrame with at least a 'Close' column.
        window: Lookback period (default 14).

    Returns:
        DataFrame with new column 'RSI'.
    """
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing (equivalent to EWM with alpha = 1/window)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # neutral default for early rows
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all available indicators in one call."""
    df = add_moving_average(df, window=20)
    df = add_rsi(df, window=14)
    return df
