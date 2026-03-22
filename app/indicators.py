"""
indicators.py - Technical indicator calculations.

Each function accepts a DataFrame (must contain a 'Close' column) and
returns the same DataFrame with new indicator columns appended.
All computations use pure pandas/numpy.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Moving Average (simple)
# ---------------------------------------------------------------------------

def add_moving_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[f"MA{window}"] = df["Close"].rolling(window=window, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    return df


# ---------------------------------------------------------------------------
# MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------------------------

def add_macd(
    df:     pd.DataFrame,
    fast:   int = 12,
    slow:   int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Append MACD line, signal line, and histogram columns.

    Columns added: MACD, MACD_signal, MACD_hist.
    """
    df       = df.copy()
    ema_fast = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def add_bollinger_bands(
    df:      pd.DataFrame,
    window:  int   = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Append Bollinger Band columns.

    Columns added:
        BB_upper  — upper band (MA + num_std * rolling σ)
        BB_lower  — lower band (MA − num_std * rolling σ)
        BB_pct    — %B position: 0 = at lower band, 1 = at upper band
    """
    df        = df.copy()
    roll_mean = df["Close"].rolling(window=window, min_periods=1).mean()
    roll_std  = df["Close"].rolling(window=window, min_periods=1).std().fillna(0)
    df["BB_upper"] = roll_mean + num_std * roll_std
    df["BB_lower"] = roll_mean - num_std * roll_std
    band_width     = (df["BB_upper"] - df["BB_lower"]).replace(0, float("nan"))
    df["BB_pct"]   = ((df["Close"] - df["BB_lower"]) / band_width).clip(0, 1).fillna(0.5)
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all available indicators in one call."""
    df = add_moving_average(df, window=20)
    df = add_rsi(df, window=14)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    return df
