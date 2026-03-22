"""
data.py - Stock data fetching layer using yfinance.
"""

import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given stock ticker.

    Args:
        ticker: Stock symbol (e.g., "AAPL", "TSLA").
        period: Lookback period accepted by yfinance (e.g., "1y", "6mo", "3mo").

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume (DatetimeIndex).

    Raises:
        ValueError: If the ticker is invalid or no data is returned.
    """
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as exc:
        raise ValueError(f"Failed to download data for '{ticker}': {exc}") from exc

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check that the symbol is correct and the market is active."
        )

    # Flatten MultiIndex columns produced by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only the standard OHLCV columns that are present
    desired = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in desired if c in raw.columns]
    df = raw[available].copy()

    df.index.name = "Date"
    return df
