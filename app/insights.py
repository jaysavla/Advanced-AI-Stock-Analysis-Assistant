"""
insights.py - Human-readable insight engine.

Combines RSI, MA20 trend, predicted price direction, and news sentiment
to produce a concise, actionable insight string for the API response.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from app.prediction import trend_direction


# RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30


def generate_insight(
    df: pd.DataFrame,
    sentiment: Optional[dict] = None,
) -> str:
    """
    Generate a plain-English market insight for the latest data point.

    Logic:
        1. RSI signal    — overbought / oversold / neutral
        2. Trend signal  — MA20 slope direction
        3. MA position   — price above / below MA20
        4. Sentiment     — news bullish / bearish / neutral (optional)

    Args:
        df:        DataFrame with 'Close', 'MA20', and 'RSI' columns.
        sentiment: Output of sentiment.analyze_sentiment() or None to skip.

    Returns:
        A single insight string (non-empty).
    """
    parts: list[str] = []

    # -----------------------------------------------------------------------
    # 1. RSI signal
    # -----------------------------------------------------------------------
    if "RSI" in df.columns:
        rsi = float(df["RSI"].iloc[-1])
        if rsi >= RSI_OVERBOUGHT:
            parts.append(
                f"RSI is elevated at {rsi:.1f} (\u2265{RSI_OVERBOUGHT}) — "
                "stock may be overbought; consider waiting for a pullback."
            )
        elif rsi <= RSI_OVERSOLD:
            parts.append(
                f"RSI is depressed at {rsi:.1f} (\u2264{RSI_OVERSOLD}) — "
                "stock may be oversold; potential buying opportunity, confirm with other signals."
            )
        else:
            parts.append(f"RSI is neutral at {rsi:.1f}.")

    # -----------------------------------------------------------------------
    # 2. Trend signal (MA20 slope)
    # -----------------------------------------------------------------------
    trend = trend_direction(df)
    if trend == "uptrend":
        parts.append("MA20 is sloping upward — bullish short-term momentum.")
    elif trend == "downtrend":
        parts.append("MA20 is sloping downward — bearish short-term momentum.")
    else:
        parts.append("Price is moving sideways relative to MA20.")

    # -----------------------------------------------------------------------
    # 3. Price vs MA20 position
    # -----------------------------------------------------------------------
    if "MA20" in df.columns:
        close = float(df["Close"].iloc[-1])
        ma20  = float(df["MA20"].iloc[-1])
        pct   = (close - ma20) / ma20 * 100
        if close > ma20:
            parts.append(
                f"Price is {pct:.1f}% above the 20-day moving average (positive sign)."
            )
        else:
            parts.append(
                f"Price is {abs(pct):.1f}% below the 20-day moving average (caution)."
            )

    # -----------------------------------------------------------------------
    # 4. News sentiment (multi-modal signal)
    # -----------------------------------------------------------------------
    if sentiment and sentiment.get("headline_count", 0) > 0:
        label = sentiment["label"]
        score = sentiment["score"]
        count = sentiment["headline_count"]
        pos   = sentiment["positive"]
        neg   = sentiment["negative"]

        if label == "bullish":
            parts.append(
                f"News sentiment is BULLISH (score {score:.2f}, "
                f"{pos}/{count} positive headlines) — market narrative supports upside."
            )
        elif label == "bearish":
            parts.append(
                f"News sentiment is BEARISH (score {score:.2f}, "
                f"{neg}/{count} negative headlines) — market narrative signals caution."
            )
        else:
            parts.append(
                f"News sentiment is NEUTRAL (score {score:.2f} across {count} headlines)."
            )
    elif sentiment is not None:
        parts.append("No recent news headlines found for sentiment analysis.")

    return " ".join(parts)
