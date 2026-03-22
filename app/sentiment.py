"""
sentiment.py - News fetching + sentiment analysis layer.

Pipeline:
  1. Fetch recent headlines via yfinance (no extra API key needed).
  2. Score each headline with a DistilBERT SST-2 model from HuggingFace.
  3. Aggregate into a single sentiment dict consumed by insights.py.

The HuggingFace model (~67 MB) is downloaded once and cached by the
`transformers` library in ~/.cache/huggingface on first run.
"""

from __future__ import annotations

import logging
from typing import Optional

import yfinance as yf
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton pipeline
# ---------------------------------------------------------------------------
_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
_pipeline: Optional[object] = None


def _get_pipeline():
    """Return the HuggingFace sentiment pipeline, loading it once."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading sentiment model '%s' (first call only)…", _SENTIMENT_MODEL)
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=_SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------

def fetch_headlines(ticker: str, max_items: int = 15) -> list[str]:
    """
    Pull recent news headlines for a ticker from yfinance.

    Args:
        ticker:    Stock symbol (e.g. "AAPL").
        max_items: Maximum number of headlines to return.

    Returns:
        List of headline strings (may be empty if no news is available).
    """
    try:
        t     = yf.Ticker(ticker)
        news  = t.news or []
    except Exception as exc:
        logger.warning("Could not fetch news for %s: %s", ticker, exc)
        return []

    headlines: list[str] = []
    for item in news[:max_items]:
        # yfinance news schema varies across versions
        title = (
            item.get("title")
            or (item.get("content") or {}).get("title")
            or ""
        )
        if title:
            headlines.append(title.strip())

    return headlines


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def analyze_sentiment(ticker: str) -> dict:
    """
    Score recent news headlines and return an aggregated sentiment dict.

    Returns:
        {
            "label":          "bullish" | "bearish" | "neutral",
            "score":          float,   # 0 (very bearish) → 1 (very bullish)
            "headline_count": int,
            "positive":       int,
            "negative":       int,
            "headlines":      list[str],
        }

    If no headlines are found, returns a neutral result.
    """
    neutral_result = {
        "label": "neutral",
        "score": 0.5,
        "headline_count": 0,
        "positive": 0,
        "negative": 0,
        "headlines": [],
    }

    headlines = fetch_headlines(ticker)
    if not headlines:
        return neutral_result

    try:
        pipe    = _get_pipeline()
        results = pipe(headlines)
    except Exception as exc:
        logger.error("Sentiment pipeline failed: %s", exc)
        return neutral_result

    positive_count = 0
    score_sum      = 0.0

    for r in results:
        if r["label"] == "POSITIVE":
            positive_count += 1
            score_sum      += r["score"]          # confidence that it's positive
        else:
            score_sum      += (1.0 - r["score"])  # flip: how positive is a negative?

    avg_score = score_sum / len(results)

    if avg_score >= 0.62:
        label = "bullish"
    elif avg_score <= 0.38:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "label":          label,
        "score":          round(avg_score, 4),
        "headline_count": len(headlines),
        "positive":       positive_count,
        "negative":       len(results) - positive_count,
        "headlines":      headlines,
    }
