"""
main.py - FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Endpoints:
    GET /                    — health check
    GET /analyze/{ticker}    — full AI analysis (LSTM + sentiment + indicators)
    GET /chart/{ticker}      — PNG chart (price, MA20, RSI)
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.data       import get_stock_data
from app.indicators import add_all_indicators
from app.prediction import predict_price, trend_direction
from app.insights   import generate_insight
from app.plot       import generate_chart
from app.model      import lstm_predict
from app.sentiment  import analyze_sentiment

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Stock Analysis Assistant",
    description=(
        "End-to-end stock analysis: real market data · MA20 & RSI indicators · "
        "LSTM price prediction · news sentiment analysis · interactive charts."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SentimentSummary(BaseModel):
    label:          str    # bullish | bearish | neutral
    score:          float  # 0 → 1
    headline_count: int
    positive:       int
    negative:       int


class AnalysisResponse(BaseModel):
    ticker:              str
    period:              str
    latest_price:        float
    predicted_price:     float   # LSTM (falls back to baseline on failure)
    prediction_method:   str     # "lstm" | "baseline"
    trend:               str     # uptrend | downtrend | sideways
    insight:             str
    sentiment:           Optional[SentimentSummary]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check", tags=["meta"])
def root() -> dict:
    """Confirms the API is running."""
    return {"status": "ok", "version": app.version, "message": "AI Stock Analysis Assistant"}


@app.get(
    "/analyze/{ticker}",
    response_model=AnalysisResponse,
    summary="Full AI analysis for a stock ticker",
    tags=["analysis"],
)
def analyze(
    ticker: str,
    period: str = Query(default="1y", description="yfinance period: 1mo 3mo 6mo 1y 2y 5y"),
    include_sentiment: bool = Query(default=True, description="Run news sentiment analysis"),
) -> AnalysisResponse:
    """
    Full analysis pipeline:

    1. **Data** — fetch OHLCV via yfinance
    2. **Indicators** — compute MA20 and RSI
    3. **LSTM prediction** — train a 2-layer LSTM on historical closes, predict next day
       *(falls back to the rule-based baseline if not enough history)*
    4. **Sentiment** — score recent news headlines with DistilBERT SST-2
    5. **Insight** — combine all signals into a human-readable summary
    """
    ticker = ticker.upper().strip()

    # 1. Fetch data
    try:
        df = get_stock_data(ticker, period=period)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # 2. Indicators
    df = add_all_indicators(df)

    # 3. LSTM prediction (with fallback)
    prediction_method = "lstm"
    try:
        predicted = lstm_predict(df)
    except Exception as exc:
        logger.warning("LSTM prediction failed for %s (%s), falling back to baseline.", ticker, exc)
        try:
            predicted = predict_price(df)
            prediction_method = "baseline"
        except ValueError as exc2:
            raise HTTPException(status_code=422, detail=str(exc2))

    # 4. Trend + sentiment
    trend     = trend_direction(df)
    sentiment = analyze_sentiment(ticker) if include_sentiment else None

    # 5. Insight
    insight = generate_insight(df, sentiment=sentiment)

    # Build sentiment sub-object (strip 'headlines' list from JSON response)
    sentiment_out: Optional[SentimentSummary] = None
    if sentiment:
        sentiment_out = SentimentSummary(
            label          = sentiment["label"],
            score          = sentiment["score"],
            headline_count = sentiment["headline_count"],
            positive       = sentiment["positive"],
            negative       = sentiment["negative"],
        )

    return AnalysisResponse(
        ticker            = ticker,
        period            = period,
        latest_price      = round(float(df["Close"].iloc[-1]), 4),
        predicted_price   = predicted,
        prediction_method = prediction_method,
        trend             = trend,
        insight           = insight,
        sentiment         = sentiment_out,
    )


@app.get(
    "/chart/{ticker}",
    summary="PNG price chart with MA20 and RSI",
    tags=["visualization"],
    responses={200: {"content": {"image/png": {}}}},
)
def chart(
    ticker: str,
    period: str = Query(default="1y", description="yfinance period: 1mo 3mo 6mo 1y 2y 5y"),
) -> StreamingResponse:
    """
    Returns a dark-themed PNG chart with two panels:
    - **Top**: closing price + MA20 overlay
    - **Bottom**: RSI with overbought (70) and oversold (30) bands
    """
    ticker = ticker.upper().strip()

    try:
        df = get_stock_data(ticker, period=period)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    df      = add_all_indicators(df)
    png     = generate_chart(df, ticker)

    return StreamingResponse(
        iter([png]),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{ticker}_{period}.png"'},
    )
