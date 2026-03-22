"""
main.py - FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Endpoints:
    GET  /                          — health check
    GET  /analyze/{ticker}          — full AI analysis (LSTM + sentiment + indicators)
    GET  /chart/{ticker}            — PNG chart (price, MA20, RSI)
    GET  /model/status/{ticker}     — cached model metrics (MAE, RMSE, directional accuracy)
    POST /model/retrain/{ticker}    — trigger a background retrain for a ticker

Scheduler:
    APScheduler runs a background job at 02:00 UTC every day to retrain
    all stale cached models, so the first request of the day is always fast.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.data        import get_stock_data
from app.indicators  import add_all_indicators
from app.model       import lstm_predict, retrain_ticker
from app.model_store import ModelStore
from app.prediction  import predict_price, trend_direction
from app.insights    import generate_insight
from app.plot        import generate_chart
from app.sentiment   import analyze_sentiment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------

_scheduler = BackgroundScheduler(timezone="UTC")


def _retrain_all_stale() -> None:
    """Retrain every cached model that has gone stale. Runs at 02:00 UTC daily."""
    cached = ModelStore.all_cached()
    if not cached:
        logger.info("Scheduled retrain: no cached models found.")
        return

    for ticker, period in cached:
        if ModelStore.is_stale(ticker, period):
            try:
                retrain_ticker(ticker, period)
                logger.info("Scheduled retrain complete: %s/%s", ticker, period)
            except Exception as exc:
                logger.error("Scheduled retrain failed for %s/%s: %s", ticker, period, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _scheduler.add_job(
        _retrain_all_stale,
        trigger="cron",
        hour=2,
        minute=0,
        id="daily_retrain",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("APScheduler started — stale models will retrain daily at 02:00 UTC.")
    yield
    _scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Stock Analysis Assistant",
    description=(
        "End-to-end stock analysis: real market data · MA20 & RSI indicators · "
        "LSTM price prediction with disk-cached weights · news sentiment analysis · "
        "interactive charts. Models are retrained automatically at 02:00 UTC daily."
    ),
    version="3.0.0",
    lifespan=lifespan,
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


class LSTMMetrics(BaseModel):
    """Evaluation metrics from the model's held-out test set."""
    test_mae:             float   # mean absolute error in USD
    test_rmse:            float   # root mean squared error in USD
    directional_accuracy: float   # fraction of days with correct direction (0–1)
    train_loss:           float   # MSE on training set (normalised)
    val_loss:             float   # best MSE on validation set (normalised)
    n_train:              int     # training sequence count
    n_val:                int     # validation sequence count
    n_test:               int     # test sequence count
    trained_at:           str     # ISO-8601 UTC


class AnalysisResponse(BaseModel):
    ticker:            str
    period:            str
    latest_price:      float
    predicted_price:   float
    prediction_method: str              # "lstm_cached" | "lstm_fresh" | "baseline"
    trend:             str              # uptrend | downtrend | sideways
    insight:           str
    sentiment:         Optional[SentimentSummary]
    lstm_metrics:      Optional[LSTMMetrics]   # None only when baseline fallback is used


class ModelStatusResponse(BaseModel):
    ticker:  str
    period:  str
    status:  str   # "fresh" | "stale" | "not_trained"
    metrics: Optional[LSTMMetrics]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check", tags=["meta"])
def root() -> dict:
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
    3. **LSTM prediction** — load cached model (if fresh) or train a new one;
       falls back to the rule-based baseline if training fails
    4. **Sentiment** — score recent headlines with DistilBERT SST-2
    5. **Insight** — combine all signals into a human-readable summary

    The `lstm_metrics` field in the response exposes the model's held-out test
    MAE, RMSE, and directional accuracy so you can judge prediction reliability.
    """
    ticker = ticker.upper().strip()

    # 1. Fetch data
    try:
        df = get_stock_data(ticker, period=period)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # 2. Indicators
    df = add_all_indicators(df)

    # 3. LSTM prediction (cache-first, with baseline fallback)
    prediction_method = "lstm_fresh"
    lstm_meta         = None
    try:
        predicted, lstm_meta = lstm_predict(df, ticker=ticker, period=period)
        # Distinguish cache hit vs fresh training for the response
        if not ModelStore.is_stale(ticker, period):
            prediction_method = "lstm_cached"
    except Exception as exc:
        logger.warning(
            "LSTM prediction failed for %s (%s), falling back to baseline.", ticker, exc
        )
        try:
            predicted         = predict_price(df)
            prediction_method = "baseline"
        except ValueError as exc2:
            raise HTTPException(status_code=422, detail=str(exc2))

    # 4. Trend + sentiment
    trend     = trend_direction(df)
    sentiment = analyze_sentiment(ticker) if include_sentiment else None

    # 5. Insight
    insight = generate_insight(df, sentiment=sentiment)

    # Build sub-objects
    sentiment_out: Optional[SentimentSummary] = None
    if sentiment:
        sentiment_out = SentimentSummary(
            label          = sentiment["label"],
            score          = sentiment["score"],
            headline_count = sentiment["headline_count"],
            positive       = sentiment["positive"],
            negative       = sentiment["negative"],
        )

    lstm_metrics_out: Optional[LSTMMetrics] = None
    if lstm_meta is not None:
        lstm_metrics_out = LSTMMetrics(
            test_mae             = lstm_meta.test_mae,
            test_rmse            = lstm_meta.test_rmse,
            directional_accuracy = lstm_meta.directional_accuracy,
            train_loss           = lstm_meta.train_loss,
            val_loss             = lstm_meta.val_loss,
            n_train              = lstm_meta.n_train,
            n_val                = lstm_meta.n_val,
            n_test               = lstm_meta.n_test,
            trained_at           = lstm_meta.trained_at,
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
        lstm_metrics      = lstm_metrics_out,
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

    df  = add_all_indicators(df)
    png = generate_chart(df, ticker)

    return StreamingResponse(
        iter([png]),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{ticker}_{period}.png"'},
    )


@app.get(
    "/model/status/{ticker}",
    response_model=ModelStatusResponse,
    summary="Cached model health: training time, test MAE, RMSE, directional accuracy",
    tags=["model"],
)
def model_status(
    ticker: str,
    period: str = Query(default="1y", description="yfinance period used when the model was trained"),
) -> ModelStatusResponse:
    """
    Returns the on-disk model's training metadata and evaluation metrics.

    - **status: not_trained** — no cached model exists; the next /analyze call will train one.
    - **status: stale** — model is older than 24 h; the next /analyze call will retrain it,
      or you can trigger an immediate retrain via `POST /model/retrain/{ticker}`.
    - **status: fresh** — model is up to date; /analyze will use cached weights.
    """
    ticker = ticker.upper().strip()
    meta   = ModelStore.load_meta(ticker, period)

    if meta is None:
        return ModelStatusResponse(ticker=ticker, period=period, status="not_trained", metrics=None)

    is_stale = ModelStore.is_stale(ticker, period)
    status   = "stale" if is_stale else "fresh"

    return ModelStatusResponse(
        ticker  = ticker,
        period  = period,
        status  = status,
        metrics = LSTMMetrics(
            test_mae             = meta.test_mae,
            test_rmse            = meta.test_rmse,
            directional_accuracy = meta.directional_accuracy,
            train_loss           = meta.train_loss,
            val_loss             = meta.val_loss,
            n_train              = meta.n_train,
            n_val                = meta.n_val,
            n_test               = meta.n_test,
            trained_at           = meta.trained_at,
        ),
    )


@app.post(
    "/model/retrain/{ticker}",
    summary="Trigger an immediate background retrain for a ticker",
    tags=["model"],
)
def trigger_retrain(
    ticker:           str,
    background_tasks: BackgroundTasks,
    period: str = Query(default="1y", description="yfinance period to retrain on"),
) -> dict:
    """
    Enqueues a retrain job for the given ticker in a background thread and
    returns immediately. Poll `GET /model/status/{ticker}` to see when
    `trained_at` updates and the status changes to **fresh**.
    """
    ticker = ticker.upper().strip()
    background_tasks.add_task(retrain_ticker, ticker, period)
    return {
        "message": f"Retraining {ticker}/{period} started in background.",
        "poll":    f"/model/status/{ticker}?period={period}",
    }
