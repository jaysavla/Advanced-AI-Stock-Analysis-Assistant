# AI Stock Analysis Assistant

An end-to-end stock analysis system that fetches real market data, computes
technical indicators (MA20, RSI), generates a baseline next-day price
prediction, and exposes everything through a clean FastAPI backend.

---

## Project Structure

```
ai_stock_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes
│   ├── data.py          # yfinance data fetching
│   ├── indicators.py    # MA20, RSI calculations
│   ├── prediction.py    # Baseline price prediction + trend direction
│   └── insights.py      # Human-readable insight generation
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend (FastAPI)

**Terminal 1:**
```bash
# Run from the ai_stock_assistant/ directory
python -m uvicorn app.main:app --reload
```

The API server will start at `http://127.0.0.1:8000`.

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### 4. Start the frontend (Streamlit dashboard)

**Terminal 2:**
```bash
# Run from the ai_stock_assistant/ directory
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`.

---

## Running Both Together

1. **Backend** (Terminal 1):
   ```bash
   .venv\Scripts\activate       # Windows
   # or: source .venv/bin/activate   # macOS/Linux
   python -m uvicorn app.main:app --reload
   ```

2. **Frontend** (Terminal 2):
   ```bash
   .venv\Scripts\activate       # Windows
   # or: source .venv/bin/activate   # macOS/Linux
   streamlit run dashboard.py
   ```

Both run simultaneously — the frontend imports from `app/` directly and can also call the backend API endpoints if configured.

---

## Example API Calls

### Health check
```bash
curl http://127.0.0.1:8000/
```

### Analyze a stock (default 1-year window)
```bash
curl http://127.0.0.1:8000/analyze/AAPL
```

### Analyze with a custom period
```bash
curl "http://127.0.0.1:8000/analyze/TSLA?period=6mo"
```

### Example response
```json
{
  "ticker": "AAPL",
  "period": "1y",
  "latest_price": 189.3,
  "predicted_price": 190.12,
  "trend": "uptrend",
  "insight": "RSI is neutral at 58.3. MA20 indicates an uptrend — bullish short-term momentum. Price is trading above its 20-day moving average (positive sign)."
}
```

---

## Supported `period` Values

| Value | Window       |
|-------|--------------|
| `1d`  | 1 day        |
| `5d`  | 5 days       |
| `1mo` | 1 month      |
| `3mo` | 3 months     |
| `6mo` | 6 months     |
| `1y`  | 1 year (default) |
| `2y`  | 2 years      |
| `5y`  | 5 years      |

---

## How It Works

| Layer | File | Description |
|---|---|---|
| Data | `data.py` | Downloads OHLCV data via `yfinance` |
| Indicators | `indicators.py` | Computes MA20 and RSI (pure pandas, no extra deps) |
| Prediction | `prediction.py` | Blends last close + MA20 + short-term momentum for a 1-day forecast |
| Insights | `insights.py` | Translates indicator values into plain-English signals |
| API | `main.py` | FastAPI routes, request validation, error handling |

---

## Future Improvements

- **ML model** — replace the baseline with a linear regression or gradient-boosted model trained on rolling features.
- **More indicators** — MACD, Bollinger Bands, VWAP, OBV.
- **Multi-ticker comparison** — `GET /compare?tickers=AAPL,MSFT,GOOGL`.
- **Chart endpoint** — `GET /chart/{ticker}` returning a base64-encoded candlestick + indicator plot.
- **Caching** — Redis or in-memory TTL cache to avoid redundant yfinance calls.
- **Authentication** — API key middleware for production use.
- **Historical accuracy tracking** — store past predictions and compare to actual closes.
