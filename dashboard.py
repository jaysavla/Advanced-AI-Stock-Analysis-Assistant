"""
dashboard.py - Streamlit UI for the AI Stock Analysis Assistant.

Run with:
    streamlit run dashboard.py
"""

import sys
import io
from pathlib import Path

# Make sure `app/` is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app.data       import get_stock_data
from app.indicators import add_all_indicators
from app.prediction import predict_price, trend_direction
from app.insights   import generate_insight
from app.plot       import generate_chart
from app.model      import lstm_predict
from app.sentiment  import analyze_sentiment, fetch_headlines

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global background ── */
.stApp { background-color: #0F1117; color: #FAFAFA; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 18px 22px;
}
div[data-testid="metric-container"] label { color: #8B949E !important; font-size: 0.82rem; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-size: 1.6rem;
    font-weight: 700;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.9rem; }

/* ── Insight box ── */
.insight-box {
    background: #161B22;
    border-left: 4px solid #2196F3;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.96rem;
    line-height: 1.65;
    color: #C9D1D9;
    margin-top: 8px;
}

/* ── Sentiment badge ── */
.badge-bullish  { background:#1a3a2a; color:#4CAF50; border:1px solid #4CAF50; }
.badge-bearish  { background:#3a1a1a; color:#EF5350; border:1px solid #EF5350; }
.badge-neutral  { background:#2a2a1a; color:#FFC107; border:1px solid #FFC107; }
.sentiment-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-weight:700; font-size:1rem; letter-spacing:.05em; margin-bottom:12px;
}

/* ── Headline pill ── */
.headline-pill {
    background:#161B22; border:1px solid #21262D; border-radius:8px;
    padding:8px 14px; margin-bottom:6px; font-size:0.88rem; color:#C9D1D9;
}
.headline-pos { border-left:3px solid #4CAF50; }
.headline-neg { border-left:3px solid #EF5350; }
.headline-neu { border-left:3px solid #FFC107; }

/* ── Section headers ── */
h3 { color:#E6EDF3 !important; font-size:1.05rem !important; margin-bottom:4px !important; }

/* ── Buttons ── */
.stButton > button {
    width:100%; background:#2196F3; color:#fff; border:none;
    border-radius:8px; padding:10px; font-size:1rem; font-weight:600;
    transition:background .2s;
}
.stButton > button:hover { background:#1769aa; }

/* ── Divider ── */
hr { border-color:#21262D !important; }

/* ── Ticker input field ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] .stTextInput input {
    color: #0F1117 !important;
    background-color: #E6EDF3 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 AI Stock Assistant")
    st.markdown("---")

    ticker = st.text_input(
        "Ticker symbol",
        value="AAPL",
        placeholder="AAPL, TSLA, MSFT …",
    ).upper().strip()

    period = st.selectbox(
        "Historical period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )

    include_sentiment = st.toggle("News sentiment analysis", value=True)
    use_lstm          = st.toggle("LSTM price prediction", value=True)

    st.markdown("---")
    run = st.button("Analyze", use_container_width=True)

    st.markdown("---")
    st.caption("Powered by yfinance · PyTorch · HuggingFace · FastAPI")


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#E6EDF3;font-size:2rem;margin-bottom:0'>📊 AI Stock Analysis Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8B949E;margin-top:4px'>Real market data · LSTM prediction · RSI & MA20 · News sentiment</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Main analysis — triggered by button or on first load with defaults
# ---------------------------------------------------------------------------
if not run:
    st.markdown(
        "<div style='text-align:center;padding:80px 0;color:#8B949E;font-size:1.1rem'>"
        "Enter a ticker in the sidebar and press <b>Analyze</b> to begin."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ── Pipeline ────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker} …"):
    try:
        df = get_stock_data(ticker, period=period)
    except ValueError as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

df = add_all_indicators(df)

# Prediction
prediction_method = "baseline"
with st.spinner("Running prediction …"):
    if use_lstm:
        try:
            predicted = lstm_predict(df)
            prediction_method = "LSTM"
        except Exception:
            predicted = predict_price(df)
    else:
        predicted = predict_price(df)

latest_price = float(df["Close"].iloc[-1])
price_delta  = predicted - latest_price
trend        = trend_direction(df)
rsi_val      = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None

# Sentiment
sentiment = None
if include_sentiment:
    with st.spinner("Analysing news sentiment …"):
        sentiment = analyze_sentiment(ticker)

insight = generate_insight(df, sentiment=sentiment)


# ---------------------------------------------------------------------------
# Row 1 — KPI metric cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Latest Close", f"${latest_price:,.2f}")

with c2:
    st.metric(
        f"Predicted ({prediction_method})",
        f"${predicted:,.2f}",
        delta=f"{price_delta:+.2f}",
        delta_color="normal",
    )

with c3:
    trend_icon = {"uptrend": "↑ Uptrend", "downtrend": "↓ Downtrend", "sideways": "→ Sideways"}
    st.metric("Trend", trend_icon.get(trend, trend))

with c4:
    if rsi_val is not None:
        rsi_label = "Overbought" if rsi_val >= 70 else ("Oversold" if rsi_val <= 30 else "Neutral")
        st.metric("RSI", f"{rsi_val:.1f}", delta=rsi_label, delta_color="off")

with c5:
    if sentiment:
        sent_icons = {"bullish": "🟢 Bullish", "bearish": "🔴 Bearish", "neutral": "🟡 Neutral"}
        st.metric("Sentiment", sent_icons.get(sentiment["label"], "—"), delta=f"{sentiment['headline_count']} headlines", delta_color="off")

st.markdown("---")


# ---------------------------------------------------------------------------
# Row 2 — Chart (full width)
# ---------------------------------------------------------------------------
st.markdown("### Price Chart")
chart_png = generate_chart(df, ticker)
st.image(chart_png, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Row 3 — Insight   |   Sentiment breakdown
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### AI Insight")
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    # Mini RSI progress bar
    if rsi_val is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### RSI Gauge")
        col_low, col_bar, col_high = st.columns([1, 8, 1])
        with col_low:
            st.markdown("<div style='color:#4CAF50;font-size:.8rem;margin-top:6px'>30</div>", unsafe_allow_html=True)
        with col_bar:
            rsi_color = "#EF5350" if rsi_val >= 70 else ("#4CAF50" if rsi_val <= 30 else "#2196F3")
            st.markdown(
                f"""
                <div style='background:#21262D;border-radius:8px;height:14px;margin-top:8px;overflow:hidden'>
                  <div style='width:{rsi_val}%;background:{rsi_color};height:100%;border-radius:8px;
                              transition:width .5s'></div>
                </div>
                <div style='text-align:center;color:{rsi_color};font-weight:700;margin-top:4px'>{rsi_val:.1f}</div>
                """,
                unsafe_allow_html=True,
            )
        with col_high:
            st.markdown("<div style='color:#EF5350;font-size:.8rem;margin-top:6px'>70</div>", unsafe_allow_html=True)

with right:
    if sentiment:
        st.markdown("### News Sentiment")

        badge_cls = f"badge-{sentiment['label']}"
        label_txt = sentiment["label"].upper()
        st.markdown(
            f'<div class="sentiment-badge {badge_cls}">{label_txt}</div>',
            unsafe_allow_html=True,
        )

        # Score bar
        score_pct = int(sentiment["score"] * 100)
        score_col = "#4CAF50" if sentiment["label"] == "bullish" else (
                    "#EF5350" if sentiment["label"] == "bearish" else "#FFC107")
        st.markdown(
            f"""
            <div style='margin-bottom:12px'>
              <div style='color:#8B949E;font-size:.8rem;margin-bottom:4px'>Sentiment score</div>
              <div style='background:#21262D;border-radius:8px;height:10px;overflow:hidden'>
                <div style='width:{score_pct}%;background:{score_col};height:100%;border-radius:8px'></div>
              </div>
              <div style='color:{score_col};font-weight:700;margin-top:4px'>{sentiment["score"]:.2f} / 1.00</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Positive / negative breakdown
        total = sentiment["headline_count"] or 1
        pos_w = int(sentiment["positive"] / total * 100)
        neg_w = int(sentiment["negative"] / total * 100)

        st.markdown(
            f"""
            <div style='display:flex;gap:8px;margin-bottom:12px'>
              <div style='flex:1;background:#1a3a2a;border:1px solid #4CAF50;border-radius:8px;
                          padding:10px;text-align:center'>
                <div style='color:#4CAF50;font-size:1.4rem;font-weight:700'>{sentiment["positive"]}</div>
                <div style='color:#8B949E;font-size:.78rem'>Positive</div>
              </div>
              <div style='flex:1;background:#3a1a1a;border:1px solid #EF5350;border-radius:8px;
                          padding:10px;text-align:center'>
                <div style='color:#EF5350;font-size:1.4rem;font-weight:700'>{sentiment["negative"]}</div>
                <div style='color:#8B949E;font-size:.78rem'>Negative</div>
              </div>
              <div style='flex:1;background:#161B22;border:1px solid #21262D;border-radius:8px;
                          padding:10px;text-align:center'>
                <div style='color:#E6EDF3;font-size:1.4rem;font-weight:700'>{sentiment["headline_count"]}</div>
                <div style='color:#8B949E;font-size:.78rem'>Total</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Row 4 — Recent headlines (expandable)
# ---------------------------------------------------------------------------
if sentiment and sentiment.get("headlines"):
    st.markdown("---")
    with st.expander(f"📰 Recent Headlines  ({len(sentiment['headlines'])} articles)", expanded=False):
        headlines = sentiment["headlines"]

        # Re-score individually to show per-headline colour
        try:
            from app.sentiment import _get_pipeline
            pipe    = _get_pipeline()
            results = pipe(headlines)
        except Exception:
            results = [{"label": "POSITIVE", "score": 0.5}] * len(headlines)

        for headline, res in zip(headlines, results):
            if res["label"] == "POSITIVE":
                css_cls = "headline-pos"
                icon    = "🟢"
            else:
                css_cls = "headline-neg"
                icon    = "🔴"

            st.markdown(
                f'<div class="headline-pill {css_cls}">{icon} {headline}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Row 5 — Raw data preview (collapsed)
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("🗂  Raw Data (last 20 rows)", expanded=False):
    display_cols = [c for c in ["Close", "MA20", "RSI"] if c in df.columns]
    st.dataframe(
        df[display_cols].tail(20).style.format("{:.2f}"),
        use_container_width=True,
    )

# Footer
st.markdown(
    "<div style='text-align:center;color:#484F58;padding:24px 0;font-size:.8rem'>"
    "AI Stock Assistant · Built with Streamlit · For educational purposes only. Not financial advice."
    "</div>",
    unsafe_allow_html=True,
)
