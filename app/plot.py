"""
plot.py - Chart generation layer.

Produces a two-panel PNG chart:
  - Top panel:    Closing price + MA20 overlay
  - Bottom panel: RSI with overbought/oversold bands

Call generate_chart() and pipe the returned bytes directly into a
FastAPI StreamingResponse(media_type="image/png").
"""

import io

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PRICE_COLOR  = "#2196F3"   # blue
_MA_COLOR     = "#FF9800"   # amber
_RSI_COLOR    = "#9C27B0"   # purple
_OB_COLOR     = "#EF5350"   # red  (overbought zone)
_OS_COLOR     = "#66BB6A"   # green (oversold zone)
_GRID_ALPHA   = 0.25
_FIG_DPI      = 150


def generate_chart(df: pd.DataFrame, ticker: str) -> bytes:
    """
    Render a price + indicator chart and return it as PNG bytes.

    Args:
        df:     DataFrame with 'Close', optionally 'MA20' and 'RSI' columns.
        ticker: Stock symbol used in the chart title.

    Returns:
        Raw PNG bytes suitable for an HTTP image response.
    """
    has_ma  = "MA20" in df.columns
    has_rsi = "RSI"  in df.columns

    # Layout: tall price panel + short RSI panel (only if RSI exists)
    if has_rsi:
        fig, (ax_price, ax_rsi) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=(13, 5))
        ax_rsi = None

    fig.patch.set_facecolor("#0F1117")   # dark background

    # -----------------------------------------------------------------------
    # Top panel — price + MA20
    # -----------------------------------------------------------------------
    ax_price.set_facecolor("#0F1117")

    ax_price.plot(
        df.index, df["Close"],
        color=_PRICE_COLOR, linewidth=1.5, label="Close Price", zorder=3,
    )

    if has_ma:
        ax_price.plot(
            df.index, df["MA20"],
            color=_MA_COLOR, linewidth=1.2, linestyle="--",
            label="MA20", zorder=2,
        )

    # Subtle fill under the price line
    ax_price.fill_between(
        df.index, df["Close"], df["Close"].min(),
        alpha=0.07, color=_PRICE_COLOR,
    )

    ax_price.set_title(
        f"{ticker}  —  Price & Technical Indicators",
        color="white", fontsize=14, fontweight="bold", pad=12,
    )
    ax_price.set_ylabel("Price (USD)", color="white")
    ax_price.tick_params(colors="white")
    ax_price.spines[:].set_color("#333333")
    ax_price.grid(True, alpha=_GRID_ALPHA, color="white")
    ax_price.legend(facecolor="#1A1D27", labelcolor="white", framealpha=0.8)

    # -----------------------------------------------------------------------
    # Bottom panel — RSI
    # -----------------------------------------------------------------------
    if has_rsi and ax_rsi is not None:
        ax_rsi.set_facecolor("#0F1117")

        ax_rsi.plot(df.index, df["RSI"], color=_RSI_COLOR, linewidth=1.2, zorder=3)

        # Overbought / oversold reference lines
        ax_rsi.axhline(70, color=_OB_COLOR,  linestyle="--", linewidth=0.8, alpha=0.8)
        ax_rsi.axhline(30, color=_OS_COLOR,  linestyle="--", linewidth=0.8, alpha=0.8)
        ax_rsi.axhline(50, color="white",    linestyle=":",  linewidth=0.6, alpha=0.3)

        # Shaded zones
        ax_rsi.fill_between(
            df.index, df["RSI"], 70,
            where=(df["RSI"] >= 70), alpha=0.25, color=_OB_COLOR,
        )
        ax_rsi.fill_between(
            df.index, df["RSI"], 30,
            where=(df["RSI"] <= 30), alpha=0.25, color=_OS_COLOR,
        )

        # RSI zone labels
        ax_rsi.text(df.index[-1], 72, " Overbought", color=_OB_COLOR, fontsize=7, va="bottom")
        ax_rsi.text(df.index[-1], 28, " Oversold",   color=_OS_COLOR, fontsize=7, va="top")

        ax_rsi.set_ylabel("RSI", color="white")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.tick_params(colors="white")
        ax_rsi.spines[:].set_color("#333333")
        ax_rsi.grid(True, alpha=_GRID_ALPHA, color="white")

        # X-axis date formatting on the shared bottom panel
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_rsi.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_rsi.xaxis.get_majorticklabels(), rotation=30, ha="right", color="white")
    else:
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right", color="white")

    plt.tight_layout(rect=[0, 0, 1, 1])

    # -----------------------------------------------------------------------
    # Serialize to bytes
    # -----------------------------------------------------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=_FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
