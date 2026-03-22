"""
plot.py - Chart generation layer.

Produces a dark-themed PNG chart with up to three panels:
  - Top    (always):     Closing price + MA20 + Bollinger Bands
  - Middle (if RSI):     RSI with overbought/oversold bands
  - Bottom (if MACD):   MACD line, signal line, and histogram
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PRICE_COLOR    = "#2196F3"   # blue
_MA_COLOR       = "#FF9800"   # amber
_BB_COLOR       = "#546E7A"   # slate  (band lines)
_BB_FILL        = "#263238"   # dark fill between bands
_RSI_COLOR      = "#9C27B0"   # purple
_OB_COLOR       = "#EF5350"   # red   (overbought)
_OS_COLOR       = "#66BB6A"   # green (oversold)
_MACD_COLOR     = "#00BCD4"   # cyan
_SIGNAL_COLOR   = "#FF9800"   # amber
_HIST_POS       = "#26A69A"   # teal  (positive histogram bars)
_HIST_NEG       = "#EF5350"   # red   (negative histogram bars)
_GRID_ALPHA     = 0.20
_BG             = "#0F1117"
_PANEL_BG       = "#0F1117"
_FIG_DPI        = 150


def generate_chart(df: pd.DataFrame, ticker: str) -> bytes:
    """
    Render a price + indicator chart and return it as PNG bytes.

    Panels rendered depend on which indicator columns are present in df:
        - MA20, BB_upper, BB_lower  → overlaid on price panel
        - RSI                       → middle panel
        - MACD, MACD_signal         → bottom panel
    """
    has_ma   = "MA20"       in df.columns
    has_bb   = ("BB_upper"  in df.columns and "BB_lower" in df.columns)
    has_rsi  = "RSI"        in df.columns
    has_macd = ("MACD"      in df.columns and "MACD_signal" in df.columns)

    # Build panel list (price is always present)
    sub_panels: list[str] = []
    if has_rsi:
        sub_panels.append("rsi")
    if has_macd:
        sub_panels.append("macd")

    n_panels      = 1 + len(sub_panels)
    height_ratios = [3] + [1] * len(sub_panels)
    fig_height    = 5 + len(sub_panels) * 2.8

    if n_panels > 1:
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(13, fig_height),
            gridspec_kw={"height_ratios": height_ratios},
            sharex=True,
        )
        ax_price = axes[0]
        ax_rsi   = axes[sub_panels.index("rsi") + 1] if "rsi"  in sub_panels else None
        ax_macd  = axes[sub_panels.index("macd") + 1] if "macd" in sub_panels else None
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=(13, 5))
        ax_rsi  = None
        ax_macd = None

    fig.patch.set_facecolor(_BG)

    # -----------------------------------------------------------------------
    # Top panel — price, MA20, Bollinger Bands
    # -----------------------------------------------------------------------
    ax_price.set_facecolor(_PANEL_BG)

    # Bollinger Bands (drawn first so price renders on top)
    if has_bb:
        ax_price.fill_between(
            df.index, df["BB_upper"], df["BB_lower"],
            alpha=0.12, color=_BB_FILL, label="_nolegend_",
        )
        ax_price.plot(
            df.index, df["BB_upper"],
            color=_BB_COLOR, linewidth=0.8, linestyle="--",
            label="BB upper", alpha=0.7, zorder=2,
        )
        ax_price.plot(
            df.index, df["BB_lower"],
            color=_BB_COLOR, linewidth=0.8, linestyle="--",
            label="BB lower", alpha=0.7, zorder=2,
        )

    # Closing price
    ax_price.plot(
        df.index, df["Close"],
        color=_PRICE_COLOR, linewidth=1.5, label="Close", zorder=4,
    )
    ax_price.fill_between(
        df.index, df["Close"], df["Close"].min(),
        alpha=0.06, color=_PRICE_COLOR,
    )

    # MA20
    if has_ma:
        ax_price.plot(
            df.index, df["MA20"],
            color=_MA_COLOR, linewidth=1.2, linestyle="--",
            label="MA20", zorder=3,
        )

    ax_price.set_title(
        f"{ticker}  —  Price & Technical Indicators",
        color="white", fontsize=14, fontweight="bold", pad=12,
    )
    ax_price.set_ylabel("Price (USD)", color="white")
    ax_price.tick_params(colors="white")
    ax_price.spines[:].set_color("#333333")
    ax_price.grid(True, alpha=_GRID_ALPHA, color="white")
    ax_price.legend(facecolor="#1A1D27", labelcolor="white", framealpha=0.8, fontsize=8)

    # -----------------------------------------------------------------------
    # Middle panel — RSI
    # -----------------------------------------------------------------------
    if has_rsi and ax_rsi is not None:
        ax_rsi.set_facecolor(_PANEL_BG)
        ax_rsi.plot(df.index, df["RSI"], color=_RSI_COLOR, linewidth=1.2, zorder=3)
        ax_rsi.axhline(70, color=_OB_COLOR, linestyle="--", linewidth=0.8, alpha=0.8)
        ax_rsi.axhline(30, color=_OS_COLOR, linestyle="--", linewidth=0.8, alpha=0.8)
        ax_rsi.axhline(50, color="white",   linestyle=":",  linewidth=0.6, alpha=0.3)
        ax_rsi.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.25, color=_OB_COLOR)
        ax_rsi.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.25, color=_OS_COLOR)
        ax_rsi.text(df.index[-1], 72, " OB", color=_OB_COLOR, fontsize=7, va="bottom")
        ax_rsi.text(df.index[-1], 28, " OS", color=_OS_COLOR, fontsize=7, va="top")
        ax_rsi.set_ylabel("RSI", color="white")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.tick_params(colors="white")
        ax_rsi.spines[:].set_color("#333333")
        ax_rsi.grid(True, alpha=_GRID_ALPHA, color="white")

    # -----------------------------------------------------------------------
    # Bottom panel — MACD
    # -----------------------------------------------------------------------
    if has_macd and ax_macd is not None:
        ax_macd.set_facecolor(_PANEL_BG)

        hist = df["MACD_hist"] if "MACD_hist" in df.columns else (df["MACD"] - df["MACD_signal"])
        colors = np.where(hist >= 0, _HIST_POS, _HIST_NEG)
        ax_macd.bar(df.index, hist, color=colors, alpha=0.6, width=1.0, label="Histogram", zorder=2)
        ax_macd.plot(df.index, df["MACD"],        color=_MACD_COLOR,   linewidth=1.2, label="MACD",   zorder=3)
        ax_macd.plot(df.index, df["MACD_signal"], color=_SIGNAL_COLOR, linewidth=1.0, label="Signal", zorder=3, linestyle="--")
        ax_macd.axhline(0, color="white", linewidth=0.5, alpha=0.4)

        ax_macd.set_ylabel("MACD", color="white")
        ax_macd.tick_params(colors="white")
        ax_macd.spines[:].set_color("#333333")
        ax_macd.grid(True, alpha=_GRID_ALPHA, color="white")
        ax_macd.legend(facecolor="#1A1D27", labelcolor="white", framealpha=0.8, fontsize=7, loc="upper left")

    # -----------------------------------------------------------------------
    # X-axis date formatting on the lowest visible panel
    # -----------------------------------------------------------------------
    bottom_ax = ax_macd or ax_rsi or ax_price
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(bottom_ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="white")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=_FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
