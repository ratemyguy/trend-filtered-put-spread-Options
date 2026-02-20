"""
=============================================================
  QUANT STRATEGY: Trend-Filtered Put Spread Selling
  Layer 1: Data Pipeline
  Layer 2: Trend Filter Signals
=============================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# LAYER 1: DATA
# ─────────────────────────────────────────────────────────────

def fetch_data(ticker="SPY", start="2010-01-01", end="2024-12-31"):
    """
    Pull daily OHLCV from Yahoo Finance.
    We use SPY (S&P 500 ETF) as the underlying for our put spreads.
    """
    print(f"[Layer 1] Fetching {ticker} data...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    # Flatten multi-level columns (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"[Layer 1] Loaded {len(df)} trading days | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────
# LAYER 2: TREND FILTERS
# ─────────────────────────────────────────────────────────────

def compute_sma_signals(df, short=50, long=200):
    """
    Filter A: Price vs SMA(200)
      signal_A = 1 if Close > SMA(200)
      → Are we above the long-term average? Simplest possible trend filter.

    Filter B: Golden Cross / Death Cross
      signal_B = 1 if SMA(50) > SMA(200)
      → The 50-day crosses above 200-day = bullish regime.
        Famous on Wall Street, actually has statistical backing.
    """
    df = df.copy()
    df["SMA_50"]   = df["Close"].rolling(short).mean()
    df["SMA_200"]  = df["Close"].rolling(long).mean()
    df["signal_A"] = np.where(df["Close"] > df["SMA_200"], 1, 0)
    df["signal_B"] = np.where(df["SMA_50"] > df["SMA_200"], 1, 0)
    return df


def compute_adx_signal(df, period=14):
    """
    Filter C: ADX (Average Directional Index) — Wilder, 1978

    WHY THIS IS BETTER than SMA filters:
    - SMA filters tell you direction but not STRENGTH
    - ADX tells you if a trend is STRONG ENOUGH to trade
    - ADX < 25 = choppy/ranging market → avoid selling spreads (whipsaw risk)
    - ADX > 25 + (+DI > -DI) = strong uptrend → sell put spreads

    This is the filter you talk about most in interviews. Understand it cold.

    Math:
    1. True Range (TR) = max of: (H-L), |H-prev_C|, |L-prev_C|
    2. +DM = today's high - yesterday's high (if positive, else 0)
    3. -DM = yesterday's low - today's low   (if positive, else 0)
    4. Smooth TR, +DM, -DM with Wilder's EMA (alpha = 1/period)
    5. +DI = 100 * smoothed(+DM) / smoothed(TR)
    6. -DI = 100 * smoothed(-DM) / smoothed(TR)
    7. DX  = 100 * |+DI - -DI| / (+DI + -DI)
    8. ADX = smoothed(DX)
    """
    df = df.copy()
    alpha = 1 / period

    H, L, C = df["High"], df["Low"], df["Close"]

    # Step 1: True Range
    tr = pd.concat([H - L,
                    (H - C.shift(1)).abs(),
                    (L - C.shift(1)).abs()], axis=1).max(axis=1)

    # Steps 2–3: Directional Movement
    dm_plus  = np.where((H - H.shift(1)) > (L.shift(1) - L),
                         np.maximum(H - H.shift(1), 0), 0)
    dm_minus = np.where((L.shift(1) - L) > (H - H.shift(1)),
                         np.maximum(L.shift(1) - L, 0), 0)

    dm_plus  = pd.Series(dm_plus,  index=df.index)
    dm_minus = pd.Series(dm_minus, index=df.index)

    # Steps 4–6: Smooth and normalize (Wilder's EMA)
    atr      = tr.ewm(alpha=alpha, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean()  / atr
    di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr

    # Steps 7–8: ADX
    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    df["DI_plus"]  = di_plus
    df["DI_minus"] = di_minus
    df["ADX"]      = adx

    # Signal: strong trend AND direction is up
    df["signal_C"] = np.where((df["ADX"] > 25) & (df["DI_plus"] > df["DI_minus"]), 1, 0)

    return df


def build_signals(df):
    """Master function: apply all filters, drop NaN warmup rows."""
    df = compute_sma_signals(df)
    df = compute_adx_signal(df)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def signal_stats(df):
    """
    Print key stats for each signal.

    Things to look for:
    - % in-market: if it's 95%, the filter adds no value (almost always in)
    - # of signal changes: too many = whipsaw, too few = laggy
    - Correlation: if A & B & C all agree 99% of the time, use just one
    """
    print("\n" + "="*58)
    print("  SIGNAL DIAGNOSTICS")
    print("="*58)
    for sig in ["signal_A", "signal_B", "signal_C"]:
        pct_in   = df[sig].mean() * 100
        switches = int(df[sig].diff().abs().sum())
        print(f"  {sig}  |  In-market: {pct_in:.1f}%  |  Signal flips: {switches}")

    print("\n  Signal Correlation:")
    print(df[["signal_A", "signal_B", "signal_C"]].corr().round(2).to_string())
    print("="*58 + "\n")


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def shade_signal(ax, df, signal_col, color, alpha=0.15):
    """Shade background green where signal=1 (we'd be selling spreads)."""
    in_region = False
    start = None
    for date, val in df[signal_col].items():
        if val == 1 and not in_region:
            start, in_region = date, True
        elif val == 0 and in_region:
            ax.axvspan(start, date, color=color, alpha=alpha)
            in_region = False
    if in_region:
        ax.axvspan(start, df.index[-1], color=color, alpha=alpha)


def style_ax(ax, title):
    """Dark theme styling."""
    PANEL, WHITE = "#1a1a1a", "#e8e8e8"
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)


def plot_all(df, ticker="SPY", save_path="trend_filters.png"):
    """
    4-panel chart showing all three trend filters on SPY.
    Green shading = signal is ON = we would sell put spreads.
    """
    GREEN, RED    = "#00ff88", "#ff4466"
    YELLOW, BLUE  = "#f5c518", "#4488ff"
    WHITE, BG     = "#e8e8e8", "#0f0f0f"

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.50)

    # ── Panel 1: Filter A — Price vs SMA(200) ──────────────────
    ax1 = fig.add_subplot(gs[0])
    shade_signal(ax1, df, "signal_A", GREEN)
    ax1.plot(df.index, df["Close"],   color=WHITE,  lw=0.8, label="SPY Close")
    ax1.plot(df.index, df["SMA_200"], color=YELLOW, lw=1.3, linestyle="--", label="SMA 200")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax1, "Filter A — Close > SMA(200)   |   Green = Sell Spreads")

    # ── Panel 2: Filter B — Golden Cross ──────────────────────
    ax2 = fig.add_subplot(gs[1])
    shade_signal(ax2, df, "signal_B", GREEN)
    ax2.plot(df.index, df["Close"],   color=WHITE,  lw=0.8, label="SPY Close")
    ax2.plot(df.index, df["SMA_50"],  color=BLUE,   lw=1.0, linestyle="--", label="SMA 50")
    ax2.plot(df.index, df["SMA_200"], color=YELLOW, lw=1.3, linestyle="--", label="SMA 200")
    ax2.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax2, "Filter B — Golden Cross (SMA50 > SMA200)   |   Green = Sell Spreads")

    # ── Panel 3: Filter C — ADX ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    shade_signal(ax3, df, "signal_C", GREEN)
    ax3.plot(df.index, df["ADX"],      color=YELLOW, lw=1.3, label="ADX")
    ax3.plot(df.index, df["DI_plus"],  color=GREEN,  lw=0.9, label="+DI")
    ax3.plot(df.index, df["DI_minus"], color=RED,    lw=0.9, label="-DI")
    ax3.axhline(25, color=WHITE, lw=0.8, linestyle=":", alpha=0.7, label="Threshold (25)")
    ax3.set_ylim(0, 65)
    ax3.legend(loc="upper right", fontsize=8, facecolor="#1a1a1a",
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax3, "Filter C — ADX Trend Strength (ADX>25 + +DI>-DI)   |   Green = Sell Spreads")

    # ── Panel 4: All 3 Signals side by side ───────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.step(df.index, df["signal_A"] * 3, color=YELLOW, lw=1.0,
             label="A: Price>SMA200", where="post")
    ax4.step(df.index, df["signal_B"] * 2, color=BLUE,   lw=1.0,
             label="B: Golden Cross", where="post")
    ax4.step(df.index, df["signal_C"] * 1, color=GREEN,  lw=1.0,
             label="C: ADX Uptrend",  where="post")
    ax4.set_yticks([1, 2, 3])
    ax4.set_yticklabels(["C", "B", "A"], color=WHITE, fontsize=9)
    ax4.set_ylim(0, 3.8)
    ax4.legend(loc="lower right", fontsize=8, facecolor="#1a1a1a",
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax4, "Signal Comparison   |   Line UP = Signal ON = Trade Active")

    fig.suptitle(f"{ticker}  —  Trend Filter Signals  (Layer 2)",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.995)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[Layer 2] Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Fetch price data
    df = fetch_data("SPY", "2010-01-01", "2024-12-31")

    # 2. Compute all three trend signals
    df = build_signals(df)

    # 3. Understand your signals before trusting them
    signal_stats(df)

    # 4. Visualize — this is what you show interviewers
    plot_all(df, ticker="SPY", save_path="trend_filters.png")

    # 5. Save enriched data for Layer 3 (options construction)
    df.to_csv("spy_with_signals.csv")
    print("[Done] Saved → spy_with_signals.csv")

    print("\nLast 5 rows:")
    cols = ["Close", "SMA_50", "SMA_200", "ADX", "DI_plus", "DI_minus",
            "signal_A", "signal_B", "signal_C"]
    print(df[cols].tail().round(2).to_string())



