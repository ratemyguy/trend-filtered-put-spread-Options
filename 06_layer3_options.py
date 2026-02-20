"""
=============================================================
  QUANT STRATEGY: Trend-Filtered Put Spread Selling
  Layer 3: Options Pricing + Trade Simulation
=============================================================
  Requires: spy_with_signals.csv (output from Layer 1+2)
=============================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES PRICER
# ─────────────────────────────────────────────────────────────

def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes formula for a European Put option.

    Inputs:
      S     = current stock price
      K     = strike price
      T     = time to expiry in YEARS (e.g. 30 days = 30/365)
      r     = risk-free rate (e.g. 0.05 = 5%)
      sigma = annualized volatility (e.g. 0.20 = 20%)

    Output:
      put price in dollars per share

    The math:
      d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
      d2 = d1 - sigma * sqrt(T)
      Put = K*e^(-rT)*N(-d2) - S*N(-d1)
      where N() = cumulative normal distribution
    """
    if T <= 0 or sigma <= 0:
        # At expiry: put value = max(K - S, 0)
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return max(put_price, 0)


def black_scholes_delta(S, K, T, r, sigma):
    """
    Delta of a put option = how much the option price changes
    per $1 move in the underlying.

    Put delta is always between -1 and 0.
    -0.30 delta = 30 delta (common target for short put strikes)
    """
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1  # put delta


def historical_volatility(prices, window=30):
    """
    Estimate volatility from historical price data.
    We use 30-day rolling realized vol as our sigma input to BS.
    Annualized by multiplying daily vol by sqrt(252).
    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol


# ─────────────────────────────────────────────────────────────
# TRADE CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def construct_spread(S, sigma, r=0.05, dte=30, short_pct=0.95, long_pct=0.90):
    """
    Build a bull put spread:
      - Sell put at K1 = S * short_pct  (e.g. 95% of spot)
      - Buy  put at K2 = S * long_pct   (e.g. 90% of spot)

    WHY 95% and 90%?
      95% = ~0.25-0.30 delta = out of the money but not too far
            SPY needs to drop 5% for this to be in trouble
      90% = cheap protection, defines max loss
            The spread width = 5% of stock price

    Returns dict with all trade details.
    """
    T  = dte / 365.0
    K1 = round(S * short_pct, 2)   # short put (we sell this)
    K2 = round(S * long_pct,  2)   # long put  (we buy this)

    # Price both legs
    short_put_price = black_scholes_put(S, K1, T, r, sigma)
    long_put_price  = black_scholes_put(S, K2, T, r, sigma)

    # Net premium collected (this is our max profit)
    premium = short_put_price - long_put_price

    # Greeks on short leg
    short_delta = black_scholes_delta(S, K1, T, r, sigma)

    # Risk metrics
    spread_width = K1 - K2
    max_profit   = premium
    max_loss     = spread_width - premium
    breakeven    = K1 - premium

    return {
        "S": S, "K1": K1, "K2": K2,
        "T": T, "sigma": sigma,
        "short_put_price": round(short_put_price, 4),
        "long_put_price":  round(long_put_price,  4),
        "premium":         round(premium,         4),
        "short_delta":     round(short_delta,     4),
        "spread_width":    round(spread_width,    2),
        "max_profit":      round(max_profit,      4),
        "max_loss":        round(max_loss,        4),
        "breakeven":       round(breakeven,       2),
    }


# ─────────────────────────────────────────────────────────────
# TRADE SIMULATION (BACKTEST)
# ─────────────────────────────────────────────────────────────

def simulate_trades(df, signal_col="signal_B", r=0.05, dte=30,
                    short_pct=0.95, long_pct=0.90,
                    profit_target=0.50, stop_loss_mult=2.0):
    """
    Walk through history day by day.
    When signal = 1, open a put spread.
    Close it at:
      (a) 50% of max profit (profit target — industry standard)
      (b) 2x premium received (stop loss)
      (c) Expiration (30 DTE)

    Returns a DataFrame of all trades.
    """
    # Compute historical vol
    df = df.copy()
    df["HV30"] = historical_volatility(df["Close"], window=30)
    df.dropna(inplace=True)

    trades  = []
    in_trade = False
    trade_start = None
    trade_details = None

    for i in range(len(df)):
        row  = df.iloc[i]
        date = df.index[i]

        # ── Check exit if in a trade ──────────────────────────
        if in_trade:
            days_held = (date - trade_start).days
            S_now     = row["Close"]
            T_remaining = max((dte - days_held) / 365.0, 0)
            sigma_now   = row["HV30"]

            # Current value of the spread (what it costs to close)
            current_short = black_scholes_put(S_now, trade_details["K1"],
                                              T_remaining, r, sigma_now)
            current_long  = black_scholes_put(S_now, trade_details["K2"],
                                              T_remaining, r, sigma_now)
            current_spread_cost = current_short - current_long

            # PnL = premium collected - current cost to close
            pnl = trade_details["premium"] - current_spread_cost

            # Exit conditions
            hit_profit = pnl >= profit_target * trade_details["max_profit"]
            hit_stop   = pnl <= -stop_loss_mult * trade_details["premium"]
            expired    = days_held >= dte

            if hit_profit or hit_stop or expired:
                # At expiration use intrinsic value
                if expired:
                    short_intrinsic = max(trade_details["K1"] - S_now, 0)
                    long_intrinsic  = max(trade_details["K2"] - S_now, 0)
                    pnl = trade_details["premium"] - (short_intrinsic - long_intrinsic)

                exit_reason = ("profit_target" if hit_profit else
                               "stop_loss"     if hit_stop   else "expiry")

                trades.append({
                    "entry_date":   trade_start,
                    "exit_date":    date,
                    "days_held":    days_held,
                    "S_entry":      trade_details["S"],
                    "S_exit":       round(S_now, 2),
                    "K1":           trade_details["K1"],
                    "K2":           trade_details["K2"],
                    "premium":      trade_details["premium"],
                    "max_profit":   trade_details["max_profit"],
                    "max_loss":     trade_details["max_loss"],
                    "pnl":          round(pnl, 4),
                    "pnl_pct":      round(pnl / trade_details["max_loss"] * 100, 2),
                    "short_delta":  trade_details["short_delta"],
                    "sigma":        round(trade_details["sigma"], 4),
                    "exit_reason":  exit_reason,
                    "signal":       signal_col,
                })
                in_trade = False

        # ── Check entry if not in a trade ─────────────────────
        if not in_trade and row[signal_col] == 1:
            sigma = row["HV30"]
            if sigma > 0:
                trade_details  = construct_spread(row["Close"], sigma, r,
                                                  dte, short_pct, long_pct)
                trade_start    = date
                in_trade       = True

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────

def performance_stats(trades, label=""):
    """
    Print a clean tearsheet — this is what you show interviewers.
    Know every number here cold.
    """
    if trades.empty:
        print(f"No trades for {label}")
        return {}

    wins   = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    total_pnl    = trades["pnl"].sum()
    win_rate     = len(wins) / len(trades) * 100
    avg_win      = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss     = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 else float("inf")

    # Equity curve for Sharpe/drawdown
    equity = trades["pnl"].cumsum()
    rolling_max = equity.cummax()
    drawdown    = equity - rolling_max
    max_dd      = drawdown.min()

    # Sharpe (annualized) — assumes each trade ~ 30 days
    trades_per_year = 252 / 30
    sharpe = (trades["pnl"].mean() / trades["pnl"].std()) * np.sqrt(trades_per_year) if trades["pnl"].std() > 0 else 0

    exit_counts = trades["exit_reason"].value_counts()

    print(f"\n{'='*55}")
    print(f"  PERFORMANCE TEARSHEET — {label}")
    print(f"{'='*55}")
    print(f"  Total Trades    : {len(trades)}")
    print(f"  Win Rate        : {win_rate:.1f}%")
    print(f"  Total PnL       : ${total_pnl:.2f} per share")
    print(f"  Avg Win         : ${avg_win:.4f}")
    print(f"  Avg Loss        : ${avg_loss:.4f}")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : ${max_dd:.4f}")
    print(f"  Exit Reasons    : {exit_counts.to_dict()}")
    print(f"{'='*55}\n")

    return {
        "label": label, "n_trades": len(trades),
        "win_rate": win_rate, "total_pnl": total_pnl,
        "sharpe": sharpe, "max_dd": max_dd,
        "profit_factor": profit_factor
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_results(trades_A, trades_B, trades_C, spy_df):
    """
    4-panel results chart:
      1. SPY price
      2. Equity curves for all 3 signals (THE KEY CHART)
      3. PnL distribution
      4. Rolling win rate
    """
    BG, PANEL  = "#0f0f0f", "#1a1a1a"
    WHITE      = "#e8e8e8"
    GREEN, RED = "#00ff88", "#ff4466"
    YELLOW, BLUE, PURPLE = "#f5c518", "#4488ff", "#cc88ff"

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    # ── Panel 1: SPY Price ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(spy_df.index, spy_df["Close"], color=WHITE, lw=0.8)
    ax1.fill_between(spy_df.index, spy_df["Close"], alpha=0.1, color=BLUE)
    style_ax(ax1, "SPY Price 2010–2024")

    # ── Panel 2: Equity Curves ─────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for trades, color, label in [
        (trades_A, YELLOW, "Signal A (Price>SMA200)"),
        (trades_B, BLUE,   "Signal B (Golden Cross)"),
        (trades_C, GREEN,  "Signal C (ADX Trend)"),
    ]:
        if not trades.empty:
            equity = trades.set_index("exit_date")["pnl"].cumsum()
            ax2.plot(equity.index, equity.values, color=color, lw=1.5, label=label)
    ax2.axhline(0, color=WHITE, lw=0.5, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax2, "Cumulative PnL — All 3 Signals vs Each Other  (THIS is what proves filter value)")

    # ── Panel 3: PnL Distribution ──────────────────────────
    ax3 = fig.add_subplot(gs[2])
    for trades, color, label in [
        (trades_A, YELLOW, "A"), (trades_B, BLUE, "B"), (trades_C, GREEN, "C")
    ]:
        if not trades.empty:
            ax3.hist(trades["pnl"], bins=40, alpha=0.5, color=color,
                     label=f"Signal {label}", edgecolor="none")
    ax3.axvline(0, color=WHITE, lw=1, linestyle="--")
    ax3.legend(loc="upper right", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax3, "PnL Distribution Per Trade")

    # ── Panel 4: Rolling Win Rate (Signal B) ───────────────
    ax4 = fig.add_subplot(gs[3])
    for trades, color, label in [
        (trades_A, YELLOW, "A"), (trades_B, BLUE, "B"), (trades_C, GREEN, "C")
    ]:
        if len(trades) >= 10:
            rolling_wr = (trades["pnl"] > 0).rolling(20).mean() * 100
            ax4.plot(range(len(rolling_wr)), rolling_wr,
                     color=color, lw=1.2, label=f"Signal {label}")
    ax4.axhline(50, color=WHITE, lw=0.8, linestyle=":", alpha=0.6)
    ax4.set_ylim(0, 100)
    ax4.legend(loc="lower right", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax4, "Rolling 20-Trade Win Rate (%)")

    fig.suptitle("Trend-Filtered Bull Put Spread — Backtest Results (Layer 3)",
                 color=WHITE, fontsize=13, fontweight="bold", y=0.998)

    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[Layer 3] Chart saved → backtest_results.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load data from Layer 1+2
    print("[Layer 3] Loading signal data...")
    df = pd.read_csv("spy_with_signals.csv", index_col=0, parse_dates=True)
    print(f"[Layer 3] Loaded {len(df)} rows\n")

    # 2. Show a sample trade so you understand the math
    print("── SAMPLE TRADE CONSTRUCTION ──────────────────────")
    sample = construct_spread(S=500, sigma=0.18, r=0.05, dte=30)
    for k, v in sample.items():
        print(f"  {k:20s}: {v}")
    print()

    # 3. Run backtest for all 3 signals
    print("[Layer 3] Running backtests (this takes ~30 seconds)...")
    trades_A = simulate_trades(df, signal_col="signal_A")
    print(f"  Signal A: {len(trades_A)} trades")
    trades_B = simulate_trades(df, signal_col="signal_B")
    print(f"  Signal B: {len(trades_B)} trades")
    trades_C = simulate_trades(df, signal_col="signal_C")
    print(f"  Signal C: {len(trades_C)} trades")

    # 4. Print tearsheets
    stats_A = performance_stats(trades_A, "Signal A — Price > SMA(200)")
    stats_B = performance_stats(trades_B, "Signal B — Golden Cross")
    stats_C = performance_stats(trades_C, "Signal C — ADX Trend")

    # 5. Head-to-head comparison
    print("\n── HEAD TO HEAD COMPARISON ─────────────────────────")
    print(f"  {'Metric':<18} {'Signal A':>12} {'Signal B':>12} {'Signal C':>12}")
    print(f"  {'-'*56}")
    for key, label in [("n_trades","Trades"), ("win_rate","Win Rate %"),
                        ("sharpe","Sharpe"), ("total_pnl","Total PnL $"),
                        ("max_dd","Max DD $"), ("profit_factor","Prof. Factor")]:
        a = stats_A.get(key, 0)
        b = stats_B.get(key, 0)
        c = stats_C.get(key, 0)
        fmt = ".1f" if key in ["win_rate"] else ".2f"
        print(f"  {label:<18} {a:>12{fmt}} {b:>12{fmt}} {c:>12{fmt}}")

    # 6. Save trades
    trades_A.to_csv("trades_signal_A.csv", index=False)
    trades_B.to_csv("trades_signal_B.csv", index=False)
    trades_C.to_csv("trades_signal_C.csv", index=False)
    print("\n[Done] Trade logs saved → trades_signal_*.csv")

    # 7. Plot
    plot_results(trades_A, trades_B, trades_C, df)
