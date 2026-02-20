"""
=============================================================
  QUANT STRATEGY: Trend-Filtered Put Spread Selling
  Layer 4: Position Sizing + Final Tearsheet
=============================================================
  Requires: trades_signal_A/B/C.csv (output from Layer 3)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# POSITION SIZING MODELS
# ─────────────────────────────────────────────────────────────

def fixed_sizing(trades, contracts=1):
    """
    Baseline: always trade 1 contract. No sizing logic.
    This is the naive approach — used as benchmark.
    """
    trades = trades.copy()
    trades["contracts"] = contracts
    trades["sized_pnl"] = trades["pnl"] * contracts
    return trades


def fixed_fractional(trades, portfolio=100000, risk_pct=0.02):
    """
    Fixed Fractional (Kelly-inspired):
    Risk a fixed % of portfolio per trade.

    risk_pct = 0.02 means risk 2% of portfolio per trade.
    Position size = (portfolio * risk_pct) / max_loss_per_contract

    WHY THIS MATTERS:
    - Protects against ruin during drawdowns
    - Scales up naturally as portfolio grows
    - Standard in professional options desks
    """
    trades  = trades.copy()
    equity  = portfolio
    records = []

    for _, row in trades.iterrows():
        risk_per_contract = abs(row["max_loss"]) * 100  # 1 contract = 100 shares
        if risk_per_contract <= 0:
            contracts = 0
        else:
            dollar_risk = equity * risk_pct
            contracts   = max(1, int(dollar_risk / risk_per_contract))

        sized_pnl = row["pnl"] * contracts * 100  # in dollars
        equity   += sized_pnl

        records.append({**row, "contracts": contracts,
                        "sized_pnl": sized_pnl, "equity": equity})

    return pd.DataFrame(records)


def vix_adjusted_sizing(trades, spy_df, portfolio=100000,
                        base_risk=0.02, vix_col="HV30"):
    """
    Volatility-Adjusted Sizing:
    Scale DOWN position size when volatility is HIGH.
    Scale UP when volatility is LOW.

    Logic: high vol = higher risk = smaller bet
           low  vol = lower  risk = larger  bet

    multiplier = base_vol / current_vol  (capped at 2x, floored at 0.25x)

    This is what separates a quant strategy from a simple one.
    """
    trades  = trades.copy()
    equity  = portfolio
    records = []

    # Get historical vol at entry date
    spy_df = spy_df.copy()
    spy_df["HV30"] = np.log(spy_df["Close"] / spy_df["Close"].shift(1)).rolling(30).std() * np.sqrt(252)
    base_vol = spy_df["HV30"].median()  # median vol as reference

    for _, row in trades.iterrows():
        entry_date = row["entry_date"]

        # Get vol at trade entry
        if entry_date in spy_df.index:
            current_vol = spy_df.loc[entry_date, "HV30"]
        else:
            current_vol = base_vol

        if pd.isna(current_vol) or current_vol <= 0:
            current_vol = base_vol

        # Vol multiplier: trade smaller in high-vol, larger in low-vol
        vol_mult = np.clip(base_vol / current_vol, 0.25, 2.0)

        risk_per_contract = abs(row["max_loss"]) * 100
        if risk_per_contract <= 0:
            contracts = 0
        else:
            dollar_risk = equity * base_risk * vol_mult
            contracts   = max(1, int(dollar_risk / risk_per_contract))

        sized_pnl = row["pnl"] * contracts * 100
        equity   += sized_pnl

        records.append({**row, "contracts": contracts,
                        "vol_mult": round(vol_mult, 2),
                        "sized_pnl": sized_pnl, "equity": equity})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(trades, label="", starting_equity=100000):
    """
    Full metrics suite. Know every single one of these for interviews.
    """
    if trades.empty or "sized_pnl" not in trades.columns:
        return {}

    pnl_col = "sized_pnl"
    pnl     = trades[pnl_col]

    # Equity curve
    if "equity" in trades.columns:
        equity = trades["equity"]
    else:
        equity = starting_equity + pnl.cumsum()

    # Returns
    total_return = (equity.iloc[-1] - starting_equity) / starting_equity * 100

    # Annualized return
    years = len(trades) / (252 / 30)
    ann_return = ((equity.iloc[-1] / starting_equity) ** (1 / max(years, 0.1)) - 1) * 100

    # Sharpe (annualized)
    trades_per_year = 252 / 30
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(trades_per_year)) if pnl.std() > 0 else 0

    # Sortino (only penalizes downside vol — more relevant for options)
    downside = pnl[pnl < 0]
    sortino  = (pnl.mean() / downside.std() * np.sqrt(trades_per_year)) if len(downside) > 1 else 0

    # Drawdown
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    # Calmar = annualized return / max drawdown (higher = better)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win stats
    wins      = pnl[pnl > 0]
    losses    = pnl[pnl <= 0]
    win_rate  = len(wins) / len(pnl) * 100
    avg_win   = wins.mean()   if len(wins)   > 0 else 0
    avg_loss  = losses.mean() if len(losses) > 0 else 0
    pf        = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    # Expectancy = avg PnL per trade
    expectancy = pnl.mean()

    metrics = {
        "label":         label,
        "n_trades":      len(trades),
        "win_rate":      round(win_rate, 1),
        "total_return":  round(total_return, 2),
        "ann_return":    round(ann_return, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "max_dd":        round(max_dd, 2),
        "calmar":        round(calmar, 3),
        "profit_factor": round(pf, 3),
        "expectancy":    round(expectancy, 2),
        "final_equity":  round(equity.iloc[-1], 2),
    }

    print(f"\n{'='*58}")
    print(f"  FINAL TEARSHEET — {label}")
    print(f"{'='*58}")
    print(f"  Trades           : {metrics['n_trades']}")
    print(f"  Win Rate         : {metrics['win_rate']}%")
    print(f"  Total Return     : {metrics['total_return']}%")
    print(f"  Ann. Return      : {metrics['ann_return']}%")
    print(f"  Sharpe Ratio     : {metrics['sharpe']}")
    print(f"  Sortino Ratio    : {metrics['sortino']}")
    print(f"  Max Drawdown     : {metrics['max_dd']}%")
    print(f"  Calmar Ratio     : {metrics['calmar']}")
    print(f"  Profit Factor    : {metrics['profit_factor']}")
    print(f"  Expectancy/Trade : ${metrics['expectancy']:.2f}")
    print(f"  Final Equity     : ${metrics['final_equity']:,.0f}")
    print(f"{'='*58}")

    return metrics


# ─────────────────────────────────────────────────────────────
# VISUALIZATION — FINAL TEARSHEET CHART
# ─────────────────────────────────────────────────────────────

def plot_tearsheet(results, spy_df):
    """
    Professional 5-panel tearsheet:
      1. Equity curves — all sizing models on Signal C
      2. Drawdown chart
      3. Monthly PnL heatmap
      4. Trade PnL distribution
      5. Position size over time (shows vol-adjusted sizing working)
    """
    BG, PANEL   = "#0f0f0f", "#1a1a1a"
    WHITE       = "#e8e8e8"
    GREEN, RED  = "#00ff88", "#ff4466"
    YELLOW, BLUE, PURPLE = "#f5c518", "#4488ff", "#cc88ff"

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    gs  = gridspec.GridSpec(5, 1, hspace=0.50)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    # ── Panel 1: Equity Curves ────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    colors  = [YELLOW, BLUE, GREEN]
    for (label, trades), color in zip(results.items(), colors):
        if "equity" in trades.columns:
            ax1.plot(pd.to_datetime(trades["exit_date"]),
                     trades["equity"], color=color, lw=1.5, label=label)
    ax1.axhline(100000, color=WHITE, lw=0.5, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    style_ax(ax1, "Equity Curve — Signal C with Different Position Sizing Models")

    # ── Panel 2: Drawdown ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for (label, trades), color in zip(results.items(), colors):
        if "equity" in trades.columns:
            eq  = trades["equity"]
            dd  = (eq - eq.cummax()) / eq.cummax() * 100
            ax2.fill_between(range(len(dd)), dd, 0,
                             alpha=0.4, color=color, label=label)
            ax2.plot(range(len(dd)), dd, color=color, lw=0.8)
    ax2.axhline(0, color=WHITE, lw=0.5)
    ax2.legend(loc="lower right", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax2, "Drawdown (%) — Vol-Adjusted Sizing Should Show Shallower Drawdowns")

    # ── Panel 3: Monthly PnL Heatmap (Vol-Adjusted) ───────
    ax3 = fig.add_subplot(gs[2])
    best_trades = list(results.values())[-1]  # vol-adjusted
    if not best_trades.empty:
        best_trades = best_trades.copy()
        best_trades["exit_date"] = pd.to_datetime(best_trades["exit_date"])
        best_trades["year"]  = best_trades["exit_date"].dt.year
        best_trades["month"] = best_trades["exit_date"].dt.month

        monthly = best_trades.groupby(["year","month"])["sized_pnl"].sum().unstack(fill_value=0)

        im = ax3.imshow(monthly.values, aspect="auto",
                        cmap="RdYlGn", vmin=-3000, vmax=3000)
        ax3.set_xticks(range(12))
        ax3.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                             color=WHITE, fontsize=8)
        ax3.set_yticks(range(len(monthly.index)))
        ax3.set_yticklabels(monthly.index.tolist(), color=WHITE, fontsize=8)
        plt.colorbar(im, ax=ax3, label="PnL ($)")
    style_ax(ax3, "Monthly PnL Heatmap — Vol-Adjusted Sizing (Green = Profit, Red = Loss)")

    # ── Panel 4: PnL Distribution ─────────────────────────
    ax4 = fig.add_subplot(gs[3])
    best_trades = list(results.values())[-1]
    if not best_trades.empty:
        pnl = best_trades["sized_pnl"]
        ax4.hist(pnl[pnl > 0], bins=40, color=GREEN, alpha=0.7,
                 label=f"Wins ({(pnl>0).sum()})", edgecolor="none")
        ax4.hist(pnl[pnl <= 0], bins=40, color=RED, alpha=0.7,
                 label=f"Losses ({(pnl<=0).sum()})", edgecolor="none")
        ax4.axvline(0, color=WHITE, lw=1)
        ax4.axvline(pnl.mean(), color=YELLOW, lw=1.5,
                    linestyle="--", label=f"Mean ${pnl.mean():.0f}")
    ax4.legend(loc="upper right", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax4, "Trade PnL Distribution — Vol-Adjusted Sizing")

    # ── Panel 5: Position Size Over Time ──────────────────
    ax5 = fig.add_subplot(gs[4])
    for (label, trades), color in zip(results.items(), colors):
        if "contracts" in trades.columns:
            ax5.plot(range(len(trades)), trades["contracts"],
                     color=color, lw=0.8, alpha=0.8, label=label)
    ax5.legend(loc="upper right", fontsize=8, facecolor=PANEL,
               labelcolor=WHITE, framealpha=0.6)
    style_ax(ax5, "Contracts Per Trade — Vol-Adjusted Model Scales Down in High-Vol Periods")

    fig.suptitle(
        "Trend-Filtered Bull Put Spread  |  Final Tearsheet  |  Signal C (ADX)  |  2010–2024",
        color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("final_tearsheet.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[Layer 4] Tearsheet saved → final_tearsheet.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load trades from Layer 3 (use Signal C — the only profitable one)
    print("[Layer 4] Loading trades...")
    trades_C = pd.read_csv("trades_signal_C.csv", parse_dates=["entry_date","exit_date"])
    spy_df   = pd.read_csv("spy_with_signals.csv", index_col=0, parse_dates=True)
    print(f"[Layer 4] {len(trades_C)} Signal C trades loaded\n")

    # 2. Apply 3 sizing models
    print("[Layer 4] Applying position sizing models...")
    t_fixed = fixed_sizing(trades_C, contracts=1)
    t_fixed["equity"] = 100000 + (t_fixed["sized_pnl"] * 100).cumsum()

    t_fractional  = fixed_fractional(trades_C,   portfolio=100000, risk_pct=0.02)
    t_vol_adj     = vix_adjusted_sizing(trades_C, spy_df, portfolio=100000, base_risk=0.02)

    results = {
        "Fixed (1 contract)":       t_fixed,
        "Fixed Fractional (2% risk)": t_fractional,
        "Vol-Adjusted Sizing":      t_vol_adj,
    }

    # 3. Print tearsheets for each
    all_metrics = {}
    for label, trades in results.items():
        m = compute_metrics(trades, label=label)
        all_metrics[label] = m

    # 4. Final comparison table
    print(f"\n{'='*70}")
    print(f"  SIZING MODEL COMPARISON — Signal C (ADX Filter)")
    print(f"{'='*70}")
    metrics_to_show = [
        ("win_rate",      "Win Rate %"),
        ("total_return",  "Total Return %"),
        ("ann_return",    "Ann. Return %"),
        ("sharpe",        "Sharpe Ratio"),
        ("sortino",       "Sortino Ratio"),
        ("max_dd",        "Max Drawdown %"),
        ("calmar",        "Calmar Ratio"),
        ("profit_factor", "Profit Factor"),
        ("final_equity",  "Final Equity $"),
    ]
    labels = list(all_metrics.keys())
    print(f"  {'Metric':<22}", end="")
    for l in labels:
        short = l.split("(")[0].strip()[:18]
        print(f"  {short:>18}", end="")
    print()
    print(f"  {'-'*68}")
    for key, display in metrics_to_show:
        print(f"  {display:<22}", end="")
        for l in labels:
            val = all_metrics[l].get(key, 0)
            if key == "final_equity":
                print(f"  {'$'+str(f'{val:,.0f}'):>18}", end="")
            else:
                print(f"  {val:>18.2f}", end="")
        print()
    print(f"{'='*70}\n")

    # 5. Plot final tearsheet
    plot_tearsheet(results, spy_df)

    print("[Done] All outputs saved:")
    print("  → final_tearsheet.png  (show this in interviews)")
    print("  → Check the Sharpe and Calmar improvement from sizing!")
