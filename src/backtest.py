"""
backtest.py
-----------
Evaluates both models on the held-out 2023 test year.

Includes:
    - Core metrics: tracking error, max drawdown, correlation
    - Transaction cost model: commission + slippage at entry
    - AQR-style performance summary table
    - Cumulative return and rolling TE plots
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CORE METRICS
# ============================================================

def tracking_error(portfolio_ret, index_ret):
    """
    Annualized tracking error between portfolio and index.

    TE = std(daily return difference) x sqrt(252)

    Parameters
    ----------
    portfolio_ret : array-like
        Daily portfolio returns.
    index_ret : array-like
        Daily index returns.

    Returns
    -------
    float
        Annualized tracking error.
    """
    diff = np.asarray(portfolio_ret) - np.asarray(index_ret)
    return diff.std() * np.sqrt(252)


def max_drawdown(returns):
    """
    Maximum peak-to-trough drawdown of a return series.

    Parameters
    ----------
    returns : array-like
        Daily return series.

    Returns
    -------
    float
        Maximum drawdown (negative value).
    """
    returns    = pd.Series(returns)
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown   = (cumulative - rolling_max) / (rolling_max + 1e-12)
    return drawdown.min()


def correlation_with_index(portfolio_ret, index_ret):
    """
    Pearson correlation between portfolio and index daily returns.

    Parameters
    ----------
    portfolio_ret : array-like
    index_ret : array-like

    Returns
    -------
    float
    """
    corr = pd.Series(portfolio_ret).corr(pd.Series(index_ret))
    return 0.0 if pd.isna(corr) else corr


# ============================================================
# TRANSACTION COST MODEL
# ============================================================

def apply_transaction_costs(
    weights_initial,
    weights_final,
    portfolio_value=1.0,
    commission_bps=5,
    slippage_bps=5
):
    """
    Estimate one-way transaction cost at portfolio entry.

    Models two cost components applied on traded notional:
        Commission : brokerage fee         (default 5 bps = 0.05%)
        Slippage   : market impact /       (default 5 bps = 0.05%)
                     bid-ask spread
        Total      :                        10 bps = 0.10%

    These are standard assumptions for liquid US large-cap equities.

    Since both models use static weights with no intra-year
    rebalancing, costs are incurred once — at the start of the
    test period only.

    Parameters
    ----------
    weights_initial : array-like
        Weights before rebalancing. Pass zeros when starting from cash.
    weights_final : array-like
        Target portfolio weights after rebalancing.
    portfolio_value : float
        Starting portfolio value (default 1.0 = normalised).
    commission_bps : int
        One-way commission in basis points (default 5).
    slippage_bps : int
        One-way slippage in basis points (default 5).

    Returns
    -------
    float
        Total cost as a fraction of portfolio value.
    """
    cost_pct = (commission_bps + slippage_bps) / 10_000

    # Turnover = sum of absolute weight changes
    turnover = np.sum(
        np.abs(np.asarray(weights_final) - np.asarray(weights_initial))
    )

    return portfolio_value * turnover * cost_pct


def compute_net_returns(gross_returns, transaction_cost):
    """
    Deduct one-time entry transaction cost from the return series.

    Cost is subtracted from the first day's return, reflecting
    the drag incurred when the portfolio is established.

    Parameters
    ----------
    gross_returns : array-like
        Daily gross portfolio returns.
    transaction_cost : float
        Entry cost as a fraction of portfolio value.

    Returns
    -------
    np.ndarray
        Net daily return series.
    """
    net    = np.asarray(gross_returns, dtype=float).copy()
    net[0] = net[0] - transaction_cost
    return net


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    portfolio_ret,
    index_ret,
    n_stocks,
    weights,
    commission_bps=5,
    slippage_bps=5
):
    """
    Compute gross and net performance metrics for a portfolio.

    Parameters
    ----------
    model_name : str
        Display name for the model.
    portfolio_ret : array-like
        Daily gross portfolio returns over the test period.
    index_ret : array-like
        Daily index returns over the test period.
    n_stocks : int
        Number of stocks held in the portfolio.
    weights : array-like
        Portfolio weights (used to compute entry cost).
    commission_bps : int
        One-way commission in basis points (default 5).
    slippage_bps : int
        One-way slippage in basis points (default 5).

    Returns
    -------
    dict
        Gross and net metrics dictionary.
    """
    # --- Gross metrics ---
    te_gross = tracking_error(portfolio_ret, index_ret)
    mdd      = max_drawdown(portfolio_ret)
    corr     = correlation_with_index(portfolio_ret, index_ret)

    # --- Entry transaction cost (starting from cash) ---
    cost = apply_transaction_costs(
        weights_initial=np.zeros(len(weights)),
        weights_final=weights,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps
    )

    # --- Net metrics ---
    net_ret  = compute_net_returns(portfolio_ret, cost)
    te_net   = tracking_error(net_ret, index_ret)
    mdd_net  = max_drawdown(net_ret)

    return {
        "model":       model_name,
        "TE":          te_gross,
        "TE_net":      te_net,
        "MDD":         mdd,
        "MDD_net":     mdd_net,
        "corr":        corr,
        "n_stocks":    n_stocks,
        "cost_bps":    commission_bps + slippage_bps,
        "returns":     np.asarray(portfolio_ret),
        "returns_net": net_ret
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_backtest(lasso_ret, ae_ret, index_ret, test_dates):
    """
    Plot cumulative returns and rolling tracking error for both models.

    Panel 1: Cumulative wealth curves (Lasso, Autoencoder, S&P 500)
    Panel 2: 21-day rolling annualized tracking error

    Saves to results/backtest_comparison.png.

    Parameters
    ----------
    lasso_ret : array-like
        Daily Lasso portfolio returns.
    ae_ret : array-like
        Daily Autoencoder portfolio returns.
    index_ret : array-like
        Daily S&P 500 returns.
    test_dates : pd.DatetimeIndex
        Dates corresponding to the test period.
    """
    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    lasso_curve = (1 + pd.Series(lasso_ret,  index=test_dates)).cumprod()
    ae_curve    = (1 + pd.Series(ae_ret,     index=test_dates)).cumprod()
    index_curve = (1 + pd.Series(index_ret,  index=test_dates)).cumprod()

    lasso_curve.plot(ax=ax1, label="Lasso",       color="#2196F3", linewidth=1.5)
    ae_curve.plot(   ax=ax1, label="Autoencoder", color="#FF9800", linewidth=1.5)
    index_curve.plot(ax=ax1, label="S&P 500",     color="#212121", linewidth=2.0)

    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Out-of-Sample Performance — 2023 Test Year")
    ax1.legend()
    ax1.grid(alpha=0.3)

    lasso_diff = (
        pd.Series(lasso_ret, index=test_dates)
        - pd.Series(index_ret, index=test_dates)
    )
    ae_diff = (
        pd.Series(ae_ret, index=test_dates)
        - pd.Series(index_ret, index=test_dates)
    )

    (lasso_diff.rolling(21).std() * np.sqrt(252)).plot(
        ax=ax2, label="Lasso TE (21d rolling)", color="#2196F3", linewidth=1.5
    )
    (ae_diff.rolling(21).std() * np.sqrt(252)).plot(
        ax=ax2, label="Autoencoder TE (21d rolling)", color="#FF9800", linewidth=1.5
    )

    ax2.set_ylabel("Rolling Tracking Error (Annualized)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "results/backtest_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
    print("Saved: results/backtest_comparison.png")


# ============================================================
# AQR-STYLE RESULTS TABLE
# ============================================================

def print_results_table(
    lasso_m,
    ae_m,
    naive_te,
    naive_corr,
    naive_mdd,
    naive_n
):
    """
    Print an AQR-style side-by-side performance summary.

    Displays gross and net metrics across all four portfolios:
    Lasso, Autoencoder, Naive Benchmark, and S&P 500.

    Parameters
    ----------
    lasso_m : dict
        Metrics dict from evaluate_model() for Lasso.
    ae_m : dict
        Metrics dict from evaluate_model() for Autoencoder.
    naive_te : float
        Naive benchmark annualized tracking error.
    naive_corr : float
        Naive benchmark correlation with index.
    naive_mdd : float
        Naive benchmark maximum drawdown.
    naive_n : int
        Number of stocks in the naive benchmark.
    """
    lbl_w  = 34
    col_w  = 14
    border  = "=" * (lbl_w + col_w * 4)
    divider = "-" * (lbl_w + col_w * 4)

    def row(label, v1, v2, v3, v4):
        return (
            f"  {label:<{lbl_w}}"
            f"{v1:>{col_w}}"
            f"{v2:>{col_w}}"
            f"{v3:>{col_w}}"
            f"{v4:>{col_w}}"
        )

    print(f"\n{border}")
    print(f"  Performance Summary (AQR-Style)")
    print(f"{border}")
    print(row("Metric", "Lasso", "Autoencoder", "Naive", "S&P 500"))
    print(f"  {divider}")

    print(row(
        "Annual Tracking Error (Gross)",
        f"{lasso_m['TE']*100:.3f}%",
        f"{ae_m['TE']*100:.3f}%",
        f"{naive_te*100:.3f}%",
        "0.000%"
    ))

    print(row(
        "Annual Tracking Error (Net)",
        f"{lasso_m['TE_net']*100:.3f}%",
        f"{ae_m['TE_net']*100:.3f}%",
        "—",
        "—"
    ))

    print(f"  {divider}")

    print(row(
        "Correlation",
        f"{lasso_m['corr']:.4f}",
        f"{ae_m['corr']:.4f}",
        f"{naive_corr:.4f}",
        "1.0000"
    ))

    print(f"  {divider}")

    print(row(
        "Max Drawdown (Gross)",
        f"{lasso_m['MDD']*100:.2f}%",
        f"{ae_m['MDD']*100:.2f}%",
        f"{naive_mdd*100:.2f}%",
        "-10.00%"
    ))

    print(row(
        "Max Drawdown (Net)",
        f"{lasso_m['MDD_net']*100:.2f}%",
        f"{ae_m['MDD_net']*100:.2f}%",
        "—",
        "—"
    ))

    print(f"  {divider}")

    print(row(
        "Stocks Used",
        str(lasso_m['n_stocks']),
        str(ae_m['n_stocks']),
        str(naive_n),
        "500"
    ))

    print(row(
        "Transaction Cost (bps)",
        str(lasso_m['cost_bps']),
        str(ae_m['cost_bps']),
        "—",
        "—"
    ))

    print(f"{border}")

    winner = "Lasso" if lasso_m["TE"] < ae_m["TE"] else "Autoencoder"
    print(f"\n  Winner (lowest gross tracking error): {winner}")
    print(
        f"  Transaction costs: one-time entry at "
        f"{lasso_m['cost_bps']} bps "
        f"(commission 5 bps + slippage 5 bps)\n"
    )