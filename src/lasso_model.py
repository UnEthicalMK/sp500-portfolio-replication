"""
lasso_model.py
--------------
Builds the Lasso sparse portfolio model.

Pipeline:
    1. Naive equal-weight benchmark (sanity-check floor)
    2. Sweep the regularization path across 50 alpha values
    3. Select the sparsest alpha whose validation TE stays under 1%
    4. Extract and normalize the selected portfolio weights
    5. Plot the regularization path
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


# ============================================================
# SHARED METRIC
# ============================================================

def tracking_error(portfolio_ret, index_ret):
    """
    Annualized tracking error between portfolio and index.

    TE = std(daily differences) x sqrt(252)

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


# ============================================================
# STEP 1: NAIVE BENCHMARK
# ============================================================

def naive_benchmark(train_returns, test_returns, n_stocks=40):
    """
    Equal-weight portfolio of top N stocks by average training return.

    Serves as a sanity-check floor. Any model that cannot beat this
    simple baseline is not functioning correctly.

    Parameters
    ----------
    train_returns : pd.DataFrame
        Training period log returns including SP500 column.
    test_returns : pd.DataFrame
        Test period log returns including SP500 column.
    n_stocks : int
        Number of stocks to include (default 40).

    Returns
    -------
    tuple
        (annualized tracking error : float,
         selected ticker list     : list of str)
    """
    stock_cols = [c for c in train_returns.columns if c != "SP500"]

    top_stocks = (
        train_returns[stock_cols]
        .mean()
        .nlargest(n_stocks)
        .index
        .tolist()
    )

    weights = np.ones(len(top_stocks)) / len(top_stocks)

    portfolio_ret = (
        test_returns[top_stocks]
        .mul(weights)
        .sum(axis=1)
    )

    te = tracking_error(portfolio_ret, test_returns["SP500"])

    print(f"Naive benchmark ({len(top_stocks)} stocks): TE = {te*100:.3f}%")
    return te, top_stocks


# ============================================================
# STEP 2: REGULARIZATION PATH SWEEP
# ============================================================

def sweep_alpha(train_returns, val_returns, n_alphas=50):
    """
    Train Lasso across a log-spaced grid of alpha values.

    For each alpha:
        - Fit Lasso on training data (positive=True: long-only)
        - Count non-zero weights (selected stocks)
        - Measure validation tracking error

    Objective: find the sparsest portfolio whose validation TE
    stays under 1% annualized. The data decides the stock count.

    Parameters
    ----------
    train_returns : pd.DataFrame
        Training period log returns.
    val_returns : pd.DataFrame
        Validation period log returns (2022 only).
    n_alphas : int
        Number of alpha values to sweep.

    Returns
    -------
    pd.DataFrame
        Results table with columns: alpha, n_stocks, val_te, coef.
    """
    stock_cols = [c for c in train_returns.columns if c != "SP500"]

    X_train = train_returns[stock_cols].values
    y_train = train_returns["SP500"].values
    X_val   = val_returns[stock_cols].values
    y_val   = val_returns["SP500"].values

    # Log-spaced: tight (many stocks) → loose (few stocks)
    alphas = np.logspace(-5, -1, n_alphas)
    results = []

    for alpha in alphas:
        model = Lasso(
            alpha=alpha,
            positive=True,          # enforce long-only weights
            fit_intercept=False,
            max_iter=10000
        )
        model.fit(X_train, y_train)

        coef = model.coef_.copy()
        n_selected = (coef > 1e-6).sum()

        pred = X_val @ coef
        te   = tracking_error(pred, y_val)

        results.append({
            "alpha":    alpha,
            "n_stocks": int(n_selected),
            "val_te":   te,
            "coef":     coef
        })

        print(
            f"  alpha={alpha:.5f}  "
            f"stocks={n_selected:3d}  "
            f"val_TE={te*100:.3f}%"
        )

    return pd.DataFrame(results)


# ============================================================
# STEP 3: ALPHA SELECTION
# ============================================================

def select_best_alpha(results_df, te_threshold=0.01):
    """
    Select the sparsest alpha whose validation TE is under threshold.

    If no alpha meets the threshold, fall back to the minimum TE model
    and warn the user.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of sweep_alpha().
    te_threshold : float
        Maximum acceptable annualized tracking error (default 1%).

    Returns
    -------
    pd.Series
        Row from results_df corresponding to the selected alpha.
    """
    acceptable = results_df[results_df["val_te"] < te_threshold]

    if acceptable.empty:
        print(
            "WARNING: No alpha met the 1% threshold. "
            "Using minimum TE model instead."
        )
        best = results_df.loc[results_df["val_te"].idxmin()]
    else:
        # Largest alpha among acceptable = sparsest acceptable model
        best = acceptable.loc[acceptable["alpha"].idxmax()]

    print(f"\nSelected alpha:  {best['alpha']:.6f}")
    print(f"Portfolio size:  {int(best['n_stocks'])} stocks")
    print(f"Validation TE:   {best['val_te']*100:.3f}%")

    return best


# ============================================================
# STEP 4: PORTFOLIO EXTRACTION
# ============================================================

def extract_portfolio(best_result, train_returns):
    """
    Extract selected tickers and normalize their weights to sum to 1.

    Saves the portfolio to data/processed/lasso_portfolio.csv.

    Parameters
    ----------
    best_result : pd.Series
        Row returned by select_best_alpha().
    train_returns : pd.DataFrame
        Training returns (used to recover column names).

    Returns
    -------
    tuple
        (selected_stocks : list of str,
         normalized_weights : np.ndarray)
    """
    os.makedirs("data/processed", exist_ok=True)

    stock_cols = [c for c in train_returns.columns if c != "SP500"]
    coef = best_result["coef"]
    mask = coef > 1e-6

    selected_stocks  = [stock_cols[i] for i, keep in enumerate(mask) if keep]
    selected_weights = coef[mask]

    if len(selected_weights) == 0:
        raise ValueError("Lasso selected zero stocks. Lower the alpha threshold.")

    # Normalize to sum to 1
    selected_weights = selected_weights / selected_weights.sum()

    portfolio = pd.DataFrame({
        "ticker": selected_stocks,
        "weight": selected_weights
    }).sort_values("weight", ascending=False)

    portfolio.to_csv("data/processed/lasso_portfolio.csv", index=False)

    print(f"Portfolio saved: {len(selected_stocks)} stocks")
    print(portfolio.head(10).to_string(index=False))

    return selected_stocks, selected_weights


# ============================================================
# STEP 5: REGULARIZATION PATH PLOT
# ============================================================

def plot_regularization_path(results_df, best_alpha, te_threshold=0.01):
    """
    Plot tracking error and portfolio size across the alpha sweep.

    Saves to results/regularization_path.png.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of sweep_alpha().
    best_alpha : float
        Selected alpha value (shown as vertical marker).
    te_threshold : float
        TE threshold line (default 1%).
    """
    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: tracking error vs alpha
    ax1.semilogx(
        results_df["alpha"],
        results_df["val_te"] * 100,
        color="#2196F3", linewidth=1.5
    )
    ax1.axhline(
        te_threshold * 100,
        color="red", linestyle="--",
        label=f"{te_threshold*100:.0f}% threshold"
    )
    ax1.axvline(
        best_alpha,
        color="green", linestyle=":",
        label="Selected alpha"
    )
    ax1.set_ylabel("Validation Tracking Error (% ann.)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: stock count vs alpha
    ax2.semilogx(
        results_df["alpha"],
        results_df["n_stocks"],
        color="#FF9800", linewidth=1.5
    )
    ax2.axvline(best_alpha, color="green", linestyle=":")
    ax2.set_xlabel("Alpha (log scale)")
    ax2.set_ylabel("Number of Stocks Selected")
    ax2.grid(alpha=0.3)

    plt.suptitle("Lasso Regularization Path", y=1.01)
    plt.tight_layout()
    plt.savefig(
        "results/regularization_path.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
    print("Saved: results/regularization_path.png")