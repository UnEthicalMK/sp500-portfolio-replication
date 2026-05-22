import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


def tracking_error(portfolio_ret, index_ret):
    """
    Computes annualized tracking error vs benchmark.

    Measures volatility of the difference between portfolio and index returns.
    Scaled to yearly frequency using sqrt(252).
    """
    diff = np.array(portfolio_ret) - np.array(index_ret)
    return diff.std() * np.sqrt(252)


def naive_benchmark(train_returns, test_returns, n_stocks=40):
    """
    Builds a simple equal-weight benchmark portfolio.

    Selects top stocks based on mean training returns and allocates equal weights.
    Used as a baseline to compare model performance.
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

    portfolio_ret = test_returns[top_stocks].mul(weights).sum(axis=1)

    te = tracking_error(portfolio_ret, test_returns["SP500"])

    print(f"Naive benchmark ({len(top_stocks)} stocks): TE={te*100:.3f}%")
    return te, top_stocks


def sweep_alpha(train_returns, val_returns, n_alphas=50):
    """
    Trains Lasso models over multiple regularization strengths.

    Evaluates sparsity (number of selected stocks) and tracking error
    on validation set for each alpha value.
    """

    stock_cols = [c for c in train_returns.columns if c != "SP500"]

    X_train = train_returns[stock_cols].values
    y_train = train_returns["SP500"].values
    X_val = val_returns[stock_cols].values
    y_val = val_returns["SP500"].values

    alphas = np.logspace(-5, -1, n_alphas)
    results = []

    for alpha in alphas:

        model = Lasso(
            alpha=alpha,
            positive=True,
            fit_intercept=False,
            max_iter=10000
        )

        model.fit(X_train, y_train)

        coef = model.coef_.copy()

        # Number of active (non-zero) stocks in portfolio
        n_stocks = (coef > 1e-6).sum()

        pred = X_val @ coef

        te = tracking_error(pred, y_val)

        results.append({
            "alpha": alpha,
            "n_stocks": int(n_stocks),
            "val_te": te,
            "coef": coef
        })

        print(f"alpha={alpha:.5f} stocks={n_stocks} TE={te*100:.3f}%")

    return pd.DataFrame(results)


def select_best_alpha(results_df, te_threshold=0.01):
    """
    Selects best Lasso model balancing sparsity and tracking error.

    Prefers highest alpha (sparser model) under TE constraint.
    If no model satisfies constraint, selects lowest TE model.
    """

    acceptable = results_df[results_df["val_te"] < te_threshold]

    if acceptable.empty:
        best = results_df.loc[results_df["val_te"].idxmin()]
        print("No alpha met threshold.")
    else:
        best = acceptable.loc[acceptable["alpha"].idxmax()]

    print(f"Selected alpha: {best['alpha']}")
    print(f"Stocks: {best['n_stocks']}")
    print(f"Validation TE: {best['val_te']*100:.3f}%")

    return best


def extract_portfolio(best_result, train_returns):
    """
    Converts Lasso coefficients into final normalized portfolio.

    Selects non-zero weights and normalizes them to sum to 1.
    Saves resulting portfolio as CSV.
    """

    os.makedirs("data/processed", exist_ok=True)

    stock_cols = [c for c in train_returns.columns if c != "SP500"]
    coef = best_result["coef"]

    mask = coef > 1e-6

    selected = [stock_cols[i] for i, keep in enumerate(mask) if keep]
    weights = coef[mask]

    if len(weights) == 0:
        raise ValueError("Lasso selected zero stocks")

    weights = weights / weights.sum()

    portfolio = pd.DataFrame({
        "ticker": selected,
        "weight": weights
    }).sort_values("weight", ascending=False)

    portfolio.to_csv("data/processed/lasso_portfolio.csv", index=False)

    print(f"Saved {len(selected)} stocks")
    return selected, weights


def plot_regularization_path(results_df, best_alpha, te_threshold=0.01):
    """
    Visualizes impact of regularization strength.

    Shows tradeoff between sparsity (number of stocks)
    and tracking error across different alpha values.
    """

    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.semilogx(results_df["alpha"], results_df["val_te"] * 100)
    ax1.axhline(te_threshold * 100, color="red", linestyle="--")
    ax1.axvline(best_alpha, color="green")
    ax1.set_ylabel("Tracking Error %")

    ax2.semilogx(results_df["alpha"], results_df["n_stocks"])
    ax2.axvline(best_alpha, color="green")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Stocks")

    plt.tight_layout()
    plt.savefig("results/regularization_path.png", dpi=150)
    plt.show()

    print("Saved regularization plot")