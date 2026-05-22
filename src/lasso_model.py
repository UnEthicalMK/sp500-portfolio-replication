import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def tracking_error(portfolio_ret, index_ret):
    """Annualized tracking error: std of daily difference x sqrt(252)."""
    diff = np.array(portfolio_ret) - np.array(index_ret)
    return diff.std() * np.sqrt(252)

def naive_benchmark(train_returns, test_returns, n_stocks=40):
    stock_cols = [c for c in train_returns.columns if c != "SP500"]

    # Use average return over training period as a rough market cap proxy
    avg_ret = train_returns[stock_cols].mean().nlargest(n_stocks)
    top_stocks = avg_ret.index.tolist()

    weights = np.ones(n_stocks) / n_stocks
    portfolio_ret = (test_returns[top_stocks] * weights).sum(axis=1)
    index_ret = test_returns["SP500"]

    te = tracking_error(portfolio_ret, index_ret)
    print(f"Naive benchmark ({n_stocks} stocks): TE = {te*100:.3f}% annualized")
    return te, top_stocks

def sweep_alpha(train_returns, val_returns, n_alphas=50):
    stock_cols = [c for c in train_returns.columns if c != "SP500"]

    X_train = train_returns[stock_cols].values
    y_train = train_returns["SP500"].values
    X_val   = val_returns[stock_cols].values
    y_val   = val_returns["SP500"].values

    # Log-spaced alphas from very tight (many stocks) to very loose (few stocks)
    alphas = np.logspace(-5, -1, n_alphas)
    results = []

    for alpha in alphas:
        model = Lasso(
            alpha=alpha,
            positive=True,       # no short selling
            max_iter=10000,
            fit_intercept=False
        )
        model.fit(X_train, y_train)

        n_stocks = (model.coef_ > 1e-6).sum()
        pred = X_val @ model.coef_
        te = tracking_error(pd.Series(pred), pd.Series(y_val))

        results.append({
            "alpha":    alpha,
            "n_stocks": n_stocks,
            "val_te":   te,
            "coef":     model.coef_.copy()
        })
        print(f"alpha={alpha:.5f}  stocks={n_stocks:3d}  val_TE={te*100:.3f}%")

    return pd.DataFrame(results)

def select_best_alpha(results_df, te_threshold=0.01):
    acceptable = results_df[results_df["val_te"] < te_threshold]

    if len(acceptable) == 0:
        print("WARNING: No alpha meets the 1% threshold.")
        print("Using the alpha with minimum tracking error instead.")
        best = results_df.loc[results_df["val_te"].idxmin()]
    else:
        # Largest alpha = sparsest portfolio that still meets the threshold
        best = acceptable.loc[acceptable["alpha"].idxmax()]

    n = int(best["n_stocks"])
    print(f"\nSelected alpha:  {best['alpha']:.6f}")
    print(f"Portfolio size:  {n} stocks")
    print(f"Validation TE:   {best['val_te']*100:.3f}%")

    return best

def extract_portfolio(best_result, train_returns):
    stock_cols = [c for c in train_returns.columns if c != "SP500"]
    coef = best_result["coef"]

    selected_mask = coef > 1e-6
    selected_stocks = [stock_cols[i] for i, m in enumerate(selected_mask) if m]
    selected_weights = coef[selected_mask]

    # Normalize weights to sum to 1
    selected_weights = selected_weights / selected_weights.sum()

    portfolio = pd.DataFrame({
        "ticker": selected_stocks,
        "weight": selected_weights
    }).sort_values("weight", ascending=False)

    portfolio.to_csv("data/processed/lasso_portfolio.csv", index=False)
    print(f"\nSaved portfolio with {len(selected_stocks)} stocks")
    print(portfolio.head(10).to_string())

    return selected_stocks, selected_weights

def plot_regularization_path(results_df, best_alpha, te_threshold=0.01):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.semilogx(results_df["alpha"], results_df["val_te"] * 100,
                 color="#2196F3", linewidth=1.5)
    ax1.axhline(te_threshold * 100, color="red", linestyle="--",
                label=f"{te_threshold*100}% threshold")
    ax1.axvline(best_alpha, color="green", linestyle=":",
                label="Selected alpha")
    ax1.set_ylabel("Validation Tracking Error (% annualized)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.semilogx(results_df["alpha"], results_df["n_stocks"],
                 color="#FF9800", linewidth=1.5)
    ax2.axvline(best_alpha, color="green", linestyle=":")
    ax2.set_xlabel("Alpha (log scale)")
    ax2.set_ylabel("Number of stocks selected")
    ax2.grid(alpha=0.3)

    plt.suptitle("Lasso Regularization Path", y=1.01)
    plt.tight_layout()
    plt.savefig("results/regularization_path.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/regularization_path.png")

