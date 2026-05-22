import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tracking_error(portfolio_ret, index_ret):
    diff = np.array(portfolio_ret) - np.array(index_ret)
    return diff.std() * np.sqrt(252)

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def correlation_with_index(portfolio_ret, index_ret):
    return pd.Series(portfolio_ret).corr(pd.Series(index_ret))

def evaluate_model(model_name, portfolio_ret, index_ret, n_stocks):
    te   = tracking_error(portfolio_ret, index_ret)
    mdd  = max_drawdown(pd.Series(portfolio_ret))
    corr = correlation_with_index(portfolio_ret, index_ret)

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  Tracking Error (annualized): {te*100:.3f}%")
    print(f"  Max Drawdown:                {mdd*100:.2f}%")
    print(f"  Correlation with S&P 500:    {corr:.4f}")
    print(f"  Stocks used:                 {n_stocks}")

    return {"model": model_name, "TE": te, "MDD": mdd,
            "corr": corr, "n_stocks": n_stocks,
            "returns": portfolio_ret}

def plot_backtest(lasso_ret, ae_ret, index_ret, test_dates):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: cumulative returns
    (1 + pd.Series(lasso_ret, index=test_dates)).cumprod().plot(
        ax=ax1, label="Lasso Clone", color="#2196F3", linewidth=1.5)
    (1 + pd.Series(ae_ret, index=test_dates)).cumprod().plot(
        ax=ax1, label="Autoencoder Clone", color="#FF9800", linewidth=1.5)
    (1 + pd.Series(index_ret, index=test_dates)).cumprod().plot(
        ax=ax1, label="S&P 500", color="#212121", linewidth=2)
    ax1.set_ylabel("Cumulative Return (base = 1.0)")
    ax1.set_title("Out-of-sample Backtest — 2023 Test Year")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom panel: rolling 21-day tracking error
    lasso_diff = pd.Series(lasso_ret) - pd.Series(index_ret)
    ae_diff    = pd.Series(ae_ret)    - pd.Series(index_ret)
    lasso_diff.rolling(21).std().plot(
        ax=ax2, label="Lasso TE (21-day rolling)", color="#2196F3")
    ae_diff.rolling(21).std().plot(
        ax=ax2, label="AE TE (21-day rolling)", color="#FF9800")
    ax2.set_ylabel("Rolling Tracking Error")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/backtest_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/backtest_comparison.png")

def print_results_table(lasso_m, ae_m, benchmark_te):
    print(f"\n{'='*58}")
    print(f"  {'Metric':<28} {'Lasso':>12} {'Autoencoder':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Tracking Error (annualized)':<28} "
          f"{lasso_m['TE']*100:>11.3f}% {ae_m['TE']*100:>11.3f}%")
    print(f"  {'Max Drawdown':<28} "
          f"{lasso_m['MDD']*100:>11.2f}% {ae_m['MDD']*100:>11.2f}%")
    print(f"  {'Correlation with S&P 500':<28} "
          f"{lasso_m['corr']:>12.4f} {ae_m['corr']:>12.4f}")
    print(f"  {'Stocks used':<28} "
          f"{lasso_m['n_stocks']:>12} {ae_m['n_stocks']:>12}")
    print(f"  {'Naive benchmark TE':<28} {benchmark_te*100:>11.3f}%")
    print(f"{'='*58}")

    winner = "Lasso" if lasso_m["TE"] < ae_m["TE"] else "Autoencoder"
    print(f"\n  Winner (lowest tracking error): {winner}")

