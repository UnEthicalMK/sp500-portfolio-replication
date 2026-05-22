import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def tracking_error(
    portfolio_ret,
    index_ret
):

    diff = (
        np.asarray(portfolio_ret)
        -
        np.asarray(index_ret)
    )

    return (
        np.std(diff)
        *
        np.sqrt(252)
    )


def max_drawdown(
    returns
):

    returns = pd.Series(
        returns
    )

    cumulative = (
        1 + returns
    ).cumprod()

    rolling_max = (
        cumulative
        .cummax()
    )

    drawdown = (

        cumulative
        -
        rolling_max

    ) / (

        rolling_max
        + 1e-12

    )

    return (
        drawdown.min()
    )


def correlation_with_index(
    portfolio_ret,
    index_ret
):

    corr = pd.Series(
        portfolio_ret
    ).corr(
        pd.Series(
            index_ret
        )
    )

    if pd.isna(corr):

        return 0.0

    return corr


def evaluate_model(
    model_name,
    portfolio_ret,
    index_ret,
    n_stocks
):

    te = tracking_error(
        portfolio_ret,
        index_ret
    )

    mdd = max_drawdown(
        portfolio_ret
    )

    corr = correlation_with_index(
        portfolio_ret,
        index_ret
    )

    print(
        "\n"
        + "="*40
    )

    print(
        f"{model_name}"
    )

    print(
        "="*40
    )

    print(
        f"Tracking Error:"
        f" {te*100:.3f}%"
    )

    print(
        f"Max Drawdown:"
        f" {mdd*100:.2f}%"
    )

    print(
        f"Correlation:"
        f" {corr:.4f}"
    )

    print(
        f"Stocks:"
        f" {n_stocks}"
    )

    return {

        "model":
        model_name,

        "TE":
        te,

        "MDD":
        mdd,

        "corr":
        corr,

        "n_stocks":
        n_stocks,

        "returns":
        portfolio_ret

    }


def plot_backtest(

    lasso_ret,

    ae_ret,

    index_ret,

    test_dates

):

    os.makedirs(
        "results",
        exist_ok=True
    )

    fig, (

        ax1,

        ax2

    ) = plt.subplots(

        2,

        1,

        figsize=(12,8),

        sharex=True

    )

    lasso_curve = (

        1
        +
        pd.Series(
            lasso_ret,
            index=test_dates
        )

    ).cumprod()

    ae_curve = (

        1
        +
        pd.Series(
            ae_ret,
            index=test_dates
        )

    ).cumprod()

    index_curve = (

        1
        +
        pd.Series(
            index_ret,
            index=test_dates
        )

    ).cumprod()

    lasso_curve.plot(
        ax=ax1,
        label="Lasso"
    )

    ae_curve.plot(
        ax=ax1,
        label="Autoencoder"
    )

    index_curve.plot(
        ax=ax1,
        label="SP500"
    )

    ax1.legend()

    ax1.grid(
        alpha=0.3
    )

    ax1.set_ylabel(
        "Cumulative Return"
    )

    lasso_te = (

        pd.Series(
            lasso_ret,
            index=test_dates
        )

        -

        pd.Series(
            index_ret,
            index=test_dates
        )

    )

    ae_te = (

        pd.Series(
            ae_ret,
            index=test_dates
        )

        -

        pd.Series(
            index_ret,
            index=test_dates
        )

    )

    (

        lasso_te

        .rolling(21)

        .std()

        *

        np.sqrt(252)

    ).plot(

        ax=ax2,

        label="Lasso"

    )

    (

        ae_te

        .rolling(21)

        .std()

        *

        np.sqrt(252)

    ).plot(

        ax=ax2,

        label="Autoencoder"

    )

    ax2.legend()

    ax2.grid(
        alpha=0.3
    )

    ax2.set_ylabel(
        "Rolling TE"
    )

    plt.tight_layout()

    plt.savefig(

        "results/"
        "backtest_comparison.png",

        dpi=150

    )

    plt.show()

    print(
        "Saved backtest plot"
    )


def print_results_table(

    lasso_m,

    ae_m,

    benchmark_te

):

    print(
        "\n"
        + "="*60
    )

    print(

        f"{'Metric':<25}"

        f"{'Lasso':>15}"

        f"{'AE':>15}"

    )

    print(
        "-"*60
    )

    print(

        f"{'Tracking Error':<25}"

        f"{lasso_m['TE']*100:>14.3f}%"

        f"{ae_m['TE']*100:>14.3f}%"

    )

    print(

        f"{'Max Drawdown':<25}"

        f"{lasso_m['MDD']*100:>14.2f}%"

        f"{ae_m['MDD']*100:>14.2f}%"

    )

    print(

        f"{'Correlation':<25}"

        f"{lasso_m['corr']:>15.4f}"

        f"{ae_m['corr']:>15.4f}"

    )

    print(

        f"{'Stocks':<25}"

        f"{lasso_m['n_stocks']:>15}"

        f"{ae_m['n_stocks']:>15}"

    )

    print(
        "="*60
    )

    print(

        f"Naive benchmark TE: "

        f"{benchmark_te*100:.3f}%"

    )

    winner = (

        "Lasso"

        if

        lasso_m["TE"]

        <

        ae_m["TE"]

        else

        "Autoencoder"

    )

    print(
        f"Winner: {winner}"
    )