import pandas as pd
import numpy as np
import yfinance as yf
import time
import requests
import os

from io import StringIO


def get_sp500_tickers():

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    table = pd.read_html(
        StringIO(response.text)
    )[0]

    tickers = (
        table["Symbol"]
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    print(f"Found {len(tickers)} tickers")

    return tickers


def download_prices(
    tickers,
    start="2014-01-01",
    end="2024-01-01"
):

    os.makedirs("data/raw", exist_ok=True)

    all_prices = {}

    print("Downloading SP500...")

    sp500 = yf.download(
        "^GSPC",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if sp500.empty:
        raise ValueError(
            "Failed to download SP500 data"
        )

    all_prices["SP500"] = (
        sp500["Close"]
        .squeeze()
    )

    total_batches = (
        len(tickers) + 49
    ) // 50

    for i in range(
        0,
        len(tickers),
        50
    ):

        batch = tickers[i:i+50]

        try:

            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if data.empty:

                print(
                    f"Skipped batch "
                    f"{i//50 + 1}"
                )

                continue

            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                if (
                    "Close"
                    not in
                    data.columns.levels[0]
                ):

                    continue

                closes = data["Close"]

            else:

                closes = data[
                    ["Close"]
                ]

            for col in closes.columns:

                series = closes[col].squeeze()

                if not hasattr(series, "ndim"):
                    continue

                if series.ndim != 1:
                    continue

                # Skip assets with too little history
                valid_fraction = series.notna().mean()

                if valid_fraction < 0.50:
                    print(
                        f"Skipping {col}"
                        f" ({valid_fraction:.1%} history)"
                    )
                    continue

                all_prices[col] = series

            print(
                f"Downloaded batch "
                f"{i//50 + 1}"
                f"/{total_batches}"
            )

            time.sleep(1)

        except Exception as e:

            print(
                f"Batch "
                f"{i//50 + 1}"
                f" failed:"
                f" {e}"
            )

    df = pd.DataFrame(
        all_prices
    )

    df = df.sort_index()

    df.to_csv(
        "data/raw/prices.csv"
    )

    print(
        f"Saved prices:"
        f" {df.shape[0]}"
        f" days ×"
        f" {df.shape[1]}"
        f" assets"
    )

    return df


def compute_log_returns(
    prices_df,
    missing_threshold=0.05
):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    missing_pct = (
        prices_df
        .isnull()
        .mean()
    )

    prices_df = prices_df.loc[
        :,
        missing_pct <
        missing_threshold
    ]

    print(
        f"Kept "
        f"{prices_df.shape[1]-1}"
        f" stocks"
    )

    prices_df = (
        prices_df
        .ffill(limit=3)
    )

    log_returns = np.log(
        prices_df /
        prices_df.shift(1)
    )

    log_returns = (
        log_returns
        .dropna()
    )

    log_returns = (
        log_returns
        .clip(
            lower=-0.10,
            upper=0.10
        )
    )

    log_returns.to_csv(
        "data/processed/log_returns.csv"
    )

    print(
        f"Log returns:"
        f" {log_returns.shape}"
    )

    return log_returns


def build_covariance_matrix(
    log_returns,
    window=252
):

    rolling_cov = (
        log_returns
        .rolling(
            window=window
        )
        .cov()
    )

    rolling_cov = (
        rolling_cov
        .dropna()
    )

    if len(
        rolling_cov
    ) == 0:

        raise ValueError(
            "No covariance matrix produced"
        )

    last_date = (
        rolling_cov
        .index
        .get_level_values(0)[-1]
    )

    cov_matrix = (
        rolling_cov
        .loc[last_date]
    )

    cov_matrix.to_csv(
        "data/processed/cov_matrix.csv"
    )

    print(
        f"Covariance matrix:"
        f" {cov_matrix.shape}"
    )

    return cov_matrix


def split_data(
    log_returns
):

    train = log_returns.loc[
        "2014-01-01":
        "2021-12-31"
    ]

    val = log_returns.loc[
        "2022-01-01":
        "2022-12-31"
    ]

    test = log_returns.loc[
        "2023-01-01":
        "2023-12-31"
    ]

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    print(
        f"Train:"
        f" {len(train)}"
    )

    print(
        f"Validation:"
        f" {len(val)}"
    )

    print(
        f"Test:"
        f" {len(test)}"
    )

    return (
        train,
        val,
        test
    )