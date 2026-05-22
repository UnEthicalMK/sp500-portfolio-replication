import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def get_sp500_tickers():
    """
    Fetch S&P 500 tickers from Wikipedia.

    Scrapes the official S&P 500 constituents table and returns
    a cleaned list of ticker symbols compatible with yfinance.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    table = pd.read_html(StringIO(response.text))[0]

    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()

    print(f"Found {len(tickers)} tickers")
    return tickers


def download_prices(tickers, start="2014-01-01", end="2024-01-01"):
    """
    Download historical stock prices and S&P 500 index data.

    Fetches data in batches using yfinance, handles missing data,
    and filters out stocks with insufficient history.
    """

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
        raise ValueError("Failed to download SP500 data")

    all_prices["SP500"] = sp500["Close"].squeeze()

    batch_size = 50
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):

        batch = tickers[i:i + batch_size]

        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                print(f"Skipped batch {i // batch_size + 1}")
                continue

            # Handle single vs multi-ticker structure
            closes = (
                data["Close"]
                if isinstance(data.columns, pd.MultiIndex)
                else data[["Close"]]
            )

            for col in closes.columns:

                series = closes[col].squeeze()

                # Ensure valid 1D series
                if not hasattr(series, "ndim") or series.ndim != 1:
                    continue

                # Remove low-history assets
                if series.notna().mean() < 0.50:
                    print(f"Skipping {col} ({series.notna().mean():.1%} history)")
                    continue

                all_prices[col] = series

            print(f"Downloaded batch {i // batch_size + 1}/{total_batches}")
            time.sleep(1)

        except Exception as e:
            print(f"Batch {i // batch_size + 1} failed: {e}")

    df = pd.DataFrame(all_prices).sort_index()
    df.to_csv("data/raw/prices.csv")

    print(f"Saved prices: {df.shape[0]} days × {df.shape[1]} assets")
    return df


def compute_log_returns(prices_df, missing_threshold=0.05):
    """
    Compute log returns from price data with basic cleaning.

    Steps:
    - Remove assets with high missing values
    - Forward fill short gaps
    - Compute log returns
    - Clip extreme values for stability
    """

    os.makedirs("data/processed", exist_ok=True)

    missing_pct = prices_df.isnull().mean()
    prices_df = prices_df.loc[:, missing_pct < missing_threshold]

    print(f"Kept {prices_df.shape[1] - 1} stocks")

    prices_df = prices_df.ffill(limit=3)

    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

    # Clip extreme returns to reduce outlier impact
    log_returns = log_returns.clip(lower=-0.10, upper=0.10)

    log_returns.to_csv("data/processed/log_returns.csv")

    print(f"Log returns: {log_returns.shape}")
    return log_returns


def build_covariance_matrix(log_returns, window=252):
    """
    Compute rolling covariance matrix of returns.

    Uses a 1-year rolling window to estimate time-varying covariance,
    and returns the latest covariance snapshot.
    """

    rolling_cov = log_returns.rolling(window=window).cov().dropna()

    if rolling_cov.empty:
        raise ValueError("No covariance matrix produced")

    last_date = rolling_cov.index.get_level_values(0)[-1]
    cov_matrix = rolling_cov.loc[last_date]

    cov_matrix.to_csv("data/processed/cov_matrix.csv")

    print(f"Covariance matrix: {cov_matrix.shape}")
    return cov_matrix


def split_data(log_returns):
    """
    Split dataset into train, validation, and test sets by date.

    Designed for financial time series (no shuffling).
    """

    train = log_returns.loc["2014-01-01":"2021-12-31"]
    val = log_returns.loc["2022-01-01":"2022-12-31"]
    test = log_returns.loc["2023-01-01":"2023-12-31"]

    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

    return train, val, test