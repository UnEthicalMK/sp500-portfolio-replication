"""
data_pipeline.py
----------------
Handles all data acquisition, cleaning, and splitting for the
S&P 500 sparse replication project.

Pipeline:
    1. Fetch S&P 500 constituent tickers from Wikipedia
    2. Download 10 years of daily adjusted closing prices via yfinance
    3. Compute daily log returns and winsorize outliers
    4. Build a rolling covariance matrix
    5. Split data into train / validation / test sets
"""

import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# STEP 1: CONSTITUENT TICKERS
# ============================================================

def get_sp500_tickers():
    """
    Scrape current S&P 500 constituent tickers from Wikipedia.

    Note: This reflects today's index membership, not historical.
    Survivorship bias is acknowledged and documented in the README.

    Returns
    -------
    list of str
        List of ticker symbols with '.' replaced by '-' for yfinance.
    """
    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    table = pd.read_html(StringIO(response.text))[0]

    tickers = (
        table["Symbol"]
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    print(f"Found {len(tickers)} tickers")
    return tickers


# ============================================================
# STEP 2: PRICE DOWNLOAD
# ============================================================

def download_prices(tickers, start="2014-01-01", end="2024-01-01"):
    """
    Download daily adjusted closing prices for the index and all
    constituent stocks via yfinance. Downloads in batches of 50
    to avoid rate limiting.

    Parameters
    ----------
    tickers : list of str
        S&P 500 constituent tickers.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices. Columns = tickers + SP500.
    """
    os.makedirs("data/raw", exist_ok=True)

    all_prices = {}

    # Download the index
    print("Downloading S&P 500 index (^GSPC)...")
    sp500 = yf.download(
        "^GSPC",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if sp500.empty:
        raise ValueError("S&P 500 index download failed.")

    all_prices["SP500"] = sp500["Close"].squeeze()

    # Download constituents in batches
    batch_size = 50
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                print(f"  Batch {batch_num}/{total_batches} — empty, skipping.")
                continue

            closes = (
                data["Close"]
                if isinstance(data.columns, pd.MultiIndex)
                else data[["Close"]]
            )

            for col in closes.columns:
                series = closes[col].squeeze()

                if not hasattr(series, "ndim") or series.ndim != 1:
                    continue

                # Skip tickers with less than 50% data history
                if series.notna().mean() < 0.50:
                    continue

                all_prices[col] = series

            print(f"  Batch {batch_num}/{total_batches} downloaded.")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Batch {batch_num} failed: {e}")

    df = pd.DataFrame(all_prices).sort_index()
    df.to_csv("data/raw/prices.csv")
    print(f"Saved prices: {df.shape[0]} days x {df.shape[1]} assets")
    return df


# ============================================================
# STEP 3: LOG RETURNS
# ============================================================

def compute_log_returns(prices_df, missing_threshold=0.05):
    """
    Clean price data and compute daily log returns.

    Steps:
        - Drop stocks with more than 5% missing data
        - Forward-fill small gaps (max 3 consecutive days)
        - Compute log returns
        - Winsorize at 1st and 99th percentile to remove outliers

    Parameters
    ----------
    prices_df : pd.DataFrame
        Raw adjusted closing prices.
    missing_threshold : float
        Maximum allowed fraction of missing data per stock.

    Returns
    -------
    pd.DataFrame
        Cleaned daily log returns.
    """
    os.makedirs("data/processed", exist_ok=True)

    # Drop high-missing columns
    missing = prices_df.isna().mean()
    prices_df = prices_df.loc[:, missing < missing_threshold]
    print(f"Kept {prices_df.shape[1] - 1} stocks after missing data filter")

    # Forward-fill short gaps
    prices_df = prices_df.ffill(limit=3).dropna()

    # Log returns
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

    # Winsorize per column at 1st / 99th percentile
    lower = log_returns.quantile(0.01)
    upper = log_returns.quantile(0.99)
    log_returns = log_returns.clip(lower=lower, upper=upper, axis=1)

    log_returns.to_csv("data/processed/log_returns.csv")
    print(f"Log returns shape: {log_returns.shape}")
    return log_returns


# ============================================================
# STEP 4: COVARIANCE MATRIX
# ============================================================

def build_covariance_matrix(log_returns, window=252):
    """
    Build a rolling covariance matrix using a 252-day window.
    Extracts the most recent full covariance snapshot.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    window : int
        Rolling window size in trading days.

    Returns
    -------
    pd.DataFrame
        Most recent covariance matrix snapshot.
    """
    rolling_cov = log_returns.rolling(window).cov().dropna()

    if rolling_cov.empty:
        raise ValueError("Covariance matrix computation failed.")

    last_date = rolling_cov.index.get_level_values(0)[-1]
    cov = rolling_cov.loc[last_date]

    cov.to_csv("data/processed/cov_matrix.csv")
    print(f"Covariance matrix shape: {cov.shape}")
    return cov


# ============================================================
# STEP 5: TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(log_returns):
    """
    Split log returns into train, validation, and test sets.

    Split:
        Train      : 2014 – 2021  (model fitting)
        Validation : 2022          (hyperparameter tuning only)
        Test       : 2023          (final evaluation, never touched earlier)

    Parameters
    ----------
    log_returns : pd.DataFrame
        Full daily log return series.

    Returns
    -------
    tuple of pd.DataFrame
        (train, val, test)
    """
    train = log_returns.loc["2014":"2021"]
    val   = log_returns.loc["2022":"2022"]
    test  = log_returns.loc["2023":"2023"]

    if any(len(s) == 0 for s in [train, val, test]):
        raise ValueError("One or more splits are empty. Check date ranges.")

    print(f"Train:      {len(train)} days")
    print(f"Validation: {len(val)} days")
    print(f"Test:       {len(test)} days")

    return train, val, test