import pandas as pd
import numpy as np
import yfinance as yf
import time

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"Found {len(tickers)} tickers")
    return tickers

def download_prices(tickers, start="2014-01-01", end="2024-01-01"):
    all_prices = {}

    # Download the index itself
    sp500 = yf.download("^GSPC", start=start, end=end, auto_adjust=True)
    all_prices["SP500"] = sp500["Close"]

    # Download stocks in batches of 50 to avoid rate limiting
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        data = yf.download(batch, start=start, end=end,
                           auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]]
        all_prices.update(closes.to_dict(orient="series"))
        time.sleep(1)
        print(f"Downloaded batch {i//50 + 1} of {len(tickers)//50 + 1}")

    df = pd.DataFrame(all_prices)
    df.to_csv("data/raw/prices.csv")
    print(f"Saved prices: {df.shape[0]} days x {df.shape[1]} assets")
    return df

def compute_log_returns(prices_df, missing_threshold=0.05):
    # Drop any stock with more than 5% missing data
    missing_pct = prices_df.isnull().mean()
    prices_df = prices_df.loc[:, missing_pct < missing_threshold]
    print(f"Kept {prices_df.shape[1] - 1} stocks after missing data filter")

    # Forward-fill small gaps up to 3 consecutive days
    prices_df = prices_df.ffill(limit=3)

    # Compute log returns
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

    # Winsorize: clip extreme daily moves at ±10%
    # This prevents flash-crash days from dominating the covariance matrix
    log_returns = log_returns.clip(lower=-0.10, upper=0.10)

    log_returns.to_csv("data/processed/log_returns.csv")
    print(f"Log returns shape: {log_returns.shape}")
    return log_returns

def build_covariance_matrix(log_returns, window=252):
    # 252 trading days = 1 year rolling window
    rolling_cov = log_returns.rolling(window=window).cov()

    # Extract the most recent full covariance matrix
    last_date = log_returns.index[-1]
    cov_matrix = rolling_cov.loc[last_date]

    print(f"Covariance matrix shape: {cov_matrix.shape}")
    cov_matrix.to_csv("data/processed/cov_matrix.csv")
    return cov_matrix

def split_data(log_returns):
    train = log_returns.loc["2014-01-01":"2021-12-31"]  # 8 years, model fitting
    val   = log_returns.loc["2022-01-01":"2022-12-31"]  # 1 year, alpha tuning
    test  = log_returns.loc["2023-01-01":"2023-12-31"]  # 1 year, final evaluation

    print(f"Train: {len(train)} days")
    print(f"Val:   {len(val)} days")
    print(f"Test:  {len(test)} days")
    return train, val, test

