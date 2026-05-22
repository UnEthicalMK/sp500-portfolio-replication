from src.data_pipeline import (
    get_sp500_tickers, download_prices,
    compute_log_returns, build_covariance_matrix, split_data
)
from src.lasso_model import (
    sweep_alpha, select_best_alpha,
    extract_portfolio, naive_benchmark,
    plot_regularization_path
)
from src.autoencoder import train_autoencoder, plot_latent_factors
from src.backtest import (
    evaluate_model, plot_backtest, print_results_table
)

if __name__ == "__main__":

    print("\n=== PHASE 1: DATA PIPELINE ===")
    tickers = get_sp500_tickers()
    prices  = download_prices(tickers)
    returns = compute_log_returns(prices)
    cov     = build_covariance_matrix(returns)
    train, val, test = split_data(returns)

    print("\n=== PHASE 2: LASSO MODEL ===")
    benchmark_te, _ = naive_benchmark(train, test)
    results = sweep_alpha(train, val)
    best    = select_best_alpha(results)
    selected_stocks, lasso_weights = extract_portfolio(best, train)
    plot_regularization_path(results, best["alpha"])

    print("\n=== PHASE 3: AUTOENCODER ===")
    stock_cols = [c for c in train.columns if c != "SP500"]
    model, selected_idx = train_autoencoder(
        train, selected_stocks, stock_cols
    )
    plot_latent_factors(model, train, stock_cols)

    print("\n=== PHASE 4: BACKTEST ===")
    import numpy as np
    import torch

    # Lasso predictions on test set
    lasso_ret = (test[selected_stocks] * lasso_weights).sum(axis=1).values

    # Autoencoder predictions on test set
    model.eval()
    import torch
    X_test = torch.FloatTensor(test[stock_cols].values.astype("float32"))
    with torch.no_grad():
        full_out, _ = model(X_test)
    ae_weights = full_out[0, selected_idx].numpy()
    ae_weights = ae_weights / ae_weights.sum()
    ae_ret = (test[selected_stocks] * ae_weights).sum(axis=1).values

    index_ret = test["SP500"].values

    lasso_m = evaluate_model("Lasso", lasso_ret, index_ret,
                              len(selected_stocks))
    ae_m    = evaluate_model("Autoencoder", ae_ret, index_ret,
                              len(selected_stocks))

    plot_backtest(lasso_ret, ae_ret, index_ret, test.index)
    print_results_table(lasso_m, ae_m, benchmark_te)

    print("\n=== DONE ===")
    print("Check the results/ folder for all plots.")