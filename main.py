"""
main.py
-------
Full end-to-end pipeline for S&P 500 sparse portfolio replication.

Phases:
    1. Data Pipeline   — download, clean, split
    2. Lasso Model     — alpha sweep, portfolio selection
    3. Autoencoder     — train, visualize latent factors
    4. Portfolio Returns — compute gross returns for both models
    5. Evaluation      — gross + net metrics with transaction costs
    6. Results         — AQR-style table + backtest plot

Run:
    python main.py
"""

import numpy as np
import torch

from src.data_pipeline import (
    get_sp500_tickers,
    download_prices,
    compute_log_returns,
    build_covariance_matrix,
    split_data
)
from src.lasso_model import (
    naive_benchmark,
    sweep_alpha,
    select_best_alpha,
    extract_portfolio,
    plot_regularization_path
)
from src.autoencoder import (
    train_autoencoder,
    plot_latent_factors
)
from src.backtest import (
    evaluate_model,
    plot_backtest,
    print_results_table,
    tracking_error,
    max_drawdown,
    correlation_with_index
)


if __name__ == "__main__":

    # ============================================================
    # PHASE 1: DATA PIPELINE
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 1: DATA PIPELINE")
    print("=" * 50)

    tickers = get_sp500_tickers()
    prices  = download_prices(tickers)
    returns = compute_log_returns(prices)
    build_covariance_matrix(returns)
    train, val, test = split_data(returns)

    # ============================================================
    # PHASE 2: LASSO MODEL
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 2: LASSO MODEL")
    print("=" * 50)

    benchmark_te, benchmark_stocks = naive_benchmark(train, test)

    results          = sweep_alpha(train, val)
    best             = select_best_alpha(results)
    selected_stocks, lasso_weights = extract_portfolio(best, train)

    plot_regularization_path(results, best["alpha"])

    # ============================================================
    # PHASE 3: AUTOENCODER
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 3: AUTOENCODER")
    print("=" * 50)

    stock_cols = [c for c in train.columns if c != "SP500"]

    model, selected_idx = train_autoencoder(
        train_returns  = train,
        selected_stocks= selected_stocks,
        all_stock_cols = stock_cols
    )

    plot_latent_factors(model, train, stock_cols)

    # ============================================================
    # PHASE 4: PORTFOLIO RETURNS
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 4: PORTFOLIO RETURNS")
    print("=" * 50)

    # --- Lasso portfolio ---
    lasso_ret = (
        test[selected_stocks]
        .mul(lasso_weights)
        .sum(axis=1)
        .values
    )

    # --- Autoencoder inference ---
    model.eval()
    X_test = torch.FloatTensor(
        test[stock_cols].values.astype("float32")
    )

    with torch.no_grad():
        full_out, _ = model(X_test)

    # Average decoder output across test period as portfolio weights
    ae_weights = full_out[:, selected_idx].mean(dim=0).numpy()

    weight_sum = ae_weights.sum()
    if abs(weight_sum) < 1e-12:
        print("WARNING: AE weights collapsed — using uniform fallback.")
        ae_weights = np.ones(len(selected_idx)) / len(selected_idx)
    else:
        ae_weights = ae_weights / weight_sum

    # --- AE portfolio returns ---
    ae_ret = (
        test[selected_stocks]
        .mul(ae_weights)
        .sum(axis=1)
        .values
    )

    # --- Index returns ---
    index_ret = test["SP500"].values

    print(f"Lasso portfolio return computed  ({len(selected_stocks)} stocks)")
    print(f"AE portfolio return computed     ({len(selected_stocks)} stocks)")

    # ============================================================
    # PHASE 5: EVALUATION
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 5: EVALUATION")
    print("=" * 50)

    # --- Lasso metrics (gross + net) ---
    lasso_metrics = evaluate_model(
        model_name    ="Lasso",
        portfolio_ret = lasso_ret,
        index_ret     = index_ret,
        n_stocks      = len(selected_stocks),
        weights       = lasso_weights,
        commission_bps= 5,
        slippage_bps  = 5
    )

    # --- Autoencoder metrics (gross + net) ---
    ae_metrics = evaluate_model(
        model_name    ="Autoencoder",
        portfolio_ret = ae_ret,
        index_ret     = index_ret,
        n_stocks      = len(selected_stocks),
        weights       = ae_weights,
        commission_bps= 5,
        slippage_bps  = 5
    )

    # --- Naive benchmark full metrics ---
    naive_ret  = (
        test[benchmark_stocks]
        .mul(1.0 / len(benchmark_stocks))
        .sum(axis=1)
        .values
    )

    naive_corr = correlation_with_index(naive_ret, index_ret)
    naive_mdd  = max_drawdown(naive_ret)
    naive_n    = len(benchmark_stocks)

    # ============================================================
    # PHASE 6: RESULTS
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 6: RESULTS")
    print("=" * 50)

    plot_backtest(
        lasso_ret  = lasso_ret,
        ae_ret     = ae_ret,
        index_ret  = index_ret,
        test_dates = test.index
    )

    print_results_table(
        lasso_m    = lasso_metrics,
        ae_m       = ae_metrics,
        naive_te   = benchmark_te,
        naive_corr = naive_corr,
        naive_mdd  = naive_mdd,
        naive_n    = naive_n
    )

    print("=" * 50)
    print("  DONE — check results/ for all output plots")
    print("=" * 50 + "\n")