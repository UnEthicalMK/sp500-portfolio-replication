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
    print_results_table
)


if __name__ == "__main__":

    # =========================
    # PHASE 1: DATA PIPELINE
    # =========================
    print("\n=== PHASE 1 : DATA PIPELINE ===")

    tickers = get_sp500_tickers()
    prices = download_prices(tickers)
    returns = compute_log_returns(prices)
    build_covariance_matrix(returns)

    train, val, test = split_data(returns)

    # =========================
    # PHASE 2: LASSO MODEL
    # =========================
    print("\n=== PHASE 2 : LASSO ===")

    benchmark_te, _ = naive_benchmark(train, test)

    results = sweep_alpha(train, val)
    best = select_best_alpha(results)

    selected_stocks, lasso_weights = extract_portfolio(best, train)

    plot_regularization_path(results, best["alpha"])

    # =========================
    # PHASE 3: AUTOENCODER
    # =========================
    print("\n=== PHASE 3 : AUTOENCODER ===")

    stock_cols = [c for c in train.columns if c != "SP500"]

    model, selected_idx = train_autoencoder(
        train,
        selected_stocks,
        stock_cols
    )

    plot_latent_factors(model, train, stock_cols)

    # =========================
    # PHASE 4: BACKTEST
    # =========================
    print("\n=== PHASE 4 : BACKTEST ===")

    # --- Lasso portfolio returns ---
    lasso_ret = (
        test[selected_stocks] * lasso_weights
    ).sum(axis=1).values

    # --- Autoencoder inference ---
    model.eval()

    X_test = torch.FloatTensor(
        test[stock_cols].values.astype("float32")
    )

    with torch.no_grad():
        full_out, _ = model(X_test)

    # Average decoder output as proxy weights
    ae_weights = full_out[:, selected_idx].mean(dim=0).numpy()

    # Handle numerical collapse case
    weight_sum = ae_weights.sum()

    if abs(weight_sum) < 1e-12:
        print("AE weights collapsed. Using uniform weights.")
        ae_weights = np.ones(len(selected_idx)) / len(selected_idx)
    else:
        ae_weights = ae_weights / weight_sum

    # --- AE portfolio returns ---
    ae_ret = (
        test[selected_stocks] * ae_weights
    ).sum(axis=1).values

    # --- Benchmark returns ---
    index_ret = test["SP500"].values

    # =========================
    # MODEL EVALUATION
    # =========================
    lasso_metrics = evaluate_model(
        "Lasso",
        lasso_ret,
        index_ret,
        len(selected_stocks)
    )

    ae_metrics = evaluate_model(
        "Autoencoder",
        ae_ret,
        index_ret,
        len(selected_stocks)
    )

    # =========================
    # VISUALIZATION
    # =========================
    plot_backtest(
        lasso_ret,
        ae_ret,
        index_ret,
        test.index
    )

    print_results_table(
        lasso_metrics,
        ae_metrics,
        benchmark_te
    )

    print("\n=== DONE ===")
    print("Check results/")