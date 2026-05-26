"""
autoencoder.py
--------------
Builds and trains the sparse autoencoder model.

Architecture:
    Encoder : ~500 stocks → Dense(64, ReLU) → Bottleneck(5, ReLU)
    Decoder : Bottleneck(5) → Sparse output (Lasso-selected stocks only)

Key design decisions:
    - 5-neuron bottleneck forces the network to compress the entire
      market into 5 latent factors
    - Output layer is masked: only Lasso-selected stocks have active
      weights; all others are hard-zeroed before loss computation
    - Loss is computed on portfolio return (not plain reconstruction)
      so the training objective is directly aligned with replication
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# MODEL DEFINITION
# ============================================================

class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for market factor compression.

    Encodes the full stock universe into a low-dimensional latent
    factor space, then reconstructs a sparse replicating portfolio
    using only the Lasso-selected stock subset.

    Parameters
    ----------
    n_stocks_total : int
        Total number of stocks in the universe (~500).
    selected_idx : list of int
        Column indices of Lasso-selected stocks.
    n_factors : int
        Bottleneck dimension (default 5).
    """

    def __init__(self, n_stocks_total, selected_idx, n_factors=5):
        super().__init__()

        self.selected_idx    = selected_idx
        self.n_stocks_total  = n_stocks_total

        # Encoder: full market → latent factor space
        self.encoder = nn.Sequential(
            nn.Linear(n_stocks_total, 64),
            nn.ReLU(),
            nn.Linear(64, n_factors),
            nn.ReLU()           # ReLU enforces non-negative factor loadings
        )

        # Decoder: latent factors → selected stocks only
        self.decoder = nn.Linear(n_factors, len(selected_idx), bias=False)

    def forward(self, x):
        # Compress market returns into latent factors
        z = self.encoder(x)

        # Decode to selected stock weights only
        sparse_out = self.decoder(z)

        # Expand back to full stock space — non-selected stocks stay zero
        full_out = torch.zeros(
            x.shape[0],
            self.n_stocks_total,
            device=x.device,
            dtype=x.dtype
        )
        full_out[:, self.selected_idx] = sparse_out

        return full_out, z


# ============================================================
# TRAINING
# ============================================================

def train_autoencoder(
    train_returns,
    selected_stocks,
    all_stock_cols,
    n_factors=5,
    epochs=200,
    lr=1e-3,
    batch_size=64
):
    """
    Train the sparse autoencoder on portfolio-return MSE loss.

    Loss is computed as MSE between the reconstructed portfolio
    return and the actual S&P 500 return — directly optimizing
    for replication quality rather than reconstruction quality.

    Parameters
    ----------
    train_returns : pd.DataFrame
        Training period log returns (stocks + SP500 column).
    selected_stocks : list of str
        Tickers selected by the Lasso model.
    all_stock_cols : list of str
        Full list of stock column names (excluding SP500).
    n_factors : int
        Bottleneck size (default 5).
    epochs : int
        Number of training epochs (default 200).
    lr : float
        Adam learning rate (default 1e-3).
    batch_size : int
        Mini-batch size (default 64).

    Returns
    -------
    tuple
        (trained model : SparseAutoencoder,
         selected_idx  : list of int)
    """
    os.makedirs("models", exist_ok=True)

    # Map tickers to column indices
    selected_idx = [
        all_stock_cols.index(s)
        for s in selected_stocks
        if s in all_stock_cols
    ]

    if not selected_idx:
        raise ValueError("No selected stocks found in all_stock_cols.")

    X_train = train_returns[all_stock_cols].values.astype(np.float32)
    y_train = train_returns["SP500"].values.astype(np.float32)

    model     = SparseAutoencoder(len(all_stock_cols), selected_idx, n_factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            full_out, _ = model(X_batch)

            # Portfolio return = sum of (output weights x stock returns)
            # computed only over the selected stocks
            port_ret = (
                full_out[:, selected_idx] *
                X_batch[:, selected_idx]
            ).sum(dim=1)

            loss = loss_fn(port_ret, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg = total_loss / len(loader)
            print(f"  Epoch {epoch+1:3d}/{epochs}  Loss: {avg:.8f}")

    torch.save(model.state_dict(), "models/autoencoder.pt")
    print("Saved: models/autoencoder.pt")

    return model, selected_idx


# ============================================================
# LATENT FACTOR VISUALIZATION
# ============================================================

def plot_latent_factors(model, train_returns, all_stock_cols):
    """
    Plot the 5 latent factor time series learned by the encoder.

    Each factor represents a compressed market signal. They often
    correspond to recognizable regimes (growth vs value, rate
    sensitivity, sector rotations, etc.).

    Saves to results/latent_factors.png.

    Parameters
    ----------
    model : SparseAutoencoder
        Trained autoencoder.
    train_returns : pd.DataFrame
        Training period returns.
    all_stock_cols : list of str
        Full list of stock column names.
    """
    os.makedirs("results", exist_ok=True)

    model.eval()

    X = torch.FloatTensor(
        train_returns[all_stock_cols].values.astype(np.float32)
    )

    with torch.no_grad():
        _, factors = model(X)

    factors    = factors.numpy()
    dates      = train_returns.index
    n_factors  = factors.shape[1]

    fig, axes = plt.subplots(
        n_factors, 1,
        figsize=(12, 2.5 * n_factors),
        sharex=True
    )

    if n_factors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(dates, factors[:, i], linewidth=0.7, color="#2196F3")
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.set_ylabel(f"Factor {i+1}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Date")
    plt.suptitle("Autoencoder: 5 Latent Market Factors (Training Period)")
    plt.tight_layout()
    plt.savefig("results/latent_factors.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/latent_factors.png")