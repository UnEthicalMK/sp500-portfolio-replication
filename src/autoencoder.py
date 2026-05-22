import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for learning latent market structure.

    Encodes full stock universe into latent factors, but only
    reconstructs a selected subset of stocks (sparsity constraint).
    """

    def __init__(self, n_stocks_total, selected_idx, n_factors=5):
        super().__init__()

        self.selected_idx = selected_idx
        self.n_stocks_total = n_stocks_total

        # Encoder: full market → latent factor space
        self.encoder = nn.Sequential(
            nn.Linear(n_stocks_total, 64),
            nn.ReLU(),
            nn.Linear(64, n_factors),
            nn.ReLU()
        )

        # Decoder: latent factors → selected stock space only
        self.decoder = nn.Linear(n_factors, len(selected_idx), bias=False)

    def forward(self, x):
        # Compress market information into latent factors
        z = self.encoder(x)

        # Reconstruct only selected stocks
        sparse_out = self.decoder(z)

        # Expand back to full stock space (others remain zero)
        full_out = torch.zeros(
            x.shape[0],
            self.n_stocks_total,
            device=x.device,
            dtype=x.dtype
        )

        full_out[:, self.selected_idx] = sparse_out

        return full_out, z


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
    Train sparse autoencoder to learn market latent factors.

    The model learns to reconstruct selected stocks while encoding
    global market structure in a low-dimensional factor space.
    """

    os.makedirs("models", exist_ok=True)

    # Convert selected tickers → column indices
    selected_idx = [
        all_stock_cols.index(s)
        for s in selected_stocks
        if s in all_stock_cols
    ]

    if not selected_idx:
        raise ValueError("No selected stocks found")

    # Full market feature matrix
    X_train = train_returns[all_stock_cols].values.astype(np.float32)

    # Benchmark target (index return)
    y_train = train_returns["SP500"].values.astype(np.float32)

    model = SparseAutoencoder(len(all_stock_cols), selected_idx, n_factors)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

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

            # Portfolio construction using selected stock exposures
            port_ret = (
                full_out[:, selected_idx] *
                X_batch[:, selected_idx]
            ).sum(dim=1)

            loss = loss_fn(port_ret, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Light logging (avoid cluttering output)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss {total_loss/len(loader):.8f}")

    torch.save(model.state_dict(), "models/autoencoder.pt")
    print("Saved autoencoder")

    return model, selected_idx


def plot_latent_factors(model, train_returns, all_stock_cols):
    """
    Visualize learned latent factors over time.

    Each factor represents a compressed market signal extracted
    from stock return dynamics.
    """

    os.makedirs("results", exist_ok=True)

    model.eval()

    X = torch.FloatTensor(
        train_returns[all_stock_cols].values.astype(np.float32)
    )

    with torch.no_grad():
        _, factors = model(X)

    factors = factors.numpy()
    dates = train_returns.index
    n_factors = factors.shape[1]

    fig, axes = plt.subplots(
        n_factors, 1,
        figsize=(12, 2 * n_factors),
        sharex=True
    )

    if n_factors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):

        # Each latent factor over time
        ax.plot(dates, factors[:, i], linewidth=0.7)
        ax.axhline(0, linewidth=0.5)
        ax.grid(alpha=0.2)
        ax.set_ylabel(f"F{i+1}")

    plt.tight_layout()
    plt.savefig("results/latent_factors.png", dpi=150)
    plt.show()

    print("Saved latent factors")