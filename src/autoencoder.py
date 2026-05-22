import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class SparseAutoencoder(nn.Module):
    def __init__(self, n_stocks_total, selected_idx, n_factors=5):
        super().__init__()
        self.selected_idx = selected_idx
        self.n_stocks_total = n_stocks_total
        n_selected = len(selected_idx)

        # Encoder: compresses all 500 stocks down to 5 latent factors
        self.encoder = nn.Sequential(
            nn.Linear(n_stocks_total, 64),
            nn.ReLU(),
            nn.Linear(64, n_factors),
            nn.ReLU()   # ReLU enforces non-negative factor loadings
        )

        # Decoder: maps 5 factors to weights for the selected stocks only
        self.decoder = nn.Linear(n_factors, n_selected, bias=False)

    def forward(self, x):
        z = self.encoder(x)           # latent factors: shape (batch, 5)
        out_sparse = self.decoder(z)  # weights for selected stocks only

        # Reconstruct full output, zeroing out non-selected stocks
        full_out = torch.zeros(x.shape[0], self.n_stocks_total,
                               device=x.device)
        full_out[:, self.selected_idx] = out_sparse
        return full_out, z
    
def train_autoencoder(train_returns, selected_stocks, all_stock_cols,
                      n_factors=5, epochs=200, lr=1e-3, batch_size=64):

    selected_idx = [all_stock_cols.index(s) for s in selected_stocks]
    n_stocks_total = len(all_stock_cols)

    X_train = train_returns[all_stock_cols].values.astype(np.float32)
    y_train = train_returns["SP500"].values.astype(np.float32)

    model = SparseAutoencoder(n_stocks_total, selected_idx, n_factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train),
                            torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            full_out, _ = model(X_batch)

            # Portfolio return = dot product of output weights and stock returns
            # We only use the selected stocks in both
            port_ret = (full_out[:, selected_idx] *
                        X_batch[:, selected_idx]).sum(dim=1)

            loss = loss_fn(port_ret, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {avg_loss:.8f}")

    torch.save(model.state_dict(), "models/autoencoder.pt")
    print("Saved: models/autoencoder.pt")
    return model, selected_idx

def plot_latent_factors(model, train_returns, all_stock_cols):
    model.eval()
    X = torch.FloatTensor(
        train_returns[all_stock_cols].values.astype(np.float32)
    )
    with torch.no_grad():
        _, factors = model(X)

    factors = factors.numpy()
    dates = train_returns.index

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(dates, factors[:, i], linewidth=0.7, color="#2196F3")
        ax.set_ylabel(f"Factor {i+1}", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Date")
    plt.suptitle("Autoencoder: 5 Latent Market Factors (Training Period)")
    plt.tight_layout()
    plt.savefig("results/latent_factors.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/latent_factors.png")