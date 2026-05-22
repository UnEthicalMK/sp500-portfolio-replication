import os

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


class SparseAutoencoder(
    nn.Module
):

    def __init__(
        self,
        n_stocks_total,
        selected_idx,
        n_factors=5
    ):

        super().__init__()

        self.selected_idx = selected_idx
        self.n_stocks_total = n_stocks_total

        n_selected = len(
            selected_idx
        )

        self.encoder = nn.Sequential(

            nn.Linear(
                n_stocks_total,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                n_factors
            ),

            nn.ReLU()

        )

        self.decoder = nn.Linear(
            n_factors,
            n_selected,
            bias=False
        )

    def forward(
        self,
        x
    ):

        z = self.encoder(x)

        sparse_out = self.decoder(z)

        full_out = torch.zeros(

            x.shape[0],

            self.n_stocks_total,

            device=x.device,

            dtype=x.dtype

        )

        full_out[
            :,
            self.selected_idx
        ] = sparse_out

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

    os.makedirs(
        "models",
        exist_ok=True
    )

    selected_idx = [

        all_stock_cols.index(s)

        for s in selected_stocks

        if s in all_stock_cols

    ]

    if len(selected_idx) == 0:

        raise ValueError(
            "No selected stocks found"
        )

    X_train = (

        train_returns[
            all_stock_cols
        ]

        .values

        .astype(np.float32)

    )

    y_train = (

        train_returns[
            "SP500"
        ]

        .values

        .astype(np.float32)

    )

    model = SparseAutoencoder(

        len(all_stock_cols),

        selected_idx,

        n_factors

    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=lr,

        weight_decay=1e-4

    )

    loss_fn = nn.MSELoss()

    dataset = TensorDataset(

        torch.FloatTensor(
            X_train
        ),

        torch.FloatTensor(
            y_train
        )

    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=True

    )

    model.train()

    for epoch in range(
        epochs
    ):

        total_loss = 0

        for (

            X_batch,

            y_batch

        ) in loader:

            optimizer.zero_grad()

            full_out, _ = model(
                X_batch
            )

            port_ret = (

                full_out[
                    :,
                    selected_idx
                ]

                *

                X_batch[
                    :,
                    selected_idx
                ]

            ).sum(dim=1)

            loss = loss_fn(

                port_ret,

                y_batch

            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        if (

            epoch + 1

        ) % 20 == 0:

            print(

                f"Epoch "

                f"{epoch+1}"

                f"/{epochs}"

                f" Loss "

                f"{total_loss/len(loader):.8f}"

            )

    torch.save(

        model.state_dict(),

        "models/autoencoder.pt"

    )

    print(
        "Saved autoencoder"
    )

    return (

        model,

        selected_idx

    )


def plot_latent_factors(

    model,

    train_returns,

    all_stock_cols

):

    os.makedirs(
        "results",
        exist_ok=True
    )

    model.eval()

    X = torch.FloatTensor(

        train_returns[
            all_stock_cols
        ]

        .values

        .astype(np.float32)

    )

    with torch.no_grad():

        _, factors = model(X)

    factors = (
        factors.numpy()
    )

    n_factors = (
        factors.shape[1]
    )

    fig, axes = plt.subplots(

        n_factors,

        1,

        figsize=(12, 2*n_factors),

        sharex=True

    )

    if n_factors == 1:

        axes = [axes]

    dates = train_returns.index

    for i, ax in enumerate(axes):

        ax.plot(

            dates,

            factors[:, i],

            linewidth=0.7

        )

        ax.axhline(
            0,
            linewidth=0.5
        )

        ax.grid(
            alpha=0.2
        )

        ax.set_ylabel(
            f"F{i+1}"
        )

    plt.tight_layout()

    plt.savefig(

        "results/"
        "latent_factors.png",

        dpi=150

    )

    plt.show()

    print(
        "Saved latent factors"
    )