from importlib import import_module

import numpy as np
from torch.nn import MSELoss
from torch.optim import AdamW

from dlasrm import (
    BINS,
    DB_NAME,
    DEVICE,
    EFF_1S,
    EFF_2S,
    HALF_INT_NS,
    LIFETIME_NS,
    SEEDS,
    T_NS,
    TOTAL_SAMPLES,
)
from dlasrm.architecture import Architecture
from dlasrm.data import modify_samples, preprocess_data, retrieve_samples
from dlasrm.evaluation import MAE, plot_landscape
from dlasrm.simulation import input_gen

lenet = import_module("1d_lenet")


if __name__ == "__main__":
    n_array = np.arange(500, 50_001, 500)
    mae_array = np.empty(len(n_array))

    for i, n in enumerate(n_array):
        print(f"\ninputs n: {n}")

        for seed in SEEDS:
            inputs = input_gen(
                n, T_NS, LIFETIME_NS, EFF_1S, EFF_2S, HALF_INT_NS, BINS, seed
            ).astype(np.float32)
            print(f"seed: {seed}, inputs shape: {inputs.shape}")

            modify_samples(
                DB_NAME, np.repeat(seed, len(EFF_1S)), EFF_1S, EFF_2S, inputs
            )

        X, y = retrieve_samples(DB_NAME, TOTAL_SAMPLES, BINS)
        train_loader, validation_loader, X_test, y_test = preprocess_data(X, y)

        model = lenet.create_model()
        model.to(DEVICE)
        architecture = Architecture(
            model,
            MSELoss(),
            AdamW(
                model.parameters(),
                lr=0.0001677022202319308,
                weight_decay=0.00012444477787113697,
            ),
            MAE(),
            30,
            early_stopping_rounds=5,
            delta=0.0005,
        )
        _ = architecture.train(train_loader, validation_loader)
        mae_array[i] = architecture.evaluate(X_test, y_test)

    plot_landscape(n_array, mae_array, "mae", "results/landscape.png")
