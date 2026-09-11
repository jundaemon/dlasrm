import numpy as np

from dlasrm import (
    BINS,
    DB_NAME,
    EFF_1S,
    EFF_2S,
    HALF_INT_NS,
    INPUTS_N,
    LABELS_N,
    LIFETIME_NS,
    SEEDS,
    T_NS,
)
from dlasrm.data import create_database, store_samples
from dlasrm.hbt import input_gen, label_gen

if __name__ == "__main__":
    create_database(DB_NAME)
    for seed in SEEDS:
        inputs = input_gen(
            INPUTS_N, T_NS, LIFETIME_NS, EFF_1S, EFF_2S, HALF_INT_NS, BINS, seed
        ).astype(np.float32)
        labels = label_gen(
            LABELS_N, T_NS, LIFETIME_NS, EFF_1S, EFF_2S, HALF_INT_NS, BINS, seed
        ).astype(np.float32)
        print(
            f"seed: {seed}, inputs shape: {inputs.shape}, labels shape: {labels.shape}"
        )

        store_samples(
            DB_NAME, np.repeat(seed, len(EFF_1S)), EFF_1S, EFF_2S, inputs, labels
        )
