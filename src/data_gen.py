import numpy as np

from dlasrm import (
    BACKGROUND_RATIO,
    BINS,
    DARK_COUNT_RATE_NS,
    DB_NAME,
    EFF_1S,
    EFF_2S,
    HALF_INT_NS,
    INPUTS_N,
    LABELS_N,
    LIFETIME_NS,
    PULSE_WIDTH_NS,
    SEEDS,
    T_NS,
)
from dlasrm.data import create_database, store_samples
from dlasrm.simulation import input_gen, label_gen

if __name__ == "__main__":
    create_database(name=DB_NAME)
    for seed in SEEDS:
        inputs = input_gen(
            INPUTS_N,
            DARK_COUNT_RATE_NS,
            BACKGROUND_RATIO,
            PULSE_WIDTH_NS,
            T_NS,
            LIFETIME_NS,
            EFF_1S,
            EFF_2S,
            HALF_INT_NS,
            BINS,
            seed,
        ).astype(np.float32)
        labels = label_gen(
            LABELS_N,
            T_NS,
            LIFETIME_NS,
            EFF_1S,
            EFF_2S,
            HALF_INT_NS,
            BINS,
            seed,
        ).astype(np.float32)
        print(
            f"seed: {seed}, inputs shape: {inputs.shape}, labels shape: {labels.shape}"
        )

        store_samples(
            name=DB_NAME,
            seeds=np.repeat(seed, len(EFF_1S)),
            eff_1s=EFF_1S,
            eff_2s=EFF_2S,
            X=inputs,
            y=labels,
        )
