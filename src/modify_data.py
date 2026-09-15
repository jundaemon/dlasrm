import numpy as np

from dlasrm import (
    BINS,
    DB_NAME,
    EFF_1S,
    EFF_2S,
    HALF_INT_NS,
    LIFETIME_NS,
    SEEDS,
    T_NS,
)
from dlasrm.data import modify_samples
from dlasrm.simulation import input_gen

if __name__ == "__main__":
    for seed in SEEDS:
        inputs = input_gen(
            4_000, T_NS, LIFETIME_NS, EFF_1S, EFF_2S, HALF_INT_NS, BINS, seed
        ).astype(np.float32)
        print(f"seed: {seed}, inputs shape: {inputs.shape}")

        modify_samples(
            name=DB_NAME,
            seeds=np.repeat(seed, len(EFF_1S)),
            eff_1s=EFF_1S,
            eff_2s=EFF_2S,
            X=inputs,
        )
