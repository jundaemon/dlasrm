import numpy as np
from torch import device

from dlasrm.hbt import seed_gen

EFF_1S = np.repeat(np.linspace(0.1, 1.0, 91), 91)
EFF_2S = EFF_1S.reshape(91, 91).T.ravel()
INPUTS_N = 4_000
LABELS_N = 500_000
T_NS = 50
LIFETIME_NS = 3
HALF_INT_NS = 250
BINS = 500

SEEDS = seed_gen(121)
TOTAL_SAMPLES = len(EFF_1S) * len(SEEDS)
EPOCHS = 100
DEVICE = device("cuda")
DB_NAME = "samples.db"
