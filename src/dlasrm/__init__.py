import numpy as np

from dlasrm.simulation import seed_gen

EFF_1S = np.repeat(np.linspace(0.005, 1.0, 200), 200)
EFF_2S = EFF_1S.reshape(200, 200).T.ravel()
LABELS_N = 300_000
INPUTS_N = 3_000
DARK_COUNT_RATE_NS = 2e-7
BACKGROUND_RATIO = 0.1
PULSE_WIDTH_NS = 1
T_NS = 50
LIFETIME_NS = 3
HALF_INT_NS = 250
BINS = 500

SEEDS = seed_gen(10)
TOTAL_SAMPLES = len(EFF_1S) * len(SEEDS)
DEVICE = "cuda"
DB_NAME = "samples.db"
DB_NAME_2 = "samples_2.db"
BATCH_SIZE = 64
