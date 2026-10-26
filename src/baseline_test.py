import numpy as np
from sklearn.metrics import mean_absolute_error

from dlasrm import BINS, DB_NAME, T_NS, TOTAL_SAMPLES
from dlasrm.data import retrieve_samples
from dlasrm.simulation import calculate_g20

if __name__ == "__main__":
    X, y = retrieve_samples(name=DB_NAME, rows=TOTAL_SAMPLES, cols=BINS)
    pred = np.empty(len(X), dtype=np.float32)
    for i, hist in enumerate(X):
        pred[i] = calculate_g20(hist.astype(np.int64), T_NS)

    print(f"test mae: {mean_absolute_error(y, pred)}")
    # test mae: 0.09627816081047058
