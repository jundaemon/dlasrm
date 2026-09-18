import cupy as cp
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

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
from dlasrm.data import modify_samples, retrieve_samples, train_validation_test_split
from dlasrm.evaluation import plot_landscape
from dlasrm.simulation import input_gen

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
                name=DB_NAME,
                seeds=np.repeat(seed, len(EFF_1S)),
                eff_1s=EFF_1S,
                eff_2s=EFF_2S,
                X=inputs,
            )

        X, y = retrieve_samples(name=DB_NAME, rows=TOTAL_SAMPLES, cols=BINS)
        X_train, y_train, X_validation, y_validation, X_test, y_test = (
            train_validation_test_split(X, y, validation_ratio=0.15, test_ratio=0.15)
        )
        X_test = cp.asarray(X_test)

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric=mean_absolute_error,
            early_stopping_rounds=10,
            n_estimators=2890,
            learning_rate=0.009651414673966265,
            max_depth=10,
            subsample=0.7,
            colsample_bytree=0.7,
            device=DEVICE,
            random_state=1,
        )
        trees = model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])

        pred = trees.predict(X_test)
        mae_array[i] = mean_absolute_error(y_test, pred)

    plot_landscape(
        n_array=n_array,
        metric_array=mae_array,
        metric="mae",
        path="results/landscape.png",
    )
