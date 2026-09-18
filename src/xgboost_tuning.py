import cupy as cp
import numpy as np
from numpy.typing import NDArray
from optuna import create_study
from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.data import retrieve_samples, train_validation_test_split


def objective(
    trial: Trial,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    X_validation: cp.ndarray,
    y_validation: NDArray[np.float32],
) -> float:
    n_estimators = trial.suggest_int("n_estimators", 100, 3_000)
    lr = trial.suggest_float("lr", 1e-5, 0.01, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    subsample = trial.suggest_float("subsample", 0.7, 1.0, step=0.1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 1.0, step=0.1)

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        device=DEVICE,
        random_state=1,
    )
    trees = model.fit(X_train, y_train)

    pred = trees.predict(X_validation)
    return mean_absolute_error(y_validation, pred)


if __name__ == "__main__":
    X, y = retrieve_samples(DB_NAME, TOTAL_SAMPLES, BINS)
    X_train, y_train, X_validation, y_validation, _, _ = train_validation_test_split(
        X, y, validation_ratio=0.15, test_ratio=0.15
    )
    X_validation = cp.asarray(X_validation)

    study = create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_validation, y_validation),
        n_trials=100,
    )
    print(f"best n_estimators: {study.best_params['n_estimators']}")
    print(f"best lr: {study.best_params['lr']}")
    print(f"best max_depth: {study.best_params['max_depth']}")
    print(f"best subsample: {study.best_params['subsample']}")
    print(f"best colsample_bytree: {study.best_params['colsample_bytree']}")
    print(f"lowest mae: {study.best_value}")
    # best n_estimators: 2890
    # best lr: 0.009651414673966265
    # best max_depth: 10
    # best subsample: 0.7
    # best colsample_bytree: 0.7
    # lowest mae: 0.010152874514460564
