from optuna.trial import Trial
from torch.nn import Conv1d, Flatten, Linear, MaxPool1d, MSELoss, ReLU, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.architecture import Architecture
from dlasrm.data import (
    preprocess_data,
    retrieve_samples,
)
from dlasrm.evaluation import MAE, plot_results


def objective(
    trial: Trial, train_loader: DataLoader, validation_loader: DataLoader
) -> float:
    learning_rate = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.01, log=True)

    model = Sequential(
        Conv1d(1, 6, 5),
        ReLU(),
        MaxPool1d(2, 2),
        Conv1d(6, 16, 5),
        ReLU(),
        MaxPool1d(2, 2),
        Flatten(),
        Linear(1_952, 84),
        ReLU(),
        Linear(84, 10),
        ReLU(),
        Linear(10, 1),
    )
    model.to(DEVICE)
    architecture = Architecture(
        model,
        MSELoss(),
        AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        MAE(),
        early_stopping_rounds=10,
        delta=0.0005,
    )
    _, validation_mae = architecture.train(train_loader, validation_loader)

    return validation_mae.max()


if __name__ == "__main__":
    X, y = retrieve_samples(DB_NAME, TOTAL_SAMPLES, BINS)
    train_loader, validation_loader, X_test, y_test = preprocess_data(X, y)

    model = Sequential(
        Conv1d(1, 6, 5),
        ReLU(),
        MaxPool1d(2, 2),
        Conv1d(6, 16, 5),
        ReLU(),
        MaxPool1d(2, 2),
        Flatten(),
        Linear(1_952, 84),
        ReLU(),
        Linear(84, 10),
        ReLU(),
        Linear(10, 1),
    )
    model.to(DEVICE)

    architecture = Architecture(
        model,
        MSELoss(),
        AdamW(model.parameters()),
        MAE(),
        early_stopping_rounds=10,
        delta=0.0005,
    )
    train_mae, validation_mae = architecture.train(train_loader, validation_loader)
    plot_results(train_mae, validation_mae, "mae", "results/1d_lenet.png")
