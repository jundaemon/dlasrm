from optuna import create_study
from optuna.trial import Trial
from torch.nn import Conv1d, Flatten, Linear, MaxPool1d, MSELoss, ReLU, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.architecture import Architecture
from dlasrm.data import preprocess_data, retrieve_samples, seed_training_env
from dlasrm.evaluation import MAE, plot_results


def create_model() -> Sequential:
    return Sequential(
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


def objective(
    trial: Trial, train_loader: DataLoader, validation_loader: DataLoader
) -> float:
    seed_training_env(1)
    learning_rate = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.01, log=True)

    model = create_model()
    model.to(DEVICE)
    architecture = Architecture(
        model,
        MSELoss(),
        AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        MAE(),
        30,
        early_stopping_rounds=5,
        delta=0.0005,
    )
    _, validation_mae = architecture.train(train_loader, validation_loader)

    return validation_mae.min()


if __name__ == "__main__":
    X, y = retrieve_samples(DB_NAME, TOTAL_SAMPLES, BINS)
    train_loader, validation_loader, X_test, y_test = preprocess_data(X, y)

    study = create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, validation_loader), 30)
    best_lr = study.best_params["lr"]
    best_weight_decay = study.best_params["weight_decay"]

    print("\nstudy results")
    print(f"best learning rate: {best_lr}")
    print(f"best weight decay: {best_weight_decay}")
    # best learning rate: 0.0001677022202319308
    # best weight decay: 0.00012444477787113697
    print(f"lowest mae: {study.best_value}")

    seed_training_env(1)
    model = create_model()
    model.to(DEVICE)
    architecture = Architecture(
        model,
        MSELoss(),
        AdamW(model.parameters(), lr=best_lr, weight_decay=best_weight_decay),
        MAE(),
        30,
        early_stopping_rounds=5,
        delta=0.0005,
    )
    train_mae, validation_mae = architecture.train(train_loader, validation_loader)

    plot_results(train_mae, validation_mae, "mae", "results/1d_lenet.png")
    print(f"\ntest mae: {architecture.evaluate(X_test, y_test)}")
    # test mae: 0.009782887995243073
    architecture.save_model("weights/1d_lenet.pth")
