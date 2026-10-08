from optuna import Trial, create_study
from torch.nn import LSTM, Linear, MSELoss, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.architecture import Architecture, LSTMBridge
from dlasrm.data import preprocess_data, retrieve_samples, seed_training_env
from dlasrm.evaluation import MAE


def create_model(hidden_size: int, num_layers: int) -> Sequential:
    return Sequential(
        LSTM(
            input_size=BINS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        ),
        LSTMBridge(),
        Linear(in_features=hidden_size, out_features=1),
    )


def objective(
    trial: Trial, train_loader: DataLoader, validation_loader: DataLoader
) -> float:
    seed_training_env(1)
    learning_rate = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.01, log=True)
    hidden_size = trial.suggest_categorical(
        "hidden_size", [32, 64, 128, 256, 512, 1_024]
    )
    num_layers = trial.suggest_int("num_layers", 1, 3)
    epochs = trial.suggest_int("epochs", 10, 100)

    model = create_model(hidden_size, num_layers)
    model.to(DEVICE)
    architecture = Architecture(
        modules=model,
        loss_fn=MSELoss(),
        optimizer=AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
        eval_metric=MAE(),
        epochs=epochs,
    )
    _, validation_mae = architecture.train(
        train_loader=train_loader,
        validation_loader=validation_loader,
        tuning=True,
        trial=trial,
    )

    return validation_mae.max()


if __name__ == "__main__":
    X, y = retrieve_samples(name=DB_NAME, rows=TOTAL_SAMPLES, cols=BINS)
    train_loader, validation_loader, _, _ = preprocess_data(X, y)

    study = create_study(direction="minimize")
    study.optimize(
        func=lambda trial: objective(trial, train_loader, validation_loader),
        n_trials=50,
    )

    print("\nstudy results")
    print(f"best learning rate: {study.best_params['lr']}")
    print(f"best weight decay: {study.best_params['weight_decay']}")
    print(f"best hidden_size: {study.best_params['hidden_size']}")
    print(f"best num_layers: {study.best_params['num_layers']}")
    print(f"best number of epochs: {study.best_params['epochs']}")
    print(f"lowest mae: {study.best_value}")
