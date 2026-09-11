from torch import device
from torch.nn import Conv1d, Flatten, Linear, MaxPool1d, MSELoss, ReLU, Sequential
from torch.optim import Adam

from dlasrm import BINS, DB_NAME, EPOCHS, TOTAL_SAMPLES
from dlasrm.architecture import Architecture
from dlasrm.data import (
    convert_to_tensors,
    create_loader,
    retrieve_samples,
    train_validation_test_split,
)
from dlasrm.evaluation import mean_absolute_error

if __name__ == "__main__":
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
    architecture = Architecture(
        model,
        MSELoss(),
        Adam(model.parameters()),
        mean_absolute_error,
        EPOCHS,
        device("cuda"),
    )

    X, y = retrieve_samples(DB_NAME, TOTAL_SAMPLES, BINS)
    X_train, y_train, X_validation, y_validation, X_test, y_test = (
        train_validation_test_split(X, y, 0.15, 0.15)
    )
    train_loader = create_loader(X_train, y_train)
    validation_loader = create_loader(X_validation, y_validation)
    X_test, y_test = convert_to_tensors(X_test, y_test)

    architecture.train(train_loader, validation_loader)
    architecture.save_model("weights/modded_1d_lenet.pth")
