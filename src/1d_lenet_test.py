from importlib import import_module

from torch.nn import MSELoss
from torch.optim import AdamW

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.architecture import Architecture
from dlasrm.data import preprocess_data, retrieve_samples, seed_training_env
from dlasrm.evaluation import MAE, plot_results

lenet = import_module("1d_lenet_tuning")


if __name__ == "__main__":
    X, y = retrieve_samples(name=DB_NAME, rows=TOTAL_SAMPLES, cols=BINS)
    train_loader, validation_loader, X_test, y_test = preprocess_data(X, y)

    seed_training_env(1)
    model = lenet.create_model()
    model.to(DEVICE)
    architecture = Architecture(
        modules=model,
        loss_fn=MSELoss(),
        optimizer=AdamW(params=model.parameters()),
        eval_metric=MAE(),
        epochs=100,
        early_stopping_rounds=10,
        delta=0.0005,
    )
    train_mae, validation_mae = architecture.train(train_loader, validation_loader)

    plot_results(
        train_arr=train_mae,
        validation_arr=validation_mae,
        metric="mae",
        path="results/1d_lenet.png",
    )
    print(f"\ntest mae: {architecture.evaluate(X_test, y_test)}")
    architecture.save_model(path="weights/1d_lenet.pth")
