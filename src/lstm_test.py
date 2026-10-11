from torch.nn import MSELoss
from torch.optim import AdamW

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.architecture import Architecture
from dlasrm.data import preprocess_data, retrieve_samples, seed_training_env
from dlasrm.evaluation import MAE, plot_results
from lstm_tuning import create_model

if __name__ == "__main__":
    X, y = retrieve_samples(name=DB_NAME, rows=TOTAL_SAMPLES, cols=BINS)
    train_loader, validation_loader, X_test, y_test = preprocess_data(X, y)

    seed_training_env(1)
    model = create_model(hidden_size=1_024, num_layers=3)
    model.to(DEVICE)
    architecture = Architecture(
        modules=model,
        loss_fn=MSELoss(),
        optimizer=AdamW(
            params=model.parameters(),
            lr=1.3e-5,
            weight_decay=1.9e-5,
        ),
        eval_metric=MAE(),
        epochs=62,
    )
    train_mae, validation_mae = architecture.train(train_loader, validation_loader)

    plot_results(
        train_arr=train_mae,
        validation_arr=validation_mae,
        metric="mae",
        path="results/lstm.png",
    )
    print(f"\ntest mae: {architecture.evaluate(X_test, y_test)}")
    architecture.save_model(path="weights/lstm.pth")
    # test mae: 0.009873180650174618
