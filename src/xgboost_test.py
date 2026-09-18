import cupy as cp
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from dlasrm import BINS, DB_NAME, DEVICE, TOTAL_SAMPLES
from dlasrm.data import retrieve_samples, train_validation_test_split

if __name__ == "__main__":
    X, y = retrieve_samples(DB_NAME, TOTAL_SAMPLES, BINS)
    X_train, y_train, X_validation, y_validation, X_test, y_test = (
        train_validation_test_split(X, y, validation_ratio=0.15, test_ratio=0.15)
    )
    X_test = cp.asarray(X_test)

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric=mean_absolute_error,
        n_estimators=2890,
        learning_rate=0.009651414673966265,
        max_depth=10,
        subsample=0.7,
        colsample_bytree=0.7,
        device=DEVICE,
        random_state=1,
    )
    trees = model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
    trees.save_model("weights/xgboost.json")

    pred = model.predict(X_test)
    print(f"test mae: {mean_absolute_error(y_test, pred)}")
    # test mae: 0.010119445621967316
