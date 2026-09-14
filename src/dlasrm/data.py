import sqlite3

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader, TensorDataset


def seed_training_env(state: int) -> None:
    torch.manual_seed(state)
    torch.cuda.manual_seed(state)


def create_database(name: str) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE samples (
    seed INTEGER NOT NULL,
    eff_1 REAL NOT NULL,
    eff_2 REAL NOT NULL,
    input BLOB NOT NULL,
    label REAL NOT NULL,
    PRIMARY KEY (seed, eff_1, eff_2)
);""")
    conn.commit()

    conn.close()


def store_samples(
    name: str,
    seeds: NDArray[np.int64],
    eff_1s: NDArray[np.float64],
    eff_2s: NDArray[np.float64],
    X: NDArray[np.float32],
    y: NDArray[np.float32],
) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    values = []
    for seed, eff_1, eff_2, input, label in zip(seeds, eff_1s, eff_2s, X, y):
        values.append((seed, eff_1, eff_2, input.tobytes(), label))

    cursor.executemany(
        "INSERT INTO samples (seed, eff_1, eff_2, input, label) VALUES (?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()

    conn.close()


def modify_samples(
    name: str,
    seeds: NDArray[np.int64],
    eff_1s: NDArray[np.float64],
    eff_2s: NDArray[np.float64],
    X: NDArray[np.float32],
) -> None:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    values = []
    for seed, eff_1, eff_2, input in zip(seeds, eff_1s, eff_2s, X):
        values.append((input.tobytes(), seed, eff_1, eff_2))

    cursor.executemany(
        "UPDATE samples SET input = ? WHERE seed = ? AND eff_1 = ? AND eff_2 = ?",
        values,
    )
    conn.commit()

    conn.close()


def retrieve_samples(
    name: str, rows: int, cols: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    inputs = np.empty((rows, cols), dtype=np.float32)
    labels = np.empty(rows, dtype=np.float32)
    for i, row in enumerate(cursor.execute("SELECT input, label FROM samples;")):
        inputs[i] = np.frombuffer(row[0], dtype=np.float32)
        labels[i] = np.frombuffer(row[1], dtype=np.float32)[0]

    conn.close()
    return inputs, labels


def train_validation_test_split(
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    validation_ratio: float,
    test_ratio: float,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    length = len(X)
    X_temp, X_validation, y_temp, y_validation = train_test_split(
        X, y, test_size=int(length * validation_ratio), shuffle=True, random_state=1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=int(length * test_ratio), shuffle=True, random_state=1
    )

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def convert_to_tensors(
    X: NDArray[np.float32], y: NDArray[np.float32]
) -> tuple[Tensor, Tensor]:
    X_tensor = from_numpy(X)
    X_tensor = X_tensor.unsqueeze(1)
    y_tensor = from_numpy(y)

    return X_tensor, y_tensor


def create_loader(X: NDArray[np.float32], y: NDArray[np.float32]) -> DataLoader:
    X_tensor, y_tensor = convert_to_tensors(X, y)

    return DataLoader(
        dataset=TensorDataset(X_tensor, y_tensor),
        batch_size=64,
        shuffle=True,
        pin_memory=True,
    )


def preprocess_data(
    X: NDArray[np.float32], y: NDArray[np.float32]
) -> tuple[DataLoader, DataLoader, Tensor, Tensor]:
    X_train, y_train, X_validation, y_validation, X_test, y_test = (
        train_validation_test_split(X, y, 0.15, 0.15)
    )

    train_loader = create_loader(X_train, y_train)
    validation_loader = create_loader(X_validation, y_validation)

    X_test, y_test = convert_to_tensors(X_test, y_test)

    return train_loader, validation_loader, X_test, y_test
