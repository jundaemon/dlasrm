from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import override


class EvalMetric(ABC):
    @staticmethod
    @abstractmethod
    def calc(predicted: Tensor, expected: Tensor) -> float:
        raise NotImplementedError


class MAE(EvalMetric):
    @staticmethod
    @override
    def calc(predicted: Tensor, expected: Tensor) -> float:
        return (expected - predicted).abs().mean().item()


def plot_results(
    train_arr: NDArray[np.float64],
    validation_arr: NDArray[np.float64],
    metric: str,
    path: str,
) -> None:
    epochs = np.arange(1, len(train_arr) + 1)
    plt.plot(epochs, train_arr, label=f"training {metric}", color="b")
    plt.plot(epochs, validation_arr, label=f"validation {metric}", color="y")

    plt.title(f"epochs - {metric}")
    plt.xlabel("epochs")
    plt.ylabel(metric)

    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close("all")
