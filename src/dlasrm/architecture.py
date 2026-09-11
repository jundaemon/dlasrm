from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, device, no_grad, save
from torch.nn import Sequential
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Architecture:
    def __init__(
        self,
        modules: Sequential,
        loss_fn: _Loss,
        optimizer: Optimizer,
        eval_metric: Callable[[Tensor, Tensor], float],
        epochs: int,
        device: device,
    ) -> None:
        self.model = modules
        self.device = device

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.eval_metric = eval_metric
        self.epochs = epochs

    def train(
        self, train_loader: DataLoader, validation_loader: DataLoader
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        train_length = len(train_loader.dataset)  # type: ignore
        validation_length = len(validation_loader.dataset)  # type: ignore
        train_metric = np.empty(self.epochs)
        validation_metric = np.empty(self.epochs)

        for i in range(self.epochs):
            train_predicted = torch.empty(train_length, device=self.device)
            train_expected = torch.empty(train_length, device=self.device)

            self.model.train()
            j = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                pred = self.model(X_batch).squeeze(-1)
                loss = self.loss_fn(pred, y_batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                train_predicted[j : j + len(pred)] = pred.detach()
                train_expected[j : j + len(y_batch)] = y_batch.detach()
                j += len(pred)

            train_metric[i] = self.eval_metric(train_predicted, train_expected)

            validation_predicted = torch.empty(validation_length, device=self.device)
            validation_expected = torch.empty(validation_length, device=self.device)

            self.model.eval()
            with no_grad():
                j = 0
                for X_batch, y_batch in validation_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    pred = self.model(X_batch).squeeze(-1)

                    validation_predicted[j : j + len(pred)] = pred.detach()
                    validation_expected[j : j + len(y_batch)] = y_batch.detach()
                    j += len(pred)

                validation_metric[i] = self.eval_metric(
                    validation_predicted, validation_expected
                )

        return train_metric, validation_metric

    def save_model(self, path: str) -> None:
        save(self.model.state_dict(), path)
