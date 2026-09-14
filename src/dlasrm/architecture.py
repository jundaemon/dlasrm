import numpy as np
import torch
from numpy.typing import NDArray
from torch import device, no_grad, save
from torch.nn import Sequential
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from dlasrm import DEVICE
from dlasrm.evaluation import EvalMetric


class Architecture:
    def __init__(
        self,
        modules: Sequential,
        loss_fn: _Loss,
        optimizer: Optimizer,
        eval_metric: EvalMetric,
        epochs: int,
        early_stopping_rounds: int = 0,
        delta: float = 0.0,
        device: device = DEVICE,
    ) -> None:
        self.model = modules
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.eval_metric = eval_metric
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.delta = delta
        self.device = device

    def train(
        self, train_loader: DataLoader, validation_loader: DataLoader
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        train_length = len(train_loader.dataset)  # type: ignore
        validation_length = len(validation_loader.dataset)  # type: ignore
        train_metrics = np.empty(self.epochs)
        validation_metrics = np.empty(self.epochs)

        best_weights = self.model.state_dict()
        patience = 0

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

            curr_train = self.eval_metric.calc(train_predicted, train_expected)
            train_metrics[i] = curr_train

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

                curr_validation = self.eval_metric.calc(
                    validation_predicted, validation_expected
                )
                validation_metrics[i] = curr_validation

                print(f"\nepoch {1 + i}")
                print(f"train: {curr_train}")
                print(f"validation: {curr_validation}")

                if i == 0:
                    print(f"patience: {patience}")
                    continue

                if self.eval_metric.is_best(validation_metrics, curr_validation):
                    best_weights = self.model.state_dict()

                if self.early_stopping_rounds != 0:
                    patience = self.eval_metric.patience_gate(
                        validation_metrics[i - 1], curr_validation, self.delta, patience
                    )
                    print(f"patience: {patience}")
                    if patience >= self.early_stopping_rounds:
                        break

        self.model.load_state_dict(best_weights)
        return train_metrics, validation_metrics

    def save_model(self, path: str) -> None:
        save(self.model.state_dict(), path)
