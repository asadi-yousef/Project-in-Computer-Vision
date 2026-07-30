"""Linear probe: s = Wz + b, trained with softmax cross-entropy on top of
frozen, cached features. Only W and b are ever trained (stage_1.pdf).
"""

import dataclasses
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import LinearProbeHyperparams
from src.utils.seeding import set_seed


class LinearProbe(nn.Module):
    """s = Wz + b: a single linear layer over frozen features."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


@dataclasses.dataclass
class EpochLog:
    """One epoch's worth of the logging stage_1.pdf requires."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float


@dataclasses.dataclass
class TrainResult:
    best_epoch: int
    best_val_accuracy: float
    best_state_dict: Dict[str, torch.Tensor]
    history: List[EpochLog]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    hyperparams: LinearProbeHyperparams,
    seed: int,
    device: torch.device,
) -> TrainResult:
    """Train a linear probe on cached features, checkpointing on the highest
    validation accuracy.

    `seed` controls both weight initialization and training-batch shuffling
    (the project's single-seed convention; see src/utils/config.py).

    Args:
        train_features, train_labels: cached training features/labels,
            already restricted to the K-shot subset if applicable.
        val_features, val_labels: cached validation features/labels — always
            the full official validation split, used only for checkpoint
            selection, never for gradient updates.
        num_classes: number of classes in this dataset.
        hyperparams: optimizer/training configuration.
        seed: seed for model initialization and data shuffling.
        device: device to train on.

    Returns:
        A TrainResult with the best epoch, its validation accuracy, that
        epoch's model state dict, and the full per-epoch history (needed
        for the required training/validation loss curves).
    """
    if hyperparams.optimizer != "adamw":
        raise ValueError(f"Only 'adamw' is currently supported, got {hyperparams.optimizer!r}")

    set_seed(seed)

    feature_dim = train_features.shape[1]
    model = LinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=hyperparams.batch_size,
        shuffle=True,
    )
    val_features_on_device = val_features.to(device)
    val_labels_on_device = val_labels.to(device)

    history: List[EpochLog] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    best_state_dict: Dict[str, torch.Tensor] = {}

    for epoch in range(1, hyperparams.max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        num_train_samples = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_labels.shape[0]
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            num_train_samples += batch_size

        train_loss = running_loss / num_train_samples
        train_accuracy = running_correct / num_train_samples

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features_on_device)
            val_loss = loss_fn(val_logits, val_labels_on_device).item()
            val_accuracy = _accuracy(val_logits, val_labels_on_device)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            EpochLog(epoch, train_loss, val_loss, train_accuracy, val_accuracy, current_lr)
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    return TrainResult(
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_state_dict=best_state_dict,
        history=history,
    )


def evaluate_linear_probe(
    state_dict: Dict[str, torch.Tensor],
    feature_dim: int,
    num_classes: int,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Load a trained linear-probe checkpoint and compute top-1 test accuracy.

    Intended to be called exactly once per run, on the best-val-accuracy
    checkpoint, after model selection is finalized: stage_1.pdf requires
    that test accuracy never influence epoch or hyperparameter selection.
    """
    model = LinearProbe(feature_dim, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(test_features.to(device))
        return _accuracy(logits, test_labels.to(device))
