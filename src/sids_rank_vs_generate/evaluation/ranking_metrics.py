from __future__ import annotations

import torch


def roc_auc(targets: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(probabilities, descending=True, stable=True)
    labels = targets[order]
    positives = labels.sum()
    negatives = labels.numel() - positives
    true_positives = labels.cumsum(0)
    false_positives = (1.0 - labels).cumsum(0)
    distinct = torch.ones(labels.numel(), dtype=torch.bool)
    distinct[:-1] = probabilities[order][:-1] != probabilities[order][1:]
    tpr = torch.cat((torch.zeros(1), true_positives[distinct] / positives))
    fpr = torch.cat((torch.zeros(1), false_positives[distinct] / negatives))
    return torch.trapezoid(tpr, fpr)


def average_precision(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    labels = targets[torch.argsort(probabilities, descending=True, stable=True)]
    precision = labels.cumsum(0) / torch.arange(1, labels.numel() + 1)
    return (precision * labels).sum() / labels.sum()


def pairwise_accuracy(
    users: torch.Tensor,
    timestamps: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    session_gap_seconds: int = 15 * 60,
) -> torch.Tensor:
    time_order = torch.argsort(timestamps, stable=True)
    user_order = torch.argsort(users[time_order], stable=True)
    order = time_order[user_order]
    sorted_users = users[order]
    sorted_times = timestamps[order]
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]
    valid = (
        (sorted_users[:-1] == sorted_users[1:])
        & ((sorted_times[1:] - sorted_times[:-1]) <= session_gap_seconds)
        & (sorted_targets[:-1] != sorted_targets[1:])
    )
    if not valid.any():
        return torch.tensor(0.0)
    label_delta = sorted_targets[1:] - sorted_targets[:-1]
    probability_delta = sorted_probabilities[1:] - sorted_probabilities[:-1]
    product = label_delta[valid] * probability_delta[valid]
    return (product.gt(0).float() + 0.5 * product.eq(0).float()).mean()


def ranking_metrics(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    users: torch.Tensor,
    timestamps: torch.Tensor,
) -> dict[str, float]:
    targets = targets.float()
    probabilities = probabilities.float().clamp(1e-7, 1 - 1e-7)
    logloss = torch.nn.functional.binary_cross_entropy(probabilities, targets)
    return {
        "roc_auc": roc_auc(targets, probabilities).item(),
        "pr_auc": average_precision(targets, probabilities).item(),
        "logloss": logloss.item(),
        "pairwise_accuracy": pairwise_accuracy(
            users, timestamps, targets, probabilities
        ).item(),
    }
