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
    statistics = pairwise_statistics(
        users,
        timestamps,
        targets,
        probabilities,
        session_gap_seconds,
    )
    return torch.tensor(statistics["pairwise_accuracy"])


def pairwise_accuracy_and_count(
    users: torch.Tensor,
    timestamps: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    session_gap_seconds: int = 15 * 60,
) -> tuple[torch.Tensor, int]:
    statistics = pairwise_statistics(
        users,
        timestamps,
        targets,
        probabilities,
        session_gap_seconds,
    )
    return (
        torch.tensor(statistics["pairwise_accuracy"]),
        int(statistics["pairwise_comparisons"]),
    )


def pairwise_statistics(
    users: torch.Tensor,
    timestamps: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    session_gap_seconds: int = 15 * 60,
) -> dict[str, float | int]:
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
        return {
            "pairwise_accuracy": 0.0,
            "pairwise_comparisons": 0,
            "pairwise_users": 0,
            "pairwise_accuracy_std_error": 0.0,
            "pairwise_accuracy_ci95_low": 0.0,
            "pairwise_accuracy_ci95_high": 0.0,
        }
    label_delta = sorted_targets[1:] - sorted_targets[:-1]
    probability_delta = sorted_probabilities[1:] - sorted_probabilities[:-1]
    product = label_delta[valid] * probability_delta[valid]
    outcomes = (
        product.gt(0).float() + 0.5 * product.eq(0).float()
    )
    accuracy = outcomes.mean()
    pair_users = sorted_users[:-1][valid]
    unique_users, user_inverse = torch.unique(pair_users, return_inverse=True)
    user_count = unique_users.numel()
    centered_sums = torch.zeros(
        user_count,
        dtype=torch.float32,
        device=outcomes.device,
    )
    centered_sums.scatter_add_(0, user_inverse, outcomes - accuracy)
    if user_count > 1:
        variance = (
            user_count
            / (user_count - 1)
            * centered_sums.square().sum()
            / outcomes.numel() ** 2
        )
        standard_error = variance.sqrt()
    else:
        standard_error = torch.tensor(0.0, device=outcomes.device)
    margin = 1.96 * standard_error
    return {
        "pairwise_accuracy": accuracy.item(),
        "pairwise_comparisons": outcomes.numel(),
        "pairwise_users": user_count,
        "pairwise_accuracy_std_error": standard_error.item(),
        "pairwise_accuracy_ci95_low": (accuracy - margin).clamp(0, 1).item(),
        "pairwise_accuracy_ci95_high": (accuracy + margin).clamp(0, 1).item(),
    }


def ranking_metrics(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    users: torch.Tensor,
    timestamps: torch.Tensor,
) -> dict[str, float | int]:
    targets = targets.float()
    probabilities = probabilities.float().clamp(1e-7, 1 - 1e-7)
    logloss = torch.nn.functional.binary_cross_entropy(probabilities, targets)
    pairwise = pairwise_statistics(
        users, timestamps, targets, probabilities
    )
    return {
        **pairwise,
        "roc_auc": roc_auc(targets, probabilities).item(),
        "pr_auc": average_precision(targets, probabilities).item(),
        "logloss": logloss.item(),
    }
