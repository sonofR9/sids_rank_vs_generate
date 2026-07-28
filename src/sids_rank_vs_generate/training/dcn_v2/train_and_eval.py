from tqdm.auto import tqdm
import torch
from torch import nn
from collections import defaultdict
from src.sids_rank_vs_generate.data.dcn_v2.preprocessing import to_device
from src.sids_rank_vs_generate.evaluation.metrics.dcn_v2.compute_pairwise_accuracy import compute_pairwise_accuracy

def evaluate_epoch(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.eval()

    collected = defaultdict(list)
    type_to_columns = {
        "meta": ["uid", "timestamp"],
        "labels": ["is_like", "is_full_play"],
    }

    with torch.inference_mode():
        for batch in test_loader:
            for _type, columns in type_to_columns.items():
                for column in columns:
                    collected[column].append(batch[_type][column])

            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", torch.bfloat16):
                predictions = model(batch)
                predictions = predictions.float()

            collected["predictions"].append(predictions.cpu())

    concatenated = {}
    for key, value in collected.items():
        concatenated[key] = torch.cat(value).numpy()

    metrics = {}

    def compute_metric_wrapper(
        main_metric: str, pair_metric: str, predictions: np.ndarray
    ):
        metrics[f"pair_accuracy_{main_metric}"] = compute_pairwise_accuracy(
            concatenated["uid"],
            concatenated["timestamp"],
            concatenated[f"is_{main_metric}"],
            predictions,
        )
        metrics[f"pair_accuracy_{main_metric}_{pair_metric}_pairs"] = (
            compute_pairwise_accuracy(
                concatenated["uid"],
                concatenated["timestamp"],
                concatenated[f"is_{pair_metric}"],
                predictions,
            )
        )

    if predictions.shape[1] == 1:
        probs = concatenated["predictions"][:, 0]
        compute_metric_wrapper("full_play", "like", probs)
    else:
        like_probs = concatenated["predictions"][:, 0]
        full_probs = concatenated["predictions"][:, 1]

        compute_metric_wrapper("full_play", "like", full_probs)
        compute_metric_wrapper("like", "full_play", like_probs)

    return metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    embed_lr: float,
    train_log_every: int,
):
    #####################
    ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
    #####################
    device = "cuda"
    model = model.to(device)

    dense_optimizer = torch.optim.Adam(
        [
            param
            for name, param in model.named_parameters()
            if "embedding" not in name
        ],
        lr=lr,
        fused=True,
    )
    embeddings_optimizer = torch.optim.Adam(
        [
            param
            for name, param in model.named_parameters()
            if "embedding" in name
        ],
        lr=embed_lr,
        fused=True,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()

        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch} [train]",
            leave=False,
        )
        loss_so_far = 0
        for i, batch in train_pbar:
            batch = to_device(batch, device)

            with torch.amp.autocast("cuda", torch.bfloat16):
                predictions = model(batch)

            if predictions.shape[1] == 1:
                target = batch["labels"]["is_full_play"].unsqueeze(1)
                loss = loss_fn(predictions, target)
            else:
                target = torch.stack(
                    [
                        batch["labels"]["is_like"],
                        batch["labels"]["is_full_play"],
                    ],
                    dim=1,
                )
                loss = loss_fn(predictions, target)

            loss_so_far += loss.detach()

            dense_optimizer.zero_grad()
            embeddings_optimizer.zero_grad()
            loss.backward()
            dense_optimizer.step()
            embeddings_optimizer.step()

            if i % train_log_every == 0:
                train_pbar.set_postfix(
                    {"loss": f"{loss_so_far.item() / (i + 1):.4f}"}
                )

        metrics = evaluate_epoch(model, test_loader, device)
        print(f"\nEpoch {epoch} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.5f}")

    return metrics