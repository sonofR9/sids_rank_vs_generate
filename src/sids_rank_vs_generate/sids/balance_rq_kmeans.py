import numpy as np
import polars as pl
import torch

from ..reproducibility import SID_EXPERIMENT_SEED
from ..tracking import ExperimentTracker, NullTracker


class BalanceRQKMeans:
    def __init__(
        self,
        num_levels=None,
        codebook_size=None,
        codebook_sizes=None,
        kmeans_batch_size=65_536,
        predict_batch_size=16_384,
        max_iter=50,
        device=None,
    ):
        if codebook_sizes is not None:
            if num_levels is not None or codebook_size is not None:
                raise ValueError(
                    "Передай либо codebook_sizes, "
                    "либо num_levels и codebook_size"
                )

            self.codebook_sizes = tuple(codebook_sizes)
            self.num_levels = len(self.codebook_sizes)

        else:
            if num_levels is None or codebook_size is None:
                raise ValueError(
                    "Нужно передать num_levels и codebook_size"
                )

            self.num_levels = num_levels
            self.codebook_sizes = (codebook_size,) * num_levels

        self.kmeans_batch_size = kmeans_batch_size
        self.predict_batch_size = predict_batch_size
        self.max_iter = max_iter

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.codebooks: list[torch.Tensor] = []
        self.fit_codes_: torch.Tensor | None = None
        self.n_iter_: list[int] = []

    def converter(
        self,
        X: pl.Series | pl.DataFrame | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(X, pl.DataFrame):
            X = X["normalized_embed"]

        if isinstance(X, pl.Series):
            X = np.stack(X.to_list())

        if isinstance(X, torch.Tensor):
            return X.to(
                device=self.device,
                dtype=torch.float32,
            )

        return torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            device=self.device,
        )

    @staticmethod
    def get_cluster_sizes(
        n_items: int,
        n_clusters: int,
        device: torch.device,
    ) -> torch.Tensor:
        if n_clusters > n_items:
            raise ValueError(
                f"Кластеров {n_clusters} больше, чем объектов {n_items}"
            )

        mean, remainder = divmod(n_items, n_clusters)

        cluster_sizes = torch.full(
            (n_clusters,),
            mean,
            dtype=torch.long,
            device=device,
        )

        cluster_sizes[:remainder] += 1

        return cluster_sizes

    def compute_distances_to_center(
        self,
        X: torch.Tensor,
        indices: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        distances = torch.empty(
            len(indices),
            dtype=X.dtype,
            device=self.device,
        )

        center_norm = torch.sum(center * center)

        for start in range(
            0,
            len(indices),
            self.kmeans_batch_size,
        ):
            end = min(
                start + self.kmeans_batch_size,
                len(indices),
            )

            batch_indices = indices[start:end]
            batch = X[batch_indices]

            # ||x - c||² = ||x||² + ||c||² - 2x·c
            distances[start:end] = (
                torch.sum(batch * batch, dim=1)
                + center_norm
                - 2 * torch.mv(batch, center)
            )

        return distances

    def balanced_assignment(
        self,
        X: torch.Tensor,
        centers: torch.Tensor,
        cluster_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_items = len(X)
        n_clusters = len(centers)

        labels = torch.full(
            (n_items,),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        new_centers = torch.empty_like(centers)

        unassigned_indices = torch.arange(
            n_items,
            device=self.device,
        )

        for cluster_idx in range(n_clusters):
            current_size = int(cluster_sizes[cluster_idx].item())

            if cluster_idx == n_clusters - 1:
                assigned_indices = unassigned_indices

            else:
                distances = self.compute_distances_to_center(
                    X=X,
                    indices=unassigned_indices,
                    center=centers[cluster_idx],
                )

                nearest_positions = torch.topk(
                    distances,
                    k=current_size,
                    largest=False,
                    sorted=False,
                ).indices

                assigned_indices = unassigned_indices[
                    nearest_positions
                ]

                keep_mask = torch.ones(
                    len(unassigned_indices),
                    dtype=torch.bool,
                    device=self.device,
                )
                keep_mask[nearest_positions] = False

                unassigned_indices = unassigned_indices[
                    keep_mask
                ]

            labels[assigned_indices] = cluster_idx

            new_centers[cluster_idx] = X[
                assigned_indices
            ].mean(dim=0)

        return labels, new_centers

    def fit_balanced_level(
        self,
        X: torch.Tensor,
        n_clusters: int,
        cluster_sizes: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        initial_indices = torch.randperm(
            len(X),
            generator=generator,
            device=self.device,
        )[:n_clusters]

        centers = X[initial_indices].clone()
        previous_labels = None

        for iteration in range(1, self.max_iter + 1):
            labels, new_centers = self.balanced_assignment(
                X=X,
                centers=centers,
                cluster_sizes=cluster_sizes,
            )

            centers = new_centers

            if (
                previous_labels is not None
                and torch.equal(labels, previous_labels)
            ):
                return centers, labels, iteration

            previous_labels = labels.clone()

        return centers, labels, self.max_iter

    @torch.no_grad()
    def fit(
        self,
        X: pl.Series | pl.DataFrame | np.ndarray | torch.Tensor,
        *,
        tracker: ExperimentTracker | None = None,
    ):
        tracker = tracker or NullTracker()
        X = self.converter(X)
        residual = X.clone()

        self.codebooks = []
        self.n_iter_ = []

        fit_codes = torch.empty(
            (len(X), self.num_levels),
            dtype=torch.int32,
            device=self.device,
        )

        generator = torch.Generator(
            device=self.device,
        )
        generator.manual_seed(SID_EXPERIMENT_SEED)

        for level, codebook_size in enumerate(self.codebook_sizes):
            cluster_sizes = self.get_cluster_sizes(
                n_items=len(X),
                n_clusters=codebook_size,
                device=self.device,
            )

            codebook, labels, n_iter = self.fit_balanced_level(
                X=residual,
                n_clusters=codebook_size,
                cluster_sizes=cluster_sizes,
                generator=generator,
            )

            self.codebooks.append(codebook)
            self.n_iter_.append(n_iter)
            fit_codes[:, level] = labels.to(torch.int32)

            residual -= codebook[labels]
            tracker.log_metrics(
                {
                    "fit.kmeans_iterations": n_iter,
                    "fit.residual_mse": torch.mean(residual.square()).item(),
                },
                step=level + 1,
            )

        self.fit_codes_ = fit_codes.cpu()

        return self

    @torch.no_grad()
    def get_balanced_semantic_ids(
        self,
        X: pl.Series | pl.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("Сначала вызови fit()")

        X = self.converter(X)
        residual = X.clone()

        codes = torch.empty(
            (len(X), self.num_levels),
            dtype=torch.int32,
            device=self.device,
        )

        for level, centers in enumerate(self.codebooks):
            cluster_sizes = self.get_cluster_sizes(
                n_items=len(X),
                n_clusters=len(centers),
                device=self.device,
            )

            labels, _ = self.balanced_assignment(
                X=residual,
                centers=centers,
                cluster_sizes=cluster_sizes,
            )

            codes[:, level] = labels.to(torch.int32)
            residual -= centers[labels]

        return codes.cpu().numpy()

    def codes_to_dataframe(
        self,
        item_ids: pl.Series,
        codes: np.ndarray | torch.Tensor,
    ) -> pl.DataFrame:
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        result = {"item_id": item_ids}

        for level in range(self.num_levels):
            result[f"sid_{level}"] = codes[:, level]

        return pl.DataFrame(result)

    def transform(
        self,
        items_df: pl.DataFrame,
    ) -> pl.DataFrame:
        codes = self.get_balanced_semantic_ids(items_df)

        return self.codes_to_dataframe(
            item_ids=items_df["item_id"],
            codes=codes,
        )

    def fit_transform(
        self,
        items_df: pl.DataFrame,
        *,
        tracker: ExperimentTracker | None = None,
    ) -> pl.DataFrame:
        self.fit(items_df, tracker=tracker)

        return self.codes_to_dataframe(
            item_ids=items_df["item_id"],
            codes=self.fit_codes_,
        )
