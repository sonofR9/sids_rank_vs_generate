import numpy as np
import polars as pl

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

from .codebook_sizes import resolve_codebook_sizes

class RQKMeans:
    def __init__(
        self,
        num_levels=None,
        codebook_size=None,
        random_state=42,
        kmeans_batch_size=1024 * 16,
        predict_batch_size=65536,
        codebook_sizes=None,
    ):  
        self.codebook_sizes = resolve_codebook_sizes(
            codebook_sizes=codebook_sizes,
            num_levels=num_levels,
            codebook_size=codebook_size,
        )
        self.num_levels = len(self.codebook_sizes)
        self.random_state = random_state
        self.kmeans_batch_size = kmeans_batch_size
        self.predict_batch_size = predict_batch_size
        self.codebooks = []
        
    @staticmethod
    def converter(X: pl.Series | pl.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            X = X['normalized_embed']
        if isinstance(X, pl.Series):
          X = np.stack(X.to_list())
        return np.asarray(X, dtype=np.float32)
        
    def fit(self,  X: pl.Series | pl.DataFrame | np.ndarray):
        X= self.converter(X)
        residual = X.copy()
        self.codebooks = []
        for current_codebook_size in self.codebook_sizes:
            kmeans = MiniBatchKMeans(
                n_clusters = current_codebook_size,
                batch_size = self.kmeans_batch_size,
                init='k-means++',
                tol=10**(-5),
                random_state=self.random_state
            )
            classes = kmeans.fit_predict(residual)
            codebook = kmeans.cluster_centers_
            residual -= codebook[classes]
            self.codebooks.append(codebook)
        return self

    def get_semantic_ids(self, X) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError('Сначала вызови fit()!')
        X = self.converter(X)
        N = len(X)
        codes_all = np.zeros((N, self.num_levels), dtype=np.int32)

        for start in range(0, N, self.predict_batch_size):
            end = min(start + self.predict_batch_size, N)
            residual = X[start:end].copy()
    
            for level, centers in enumerate(self.codebooks):
                labels = pairwise_distances_argmin(residual, centers)
                residual -= centers[labels]
                codes_all[start:end, level] = labels
    
        return codes_all
    
    def transform(self, items_df: pl.DataFrame) -> pl.DataFrame:
        codes = self.get_semantic_ids(items_df)
        res = {"item_id": items_df["item_id"]}
        for level in range(self.num_levels):
            res[f"sid_{level}"] = codes[:, level]
        return pl.DataFrame(res)
    
    def fit_transform(self, items_df: pl.DataFrame) -> pl.DataFrame:
        self.fit(items_df)
        return self.transform(items_df)
