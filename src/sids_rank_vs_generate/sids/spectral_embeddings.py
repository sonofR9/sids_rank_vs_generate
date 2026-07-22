import numpy as np
import polars as pl
import torch
import scipy.sparse as sp
class ContentGraph():
    def __init__(self, 
                need_normalize: bool = True,
                topk: int = 50,
                metric: str = 'dot_product',
                batch_size: int = 512,
                device: str = 'cuda',
                similarity_threshold: float = 0,
                ):
        self.need_normalize = need_normalize
        self.topk = topk
        self.metric = metric
        self.batch_size = batch_size
        self.device = device
        self.similarity_threshold = similarity_threshold
    @staticmethod
    def converter(X: pl.Series | pl.DataFrame | np.ndarray) -> torch.Tensor:
        if isinstance(X, pl.DataFrame):
            X = X['normalized_embed']
        if isinstance(X, pl.Series):
          X = X.to_list()
        return torch.as_tensor(X, dtype=torch.float32)
    def find_neighbors(self, content_embeddings: pl.Series | pl.DataFrame | np.ndarray):
        X = self.converter(content_embeddings).to(self.device)
        n = len(X)
        all_values = torch.empty((n, self.topk), dtype=torch.float32)
        all_indices = torch.empty((n, self.topk), dtype=torch.long)
        for start in range(0, n, self.batch_size):
            end = min(start+self.batch_size, n)
            batch = X[start:end,:]
            similarity = batch @ X.T
            indices_row = torch.arange(end - start, device = self.device)
            indices_col = torch.arange(start, end, device = self.device)
            similarity[indices_row, indices_col] = float('-inf')
            values, indices = torch.topk(similarity, 
                                         k = self.topk, 
                                         dim = 1, 
                                         largest = True)
            all_values[start:end,:] =  values.to('cpu')
            all_indices[start:end,:] = indices.to('cpu')
            
        return all_values, all_indices
        

    def get_content_graph(self, content_embeddings=None, values=None, indices=None):
        if values is None or indices is None:
            values, indices = self.find_neighbors(content_embeddings)
        n, k = indices.shape    
        rows = np.repeat(np.arange(n), k)
        cols = indices.reshape(-1)
        weights = values.reshape(-1)
        # учитываем только схожие айтемы
        mask = weights > self.similarity_threshold
    
        graph = sp.coo_matrix(
            (weights[mask], (rows[mask], cols[mask])),
            shape=(n, n),
            dtype=np.float32
        ).tocsr()
        
        graph = graph.maximum(graph.T) # у нас может быть такое, что в топ-к какое-то ребро попало для одной вершины, а для другой нет, поэтому так делаю
        graph.setdiag(0)
        graph.eliminate_zeros()
        return graph

class CollaborativeGraph:
    def __init__(self, window_size=5, min_count=2, topk=50, alpha=10):
        self.window_size = window_size
        self.min_count = min_count
        self.topk = topk
        self.alpha = alpha

    # TODO: преобразовать item_id в индексы
    # TODO: собрать отсортированные истории
    # TODO: создать пары внутри окна
    # TODO: посчитать количество пар
    # TODO: преобразовать count в weight
    # TODO: оставить top-k
    # TODO: построить scipy sparse graph