import numpy as np
import polars as pl
import torch
from collections import defaultdict, deque
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
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
        '''
        Можно передать порог, когда объекты считаются схожими, метрику сходстав, топk cоседей.
        '''
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
        

    def get_content_edges(self, content_embeddings: pl.DataFrame):
        # берём исходные индексы
        item_ids = content_embeddings["item_id"].to_numpy()
        values, indices = self.find_neighbors(content_embeddings)
    
        values = values.numpy()
        indices = indices.numpy()
        n, k = indices.shape
        
        # откуда
        rows = np.repeat(item_ids, k)
        # куда
        cols = item_ids[indices.reshape(-1)]
        weights = values.reshape(-1)
        
        # берём только те рёбра, которые свидетельствуют о схожести вершин
        mask = weights > self.similarity_threshold
    
        return pl.DataFrame({
            "src_item_id": rows[mask],
            "dst_item_id": cols[mask],
            "weight": weights[mask],
        })


class CollaborativeGraph:
    def __init__(self, window_size=5, min_count=2, alpha=10):
        self.window_size = window_size
        self.min_count = min_count
        self.alpha = alpha
    def get_collaborative_edges(self, interactions: pl.DataFrame):
        '''
        тут уже мне нужны uid timestamp item_id
        '''
        df = interactions.select(['uid', 'timestamp', 'item_id'])
        # сортируем по user_id и timestamp
        df_sorted = df.sort(['uid', 'timestamp'])
        # проходим по историям пользователей
        inter_dict = defaultdict(int)
        history = deque(maxlen=self.window_size)
        last_uid = None
        # считаем пары в словаре
        for row in df_sorted.iter_rows(named=True):
            cur_uid, cur_iid = row['uid'], row['item_id']
            if cur_uid != last_uid:
                history.clear()
                last_uid = cur_uid
            for last_iid in history:
                if last_iid == cur_iid:
                    continue
                edge = tuple(sorted((last_iid, cur_iid)))
                inter_dict[edge] += 1
            history.append(cur_iid)
    
        rows = []
        cols = []
        weights = []
    
        for key, value in inter_dict.items():
            if value >= self.min_count:
                rows.append(key[0])
                cols.append(key[1])
                weights.append(value / (value + self.alpha)) # мы вообще другую обсуждали, но эта тоже норм, она ограничена + компьюту дешевеле
        return pl.DataFrame({
            "src_item_id": rows,
            "dst_item_id": cols,
            "weight": weights,
        })

class SpectralGraph:
    def __init__(self, alpha=0.5, beta=0.5, graph_dim=128):
        self.alpha = alpha
        self.beta = beta
        self.graph_dim = graph_dim
        self.item_ids = None

    def get_graph(self, content_edges=None, collaborative_edges=None):
        if content_edges is None and collaborative_edges is None:
            raise ValueError("Передай контентные или коллаборативные рёбра")
        edges_list = []
        if content_edges is not None:
            edges_list.append(content_edges)
        if collaborative_edges is not None:
            edges_list.append(collaborative_edges)
        # сюда соберём item_id всех вершин из обоих графов
        item_ids = []
        for edges in edges_list:
            # добавляем вершины, откуда идут рёбра
            item_ids.extend(edges["src_item_id"].to_numpy())
            # добавляем вершины, куда идут рёбра
            item_ids.extend(edges["dst_item_id"].to_numpy())
        # удаляем повторения, сортируем item_id
        # позиция item_id в этом массиве станет индексом вершины
        self.item_ids = np.unique(item_ids)
        # количество вершин итогового графа
        n = len(self.item_ids)
        
        def build_graph(edges):
            # добавим возможность работать с одним типом реберр
            if edges is None or len(edges) == 0:
                return sp.csr_matrix((n, n), dtype=np.float32)
            src = edges["src_item_id"].to_numpy()
            dst = edges["dst_item_id"].to_numpy()
            weights = edges["weight"].to_numpy()

            # переводим реальные item_id в плотные
            rows = np.searchsorted(self.item_ids, src)
            cols = np.searchsorted(self.item_ids, dst)

            graph = sp.coo_matrix(
                (weights, (rows, cols)),
                shape=(n, n),
                dtype=np.float32
            ).tocsr()

            # делаем граф неориентированным
            graph = graph.maximum(graph.T)
            graph.setdiag(0)
            graph.eliminate_zeros()

            return graph

        content_graph = build_graph(content_edges)
        collaborative_graph = build_graph(collaborative_edges)     
        graph = self.alpha * content_graph + self.beta * collaborative_graph
        return graph
        
    def get_embeddings(self, graph):
        # считаем степени вершин
        degree = np.asarray(graph.sum(axis=1)).ravel()
    
        # D^(-1/2)
        degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        mask = degree > 0
        degree_inv_sqrt[mask] = 1 / np.sqrt(degree[mask])
    
        D = sp.diags(degree_inv_sqrt)
    
        # нормализованная матрица смежности
        normalized_graph = D @ graph @ D
        n_components, labels = connected_components(
            normalized_graph,
            directed=False,
            return_labels=True
        )
        print("Вершин:", normalized_graph.shape[0])
        print("Рёбер:", normalized_graph.nnz // 2)
        print("Компонент связности:", n_components)
        
        # берём на один вектор больше, чтобы удалить тривиальный
        k = min(self.graph_dim + 1, graph.shape[0] - 1)
    
        values, vectors = eigsh(
            normalized_graph.astype(np.float32),
            k=k,
            which="LA",
            tol=1e-3
        )
    
        # сортируем от большего собственного числа к меньшему
        order = np.argsort(values)[::-1]
        vectors = vectors[:, order]
    
        # удаляем первый тривиальный вектор
        embeddings = vectors[:, 1:self.graph_dim + 1]
        np.save("reports/experiments/spectral_embeddings.npy", embeddings)
        return pl.DataFrame({
            "item_id": self.item_ids,
            "spectral_embedding": embeddings.tolist()
        })
        
    def fit_transform(self, content_edges=None, collaborative_edges=None):
        graph = self.get_graph(content_edges, collaborative_edges)
        return self.get_embeddings(graph)
