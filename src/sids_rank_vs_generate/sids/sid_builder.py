from sids_rank_vs_generate.sids.rq_kmeans_pipeline import RQKMeansPipeline
from sids_rank_vs_generate.sids.rq_vae_pipeline import RQVAEPipeline


def build_semantic_ids(
    method,
    embeddings,
    rqkmeans_config,
    rqvae_config,
):
    
    if method == "rqkmeans":
        model = RQKMeansPipeline(rqkmeans_config)
    elif method == "rqvae":
        model = RQVAEPipeline(rqvae_config)
    else:
        raise ValueError(f"Не знаю такого метода: {method}")

    model.train(embeddings)
    return model.update(embeddings)