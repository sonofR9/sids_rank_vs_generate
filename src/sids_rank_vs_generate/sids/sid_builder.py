from sids_rank_vs_generate.sids.rq_kmeans_pipeline import RQKMeansPipeline
from sids_rank_vs_generate.sids.rq_vae_pipeline import RQVAEPipeline
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import PLUMRQVAEPipeline

def build_semantic_ids(
    method,
    embeddings,
    rqkmeans_config=None,
    rqvae_config=None,
    plum_config=None,
    metric_logger=None,
):
    
    if method == "rqkmeans":
        model = RQKMeansPipeline(rqkmeans_config)
    elif method == "rqvae":
        model = RQVAEPipeline(rqvae_config)
    elif method == "plum":
        model = PLUMRQVAEPipeline(plum_config, metric_logger=metric_logger)
    else:
        raise ValueError(f"Не знаю такого метода: {method}")

    model.train(embeddings)
    return model.update(embeddings)
