"""Оценка native DIGER checkpoint без повторного обучения"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


DIGER_COMMIT = "cd72dcdc25d28e0acfa5aea14788ad863b41ab4f"


def parse_cutoffs(value: str) -> tuple[int, ...]:
    """Разобрать возрастающие cutoffs"""

    cutoffs = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    return cutoffs


class CatalogIndex:
    """Индекс полных SID и допустимых префиксов"""

    def __init__(self, codes: Sequence[Sequence[int]]) -> None:
        if len(codes) < 2:
            raise ValueError("DIGER code table must contain padding and items")
        code_length = len(codes[1])
        if code_length <= 0:
            raise ValueError("DIGER codes must not be empty")

        self.code_length = code_length
        self.sid_to_items: dict[tuple[int, ...], list[int]] = defaultdict(list)
        next_tokens: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for item_id, values in enumerate(codes[1:], start=1):
            sid = tuple(int(value) for value in values)
            if len(sid) != code_length or any(value < 0 for value in sid):
                raise ValueError(f"invalid DIGER SID for item {item_id}")
            self.sid_to_items[sid].append(item_id)
            for level, token in enumerate(sid):
                next_tokens[sid[:level]].add(token)
        self.next_tokens = {
            prefix: tuple(sorted(tokens)) for prefix, tokens in next_tokens.items()
        }
        self.item_count = len(codes) - 1

    def allowed_tokens(self, _batch_id: int, decoder_ids: torch.Tensor) -> list[int]:
        """Вернуть продолжения каталожного префикса"""

        prefix = tuple(int(value) for value in decoder_ids.tolist()[1:])
        tokens = self.next_tokens.get(prefix)
        if not tokens:
            raise RuntimeError(f"DIGER beam left the SID catalog at prefix {prefix}")
        return list(tokens)

    def ranked_items(
        self,
        predicted_sids: Iterable[Sequence[int]],
        maximum: int,
    ) -> list[int]:
        """Развернуть beams в уникальные айтемы"""

        ranked: list[int] = []
        seen: set[int] = set()
        for values in predicted_sids:
            for item_id in self.sid_to_items.get(tuple(int(v) for v in values), ()):
                if item_id in seen:
                    continue
                seen.add(item_id)
                ranked.append(item_id)
                if len(ranked) >= maximum:
                    return ranked
        return ranked


def batch_metric_sums(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    catalog: CatalogIndex,
    cutoffs: Sequence[int],
) -> tuple[dict[str, float], set[int]]:
    """Посчитать суммы метрик одного batch"""

    if outputs.ndim != 3 or labels.ndim != 2:
        raise ValueError("unexpected DIGER output or label shape")
    if outputs.size(0) != labels.size(0) or outputs.size(2) != labels.size(1):
        raise ValueError("DIGER outputs and labels are incompatible")
    if max(cutoffs) > outputs.size(1):
        raise ValueError("largest cutoff exceeds returned beam count")

    totals: dict[str, float] = defaultdict(float)
    recommended: set[int] = set()
    maximum = max(cutoffs)
    for predicted, label in zip(outputs, labels, strict=True):
        matches = predicted.eq(label.unsqueeze(0)).all(dim=1)
        totals["top1_token_correct"] += float(predicted[0].eq(label).sum().item())
        totals["token_count"] += float(label.numel())
        totals["valid_sid_count"] += float(
            sum(tuple(row.tolist()) in catalog.sid_to_items for row in predicted)
        )
        totals["generated_sid_count"] += float(predicted.size(0))

        target_sid_ranks = torch.nonzero(matches, as_tuple=False).flatten()
        sid_rank = int(target_sid_ranks[0].item()) if len(target_sid_ranks) else None
        for cutoff in cutoffs:
            hit = sid_rank is not None and sid_rank < cutoff
            totals[f"sid_recall@{cutoff}"] += float(hit)
            if hit:
                totals[f"sid_mrr@{cutoff}"] += 1.0 / (sid_rank + 1)

        ranked_items = catalog.ranked_items(predicted.tolist(), maximum)
        recommended.update(ranked_items)
        totals[f"recommendation_count@{maximum}"] += float(len(ranked_items))
        target_items = catalog.sid_to_items.get(tuple(int(v) for v in label.tolist()), [])
        target_item = target_items[0] if len(target_items) == 1 else None
        item_rank = (
            ranked_items.index(target_item)
            if target_item is not None and target_item in ranked_items
            else None
        )
        for cutoff in cutoffs:
            hit = item_rank is not None and item_rank < cutoff
            totals[f"item_recall@{cutoff}"] += float(hit)
            if hit:
                totals[f"item_mrr@{cutoff}"] += 1.0 / (item_rank + 1)
                totals[f"item_ndcg@{cutoff}"] += 1.0 / math.log2(item_rank + 2)
    return dict(totals), recommended


def _merge_sums(target: dict[str, float], values: dict[str, float]) -> None:
    for name, value in values.items():
        target[name] = target.get(name, 0.0) + float(value)


@torch.inference_mode()
def catalog_beam_search(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    catalog: CatalogIndex,
    *,
    num_beams: int,
) -> torch.Tensor:
    """Авторский beam search с маской каталога"""

    batch_size = input_ids.size(0)
    (
        expanded_ids,
        expanded_mask,
        decoder_ids,
        beam_scores,
        beam_idx_offset,
    ) = model.prepare_beam_search_inputs(
        input_ids,
        attention_mask,
        batch_size,
        num_beams,
    )
    inputs_embeds = model.get_input_embeddings(expanded_ids, expanded_mask)
    encoder_outputs = model.get_encoder()(
        inputs_embeds=inputs_embeds,
        attention_mask=expanded_mask,
        return_dict=True,
    )

    while decoder_ids.size(1) < catalog.code_length + 1:
        outputs = model.forward(
            encoder_outputs=encoder_outputs,
            attention_mask=expanded_mask,
            decoder_input_ids=decoder_ids,
        )
        logits = outputs.logits
        valid_mask = torch.zeros(
            decoder_ids.size(0),
            logits.size(-1),
            dtype=torch.bool,
            device=logits.device,
        )
        for row_index, row in enumerate(decoder_ids.detach().cpu()):
            valid_mask[row_index, catalog.allowed_tokens(row_index, row)] = True
        logits[:, -1, :] = logits[:, -1, :].masked_fill(
            ~valid_mask,
            float("-inf"),
        )
        decoder_ids, beam_scores = model.beam_search_step(
            logits,
            decoder_ids,
            beam_scores,
            beam_idx_offset,
            batch_size,
            num_beams,
        )
    return decoder_ids.view(batch_size, num_beams, -1)[:, :, 1:]


@torch.inference_mode()
def evaluate_checkpoint(
    model: Any,
    data_loader: DataLoader,
    codes: Sequence[Sequence[int]],
    *,
    device: torch.device,
    num_beams: int,
    cutoffs: Sequence[int],
    constrain_to_catalog: bool,
) -> dict[str, float]:
    """Оценить native генератор на одном split"""

    if num_beams < max(cutoffs):
        raise ValueError("num_beams must cover every requested cutoff")
    catalog = CatalogIndex(codes)
    model.to(device)
    model.device = device
    model.num_beams = num_beams
    model.eval()
    item_codes = torch.tensor(codes, dtype=torch.long, device=device)
    totals: dict[str, float] = {}
    recommended_items: set[int] = set()
    samples = 0

    for batch in tqdm(data_loader, desc="Evaluate native DIGER", ncols=100):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        batch_size = input_ids.size(0)
        encoded_history = item_codes[input_ids].reshape(batch_size, -1)
        target_sids = item_codes[targets].reshape(batch_size, -1)
        attention_mask = encoded_history.ne(-1)
        if constrain_to_catalog:
            outputs = catalog_beam_search(
                model,
                encoded_history,
                attention_mask,
                catalog,
                num_beams=num_beams,
            )
        else:
            outputs = model.generate(
                input_ids=encoded_history,
                attention_mask=attention_mask,
                n_return_sequences=num_beams,
            )
        sums, recommended = batch_metric_sums(
            outputs.detach().cpu(),
            target_sids.detach().cpu(),
            catalog,
            cutoffs,
        )
        _merge_sums(totals, sums)
        recommended_items.update(recommended)
        samples += batch_size

    if samples == 0:
        raise ValueError("DIGER evaluation split is empty")
    helper_sums = {
        "top1_token_correct",
        "token_count",
        "valid_sid_count",
        "generated_sid_count",
        f"recommendation_count@{max(cutoffs)}",
    }
    metrics = {
        name: value / samples
        for name, value in totals.items()
        if name not in helper_sums
    }
    metrics["top1_token_accuracy"] = totals["top1_token_correct"] / max(
        totals["token_count"], 1.0
    )
    metrics["valid_sid_rate"] = totals["valid_sid_count"] / max(
        totals["generated_sid_count"], 1.0
    )
    metrics[f"mean_recommendations@{max(cutoffs)}"] = totals[
        f"recommendation_count@{max(cutoffs)}"
    ] / samples
    metrics[f"catalog_coverage@{max(cutoffs)}"] = len(recommended_items) / max(
        catalog.item_count, 1
    )
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(checkpoint: Path) -> dict[str, Path]:
    files = {
        "model": checkpoint,
        "rqvae": Path(f"{checkpoint}.rqvae"),
        "codes": checkpoint.with_suffix(".code.json"),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing DIGER checkpoint files: " + ", ".join(missing))
    return files


def _load_state(path: Path, load_torch_checkpoint: Any) -> dict[str, Any]:
    state = load_torch_checkpoint(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"expected state dict in {path}")
    return state


def _strict_load(model: Any, state: dict[str, Any], path: Path) -> None:
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"checkpoint is incompatible with {path}\n{error}") from error


def _build_model(
    config: dict[str, Any],
    *,
    num_items: int,
    files: dict[str, Path],
    model_class: Any,
    rqvae_class: Any,
    load_torch_checkpoint: Any,
) -> Any:
    from transformers import T5Config, T5ForConditionalGeneration

    t5_config = T5Config(
        num_layers=config["encoder_layers"],
        num_decoder_layers=config["decoder_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        d_kv=config["d_kv"],
        dropout_rate=config["dropout_rate"],
        activation_function=config["activation_function"],
        vocab_size=1,
        pad_token_id=0,
        eos_token_id=300,
        decoder_start_token_id=0,
        feed_forward_proj=config["feed_forward_proj"],
        n_positions=config["max_length"],
    )
    recommender = model_class(
        config=config,
        model=T5ForConditionalGeneration(config=t5_config),
        n_items=num_items,
        code_length=config["code_length"],
        code_number=config["code_num"],
    )
    model_state = _load_state(files["model"], load_torch_checkpoint)
    _strict_load(recommender, model_state, files["model"])

    rqvae_state = _load_state(files["rqvae"], load_torch_checkpoint)
    rqvae_config = dict(config)
    rqvae_config["use_learnable_sigma_gumbel"] = any(
        "auto_sigma_module" in name for name in rqvae_state
    )
    rqvae = rqvae_class(config=rqvae_config, in_dim=recommender.semantic_hidden_size)
    _strict_load(rqvae, rqvae_state, files["rqvae"])
    return recommender


def _validate_source(source: Path) -> str:
    if not (source / ".git").is_dir():
        raise FileNotFoundError(f"DIGER source is not a git checkout: {source}")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != DIGER_COMMIT:
        raise RuntimeError(f"expected DIGER commit {DIGER_COMMIT}, found {commit}")
    return commit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=150)
    parser.add_argument("--cutoffs", default="1,5,10,50,100")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--constrain-to-catalog", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.num_beams <= 0:
        raise ValueError("invalid DIGER evaluation settings")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max_examples must be positive")
    cutoffs = parse_cutoffs(args.cutoffs)
    if max(cutoffs) > args.num_beams:
        raise ValueError("largest cutoff exceeds num_beams")

    source = args.source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    commit = _validate_source(source)
    sys.path.insert(0, str(source))

    import yaml
    from data import Collator, SequentialSplitDataset, load_split_data
    from model import Model
    from utils import load_torch_checkpoint
    from vq import RQVAE

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data_path"] = str(Path(config["data_path"]).expanduser().resolve())
    config["eval_batch_size"] = args.eval_batch_size
    config["num_workers"] = args.num_workers
    config["num_beams"] = args.num_beams

    _, num_items, _, valid, test = load_split_data(config)
    split_sequences = valid if args.split == "valid" else test
    if args.max_examples is not None:
        split_sequences = split_sequences[: args.max_examples]
    dataset = SequentialSplitDataset(config, num_items, split_sequences)
    collator = Collator(
        eos_token_id=-1,
        pad_token_id=0,
        max_length=config["max_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collator,
    )
    files = _checkpoint_files(checkpoint)
    model = _build_model(
        config,
        num_items=num_items,
        files=files,
        model_class=Model,
        rqvae_class=RQVAE,
        load_torch_checkpoint=load_torch_checkpoint,
    )
    codes = json.loads(files["codes"].read_text(encoding="utf-8"))
    if len(codes) != num_items:
        raise ValueError(f"DIGER code table has {len(codes)} rows, expected {num_items}")
    if any(len(row) != config["code_length"] for row in codes):
        raise ValueError("DIGER code table has an unexpected code length")

    started = time.monotonic()
    metrics = evaluate_checkpoint(
        model,
        loader,
        codes,
        device=torch.device(args.device),
        num_beams=args.num_beams,
        cutoffs=cutoffs,
        constrain_to_catalog=args.constrain_to_catalog,
    )
    result = {
        "schema_version": 1,
        "reference_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": {name: _sha256(path) for name, path in files.items()},
        "config": str(config_path),
        "split": args.split,
        "samples": len(dataset),
        "num_beams": args.num_beams,
        "cutoffs": list(cutoffs),
        "constrain_to_catalog": args.constrain_to_catalog,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": metrics,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
