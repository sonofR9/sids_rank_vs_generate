from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral

import torch
import torch.nn.functional as F

from ..training.utils import autocast


class SemanticIDCatalog:
    """Prefix index for semantic IDs that actually exist in the catalog."""

    def __init__(
        self,
        semantic_ids: Iterable[Sequence[int]],
        codebook_sizes: Sequence[int],
    ) -> None:
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        if not self.codebook_sizes or any(size <= 0 for size in self.codebook_sizes):
            raise ValueError("codebook_sizes must contain positive sizes")

        paths: set[tuple[int, ...]] = set()
        for semantic_id in semantic_ids:
            path = tuple(semantic_id)
            if len(path) != len(self.codebook_sizes):
                raise ValueError(
                    "semantic ID length must match the number of codebooks"
                )
            for level, (token, size) in enumerate(
                zip(path, self.codebook_sizes)
            ):
                if not isinstance(token, Integral) or not 0 <= int(token) < size:
                    raise ValueError(
                        f"semantic ID token at level {level} is out of range"
                    )
            paths.add(tuple(int(token) for token in path))

        if not paths:
            raise ValueError("semantic ID catalog must not be empty")

        self.semantic_ids = tuple(sorted(paths))
        self._next_tokens: dict[tuple[int, ...], frozenset[int]] = {}
        mutable_next_tokens: dict[tuple[int, ...], set[int]] = {}
        for path in self.semantic_ids:
            for level, token in enumerate(path):
                mutable_next_tokens.setdefault(path[:level], set()).add(token)
        self._next_tokens = {
            prefix: frozenset(tokens)
            for prefix, tokens in mutable_next_tokens.items()
        }

    def valid_next_tokens(self, prefix: Sequence[int]) -> frozenset[int]:
        """Return tokens that keep ``prefix`` on a path in the catalog."""

        return self._next_tokens.get(tuple(int(token) for token in prefix), frozenset())


def _model_codebook_sizes(model) -> tuple[int, ...]:
    configured = getattr(model, "codebook_sizes", None)
    if configured is None:
        configured = (model.codebook_size,) * model.num_levels
    sizes = tuple(int(size) for size in configured)
    if len(sizes) != model.num_levels:
        raise ValueError("model codebook sizes do not match its number of levels")
    return sizes


def _model_codebook_offsets(model, device: torch.device) -> torch.Tensor:
    configured = getattr(model, "codebook_offsets", None)
    if configured is not None:
        return configured.to(device=device)
    sizes = _model_codebook_sizes(model)
    return torch.tensor([0, *sizes[:-1]], device=device).cumsum(dim=0)


def _encode_history(model, history, positions, mask, user_features=None):
    encode_history = getattr(model, "encode_history", None)
    if callable(encode_history):
        if user_features is not None:
            return encode_history(
                history,
                positions,
                mask,
                user_features=user_features,
            )
        return encode_history(history, positions, mask)
    if user_features is not None:
        raise ValueError("model does not support user profile conditioning")
    return model.encoder(history, positions, mask)


@torch.inference_mode()
def semantic_beam_search(
    model,
    history,
    positions,
    mask,
    beam_size: int = 150,
    use_bf16: bool = False,
    sid_catalog: SemanticIDCatalog | None = None,
    sid_logq_table=None,
    user_features: torch.Tensor | None = None,
):
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    model.eval()
    with autocast(history.device, use_bf16):
        return _semantic_beam_search(
            model,
            history,
            positions,
            mask,
            beam_size,
            sid_catalog,
            sid_logq_table,
            user_features,
        )


def _semantic_beam_search(
    model,
    history,
    positions,
    mask,
    beam_size,
    sid_catalog,
    sid_logq_table,
    user_features,
):
    codebook_sizes = _model_codebook_sizes(model)
    if sid_catalog is not None and sid_catalog.codebook_sizes != codebook_sizes:
        raise ValueError("catalog and model codebook sizes must match")
    if (
        sid_logq_table is not None
        and tuple(sid_logq_table.codebook_sizes) != codebook_sizes
    ):
        raise ValueError("SID LogQ and model codebook sizes must match")

    memory, _ = _encode_history(
        model,
        history,
        positions,
        mask,
        user_features,
    )

    batch_size = history.size(0)
    device = history.device
    codebook_offsets = _model_codebook_offsets(model, device)

    # Изначально один beam с BOS.
    beams = torch.full(
        (batch_size, 1, 1),
        model.bos_id,
        dtype=torch.long,
        device=device,
    )

    scores = torch.zeros(
        batch_size,
        1,
        device=device,
    )

    for level in range(model.num_levels):
        _, current_beam_size, sequence_len = beams.shape

        flat_tokens = beams.reshape(
            batch_size * current_beam_size,
            sequence_len,
        )

        if sequence_len > 1:
            bos = flat_tokens[:, :1]
            sid_tokens = flat_tokens[:, 1:]

            offsets = codebook_offsets[: sid_tokens.size(1)].unsqueeze(0)

            flat_tokens = torch.cat(
                [bos, sid_tokens + offsets],
                dim=1,
            )

        decoder_positions = torch.arange(
            sequence_len,
            device=device,
        ).unsqueeze(0)

        decoder_input = (
            model.decoder_embed(flat_tokens)
            + model.positions_embed(decoder_positions)
        )

        memory_rep = memory.repeat_interleave(
            current_beam_size,
            dim=0,
        )

        mask_rep = mask.repeat_interleave(
            current_beam_size,
            dim=0,
        )

        decoded = model.decoder(
            tgt=decoder_input,
            memory=memory_rep,
            tgt_mask=model.causal_mask(
                sequence_len,
                device,
            ),
            memory_key_padding_mask=~mask_rep,
        )

        logits = model.heads[level](decoded[:, -1])

        beam_prefixes = None
        if sid_logq_table is not None:
            beam_prefixes = beams[:, :, 1:].detach().cpu().tolist()
            prefix_log_q = {}
            for batch_prefixes in beam_prefixes:
                for prefix_values in batch_prefixes:
                    prefix = tuple(int(token) for token in prefix_values)
                    if prefix not in prefix_log_q:
                        prefix_log_q[prefix] = sid_logq_table.log_q_for_prefix(
                            prefix,
                            level=level,
                        )
            level_log_q = torch.stack(
                [
                    prefix_log_q[tuple(int(token) for token in prefix)]
                    for batch_prefixes in beam_prefixes
                    for prefix in batch_prefixes
                ]
            ).to(device=logits.device, dtype=torch.float32)
            if level_log_q.shape != logits.shape:
                raise ValueError("SID LogQ rows must match beam logits")
            logits = logits.float() - level_log_q

        # Beam search uses the same normalized, LogQ-adjusted categorical
        # distribution as the SID loss. The LogQ coefficient is fixed at one.
        log_probs = F.log_softmax(logits, dim=-1)

        log_probs = log_probs.reshape(
            batch_size,
            current_beam_size,
            codebook_sizes[level],
        )

        valid_counts = None
        if sid_catalog is not None:
            if beam_prefixes is None:
                beam_prefixes = beams[:, :, 1:].detach().cpu().tolist()
            valid_mask_cpu = torch.zeros(
                batch_size,
                current_beam_size,
                codebook_sizes[level],
                dtype=torch.bool,
            )
            for batch_index, batch_prefixes in enumerate(beam_prefixes):
                for beam_index, prefix in enumerate(batch_prefixes):
                    valid_tokens = sid_catalog.valid_next_tokens(prefix)
                    if valid_tokens:
                        valid_mask_cpu[
                            batch_index,
                            beam_index,
                            list(valid_tokens),
                        ] = True
            valid_mask = valid_mask_cpu.to(device=device)
            valid_counts = valid_mask.flatten(1).sum(dim=1)
            if torch.any(valid_counts == 0):
                raise RuntimeError("beam search reached a prefix outside the SID catalog")
            log_probs = log_probs.masked_fill(~valid_mask, float("-inf"))

        candidate_scores = (
            scores.unsqueeze(-1) + log_probs
        ).flatten(1)

        next_beam_size = min(
            beam_size,
            candidate_scores.size(1),
        )
        if valid_counts is not None:
            next_beam_size = min(next_beam_size, int(valid_counts.min().item()))

        new_scores, best_indices = torch.topk(
            candidate_scores,
            k=next_beam_size,
            dim=1,
        )

        previous_beam = (
            best_indices // codebook_sizes[level]
        )

        next_token = (
            best_indices % codebook_sizes[level]
        )

        selected_beams = beams.gather(
            dim=1,
            index=previous_beam
            .unsqueeze(-1)
            .expand(-1, -1, sequence_len),
        )

        beams = torch.cat(
            [selected_beams, next_token.unsqueeze(-1)],
            dim=-1,
        )

        scores = new_scores

    return beams[:, :, 1:], scores
