import torch
import torch.nn.functional as F


@torch.inference_mode()
def semantic_beam_search(
    model,
    history,
    positions,
    mask,
    beam_size: int = 150,
):
    model.eval()
    memory, _ = model.encoder(
        history,
        positions,
        mask,
    )

    batch_size = history.size(0)
    device = history.device

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

            offsets = (
                torch.arange(
                    sid_tokens.size(1),
                    device=device,
                )
                * model.codebook_size
            ).unsqueeze(0)

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
        log_probs = F.log_softmax(logits, dim=-1)

        log_probs = log_probs.reshape(
            batch_size,
            current_beam_size,
            model.codebook_size,
        )

        candidate_scores = (
            scores.unsqueeze(-1) + log_probs
        ).flatten(1)

        next_beam_size = min(
            beam_size,
            candidate_scores.size(1),
        )

        new_scores, best_indices = torch.topk(
            candidate_scores,
            k=next_beam_size,
            dim=1,
        )

        previous_beam = (
            best_indices // model.codebook_size
        )

        next_token = (
            best_indices % model.codebook_size
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
