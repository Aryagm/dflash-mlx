from __future__ import annotations

import time
from typing import Any

import mlx.core as mx

from mlx_dflash_adapters import LoadedTargetModel
from mlx_dflash_draft import DFlashDraftModel


def sample_tokens(logits: mx.array, temperature: float) -> mx.array:
    vocab_size = logits.shape[-1]
    flat = logits.reshape(-1, vocab_size)
    if temperature < 1e-5:
        sampled = mx.argmax(flat, axis=-1)
    else:
        sampled = mx.random.categorical(flat / temperature)
    return sampled.reshape(logits.shape[:-1]).astype(mx.uint32)


def trim_draft_cache(cache: list[Any], num_tokens: int) -> None:
    for layer_cache in cache:
        layer_cache.trim(num_tokens)


def longest_prefix_match(draft_tokens: list[int], verifier_tokens: list[int]) -> int:
    matched = 0
    for draft_token, verifier_token in zip(draft_tokens, verifier_tokens):
        if draft_token != verifier_token:
            break
        matched += 1
    return matched


def stop_position(tokens: list[int], start_idx: int, stop_token_ids: set[int]) -> int | None:
    for idx in range(start_idx, len(tokens)):
        if tokens[idx] in stop_token_ids:
            return idx
    return None


def peak_memory_gb() -> float:
    return mx.get_peak_memory() / 1e9


def flatten_rollback_tensors(rollback_records: Any) -> list[mx.array]:
    if isinstance(rollback_records, dict) and "layer_indices" in rollback_records:
        return [
            tensor
            for tensor in rollback_records.values()
            if hasattr(tensor, "shape") and hasattr(tensor, "dtype")
        ]
    return [
        tensor
        for record in rollback_records.values()
        for key, tensor in record.items()
        if key != "repeat_factor"
    ]


def verify_block_stream(
    target: LoadedTargetModel,
    target_cache: list[Any],
    block_tokens: list[int],
    temperature: float,
    layer_ids: list[int],
) -> tuple[int, int, mx.array]:
    verified_hidden_steps: list[mx.array] = []
    for idx, token in enumerate(block_tokens):
        logits_step, hidden_step = target.forward_with_hidden_states(
            mx.array([[token]], dtype=mx.uint32),
            target_cache,
            layer_ids,
        )
        mx.eval(logits_step, hidden_step)
        next_token = int(sample_tokens(logits_step[:, -1, :], temperature).item())
        verified_hidden_steps.append(hidden_step)

        if idx == len(block_tokens) - 1:
            return len(block_tokens), next_token, mx.concatenate(verified_hidden_steps, axis=1)

        if next_token != block_tokens[idx + 1]:
            return idx + 1, next_token, mx.concatenate(verified_hidden_steps, axis=1)

    raise RuntimeError("Streaming verifier reached an impossible state.")


def verify_block_parallel_replay(
    target: LoadedTargetModel,
    target_cache: list[Any],
    block_tokens: list[int],
    draft_block_size: int,
    temperature: float,
    layer_ids: list[int],
) -> tuple[int, int, mx.array]:
    verifier_logits, verifier_hidden, rollback_records = target.forward_with_hidden_states(
        mx.array(block_tokens, dtype=mx.uint32)[None],
        target_cache,
        layer_ids,
        return_rollback_records=True,
    )
    mx.eval(verifier_logits, verifier_hidden)

    posterior = sample_tokens(verifier_logits, temperature)[0].tolist()
    matched = longest_prefix_match(block_tokens[1:], posterior[:-1])
    accepted_inputs = matched + 1

    if accepted_inputs < draft_block_size:
        rollback_tensors = flatten_rollback_tensors(rollback_records)
        if rollback_tensors:
            mx.eval(*rollback_tensors)
        target.rewind_kv_caches(target_cache, draft_block_size - accepted_inputs)
        target.rollback_linear_caches(
            target_cache,
            rollback_records,
            accepted_inputs,
        )

    return accepted_inputs, posterior[matched], verifier_hidden[:, :accepted_inputs, :]


def verify_block_accept_all(
    target: LoadedTargetModel,
    target_cache: list[Any],
    block_tokens: list[int],
    temperature: float,
    layer_ids: list[int],
) -> tuple[int, int, mx.array]:
    # Unsafe speed-first mode: trust every drafted token, but still run the
    # target over the accepted block to keep target-side features/cache aligned.
    posterior_logits, verifier_hidden = target.forward_accept_all_block(
        mx.array(block_tokens, dtype=mx.uint32)[None],
        target_cache,
        layer_ids,
    )
    mx.eval(posterior_logits, verifier_hidden)
    posterior_token = int(sample_tokens(posterior_logits[:, -1, :], temperature).item())
    return len(block_tokens), posterior_token, verifier_hidden


def verify_block_chunked(
    target: LoadedTargetModel,
    target_cache: list[Any],
    block_tokens: list[int],
    draft_block_size: int,
    temperature: float,
    layer_ids: list[int],
    verify_chunk_size: int,
) -> tuple[int, int, mx.array]:
    verified_hidden_chunks: list[mx.array] = []
    cursor = 0

    while cursor < draft_block_size:
        chunk_end = min(cursor + verify_chunk_size, draft_block_size)
        chunk_tokens = block_tokens[cursor:chunk_end]
        linear_snapshots = target.snapshot_linear_caches(target_cache)

        chunk_logits, chunk_hidden = target.forward_with_hidden_states(
            mx.array(chunk_tokens, dtype=mx.uint32)[None],
            target_cache,
            layer_ids,
        )
        mx.eval(chunk_logits, chunk_hidden)
        posterior_chunk = sample_tokens(chunk_logits, temperature)[0].tolist()

        max_compare = min(len(chunk_tokens), draft_block_size - cursor - 1)
        local_matches = 0
        while local_matches < max_compare:
            if posterior_chunk[local_matches] != block_tokens[cursor + local_matches + 1]:
                break
            local_matches += 1

        if local_matches == max_compare and chunk_end < draft_block_size:
            verified_hidden_chunks.append(chunk_hidden)
            cursor = chunk_end
            continue

        if local_matches == max_compare and chunk_end == draft_block_size:
            verified_hidden_chunks.append(chunk_hidden)
            return (
                draft_block_size,
                posterior_chunk[-1],
                mx.concatenate(verified_hidden_chunks, axis=1),
            )

        accepted_local_inputs = local_matches + 1
        target.rewind_kv_caches(target_cache, len(chunk_tokens))
        target.restore_linear_caches(target_cache, linear_snapshots)
        replay_hidden = target.forward_with_hidden_states(
            mx.array(chunk_tokens[:accepted_local_inputs], dtype=mx.uint32)[None],
            target_cache,
            layer_ids,
        )[1]
        mx.eval(replay_hidden)
        verified_hidden_chunks.append(replay_hidden)
        return (
            cursor + accepted_local_inputs,
            posterior_chunk[local_matches],
            mx.concatenate(verified_hidden_chunks, axis=1),
        )

    raise RuntimeError("Chunked verifier reached an impossible state.")


def dflash_generate(
    target: LoadedTargetModel,
    draft: DFlashDraftModel,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    speculative_tokens: int | None,
    verify_mode: str,
    verify_chunk_size: int,
) -> tuple[list[int], dict[str, Any]]:
    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    total_max_tokens = int(prompt_tokens.shape[0]) + max_new_tokens
    prompt_len = int(prompt_tokens.shape[0])
    if speculative_tokens is None:
        block_size = draft.block_size
    elif verify_mode == "accept-all":
        block_size = max(1, speculative_tokens)
    else:
        block_size = max(1, min(speculative_tokens, draft.block_size))

    sync_start = time.perf_counter()
    logits, target_hidden = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits, target_hidden)
    prefill_time = time.perf_counter() - sync_start

    output_tokens = prompt_tokens.tolist() + [first_token]
    start = prompt_len
    acceptance_lengths: list[int] = []

    decode_start = time.perf_counter()
    while start < total_max_tokens:
        block_tokens = [output_tokens[start]] + [draft.mask_token_id] * (block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.embed_tokens(block_input)

        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        mx.eval(draft_hidden)
        trim_draft_cache(draft_cache, block_size)

        draft_logits = target.lm_head_logits(draft_hidden[:, 1:, :])
        drafted_suffix = sample_tokens(draft_logits, temperature)[0].tolist()
        block_tokens[1:] = drafted_suffix[: block_size - 1]

        if verify_mode == "stream":
            accepted_inputs, posterior_token, verifier_hidden = verify_block_stream(
                target=target,
                target_cache=target_cache,
                block_tokens=block_tokens,
                temperature=temperature,
                layer_ids=layer_ids,
            )
        elif verify_mode == "chunked":
            accepted_inputs, posterior_token, verifier_hidden = verify_block_chunked(
                target=target,
                target_cache=target_cache,
                block_tokens=block_tokens,
                draft_block_size=block_size,
                temperature=temperature,
                layer_ids=layer_ids,
                verify_chunk_size=verify_chunk_size,
            )
        elif verify_mode == "accept-all":
            accepted_inputs, posterior_token, verifier_hidden = verify_block_accept_all(
                target=target,
                target_cache=target_cache,
                block_tokens=block_tokens,
                temperature=temperature,
                layer_ids=layer_ids,
            )
        else:
            accepted_inputs, posterior_token, verifier_hidden = verify_block_parallel_replay(
                target=target,
                target_cache=target_cache,
                block_tokens=block_tokens,
                draft_block_size=block_size,
                temperature=temperature,
                layer_ids=layer_ids,
            )
        acceptance_lengths.append(accepted_inputs)

        target_hidden = verifier_hidden[:, :accepted_inputs, :]

        output_tokens = output_tokens[:start]
        output_tokens.extend(block_tokens[:accepted_inputs])
        output_tokens.append(posterior_token)
        start += accepted_inputs

        stop_idx = stop_position(output_tokens, prompt_len, stop_token_ids)
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            break

        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    generated_tokens = max(len(output_tokens) - prompt_len, 0)
    total_time = prefill_time + decode_time

    metrics = {
        "num_input_tokens": prompt_len,
        "num_output_tokens": generated_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "prompt_tps": prompt_len / max(prefill_time, 1e-9),
        "generation_tps": generated_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": generated_tokens / max(total_time, 1e-9),
        "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
        "acceptance_lengths": acceptance_lengths,
        "peak_memory_gb": peak_memory_gb(),
        "target_cache_summary": target.cache_summary(target_cache),
        "speculative_tokens": block_size,
    }
    return output_tokens, metrics
