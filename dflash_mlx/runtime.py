from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import time
from typing import Any

import mlx.core as mx

from .adapters import LoadedTargetModel
from .draft import DFlashDraftModel


@dataclass
class DFlashRuntimeEvent:
    token_ids: list[int]
    output_tokens: list[int]
    metrics: dict[str, Any] | None = None
    finished: bool = False


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


def generated_token_count(output_tokens: list[int], prompt_len: int) -> int:
    return max(len(output_tokens) - prompt_len, 0)


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


def profile_start(profile: dict[str, float] | None) -> float:
    return time.perf_counter() if profile is not None else 0.0


def add_profile_elapsed(
    profile: dict[str, float] | None,
    key: str,
    start: float,
) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + time.perf_counter() - start


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
        next_token_tensor = sample_tokens(logits_step[:, -1, :], temperature)
        mx.eval(next_token_tensor, hidden_step)
        next_token = int(next_token_tensor.item())
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
    profile: dict[str, float] | None = None,
) -> tuple[int, int, mx.array]:
    verifier_start = profile_start(profile)
    verifier_logits, verifier_hidden, rollback_records = target.forward_with_hidden_states(
        mx.array(block_tokens, dtype=mx.uint32)[None],
        target_cache,
        layer_ids,
        return_rollback_records=True,
    )

    posterior_tokens = sample_tokens(verifier_logits, temperature)
    # Do not force the full captured hidden block here. The next draft step only
    # needs the accepted prefix, and MLX can prune the rejected suffix lazily.
    mx.eval(posterior_tokens)
    add_profile_elapsed(
        profile,
        "verify_forward_logits_time_s",
        verifier_start,
    )

    prefix_start = profile_start(profile)
    posterior = posterior_tokens[0].tolist()
    matched = longest_prefix_match(block_tokens[1:], posterior[:-1])
    accepted_inputs = matched + 1
    add_profile_elapsed(profile, "verify_prefix_time_s", prefix_start)

    if accepted_inputs < draft_block_size:
        rollback_start = profile_start(profile)
        rollback_tensors = flatten_rollback_tensors(rollback_records)
        if rollback_tensors:
            mx.eval(*rollback_tensors)
        target.rewind_kv_caches(target_cache, draft_block_size - accepted_inputs)
        target.rollback_linear_caches(
            target_cache,
            rollback_records,
            accepted_inputs,
        )
        add_profile_elapsed(profile, "verify_rollback_time_s", rollback_start)

    return accepted_inputs, posterior[matched], verifier_hidden[:, :accepted_inputs, :]


def verify_block_parallel_lazy_logits(
    target: LoadedTargetModel,
    target_cache: list[Any],
    block_tokens: list[int],
    draft_block_size: int,
    temperature: float,
    layer_ids: list[int],
    logit_chunk_size: int,
    profile: dict[str, float] | None = None,
) -> tuple[int, int, mx.array]:
    state_start = profile_start(profile)
    norm_hidden_states, verifier_hidden, rollback_records = target.forward_verifier_states(
        mx.array(block_tokens, dtype=mx.uint32)[None],
        target_cache,
        layer_ids,
    )
    mx.eval(norm_hidden_states)
    add_profile_elapsed(profile, "verify_state_time_s", state_start)

    matched = 0
    accepted_inputs = draft_block_size
    posterior_token: int | None = None
    chunk_size = max(1, logit_chunk_size)

    for chunk_start in range(0, draft_block_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, draft_block_size)
        logit_start = profile_start(profile)
        logits_chunk = target.lm_head_logits(
            norm_hidden_states[:, chunk_start:chunk_end, :]
        )
        posterior_tokens = sample_tokens(logits_chunk, temperature)
        mx.eval(posterior_tokens)
        add_profile_elapsed(profile, "verify_lazy_logits_time_s", logit_start)

        prefix_start = profile_start(profile)
        posterior_chunk = posterior_tokens[0].tolist()

        for local_idx, token in enumerate(posterior_chunk):
            pos = chunk_start + local_idx
            if pos == draft_block_size - 1:
                posterior_token = token
                accepted_inputs = draft_block_size
                break

            if token == block_tokens[pos + 1]:
                matched += 1
                continue

            posterior_token = token
            accepted_inputs = matched + 1
            break

        if posterior_token is not None:
            add_profile_elapsed(
                profile,
                "verify_prefix_time_s",
                prefix_start,
            )
            break
        add_profile_elapsed(profile, "verify_prefix_time_s", prefix_start)

    if posterior_token is None:
        raise RuntimeError("Lazy-logit verifier failed to produce a posterior token.")

    if accepted_inputs < draft_block_size:
        rollback_start = profile_start(profile)
        rollback_tensors = flatten_rollback_tensors(rollback_records)
        if rollback_tensors:
            mx.eval(*rollback_tensors)
        target.rewind_kv_caches(target_cache, draft_block_size - accepted_inputs)
        target.rollback_linear_caches(
            target_cache,
            rollback_records,
            accepted_inputs,
        )
        add_profile_elapsed(profile, "verify_rollback_time_s", rollback_start)

    return accepted_inputs, posterior_token, verifier_hidden[:, :accepted_inputs, :]


def verify_block_parallel_greedy_argmax(
    target: LoadedTargetModel,
    target_cache: list[Any],
    block_tokens: list[int],
    draft_block_size: int,
    temperature: float,
    layer_ids: list[int],
    profile: dict[str, float] | None = None,
) -> tuple[int, int, mx.array]:
    if temperature >= 1e-5:
        raise ValueError("parallel-greedy-argmax only supports temperature=0.")

    state_start = profile_start(profile)
    norm_hidden_states, verifier_hidden, rollback_records = target.forward_verifier_states(
        mx.array(block_tokens, dtype=mx.uint32)[None],
        target_cache,
        layer_ids,
    )
    if profile is not None:
        mx.eval(norm_hidden_states)
    add_profile_elapsed(profile, "verify_state_time_s", state_start)

    argmax_start = profile_start(profile)
    posterior_tokens = target.lm_head_argmax(norm_hidden_states)
    mx.eval(posterior_tokens)
    add_profile_elapsed(profile, "verify_argmax_time_s", argmax_start)

    prefix_start = profile_start(profile)
    posterior = posterior_tokens[0].tolist()
    matched = longest_prefix_match(block_tokens[1:], posterior[:-1])
    accepted_inputs = matched + 1
    add_profile_elapsed(profile, "verify_prefix_time_s", prefix_start)

    if accepted_inputs < draft_block_size:
        rollback_start = profile_start(profile)
        rollback_tensors = flatten_rollback_tensors(rollback_records)
        if rollback_tensors:
            mx.eval(*rollback_tensors)
        target.rewind_kv_caches(target_cache, draft_block_size - accepted_inputs)
        target.rollback_linear_caches(
            target_cache,
            rollback_records,
            accepted_inputs,
        )
        add_profile_elapsed(profile, "verify_rollback_time_s", rollback_start)

    return accepted_inputs, posterior[matched], verifier_hidden[:, :accepted_inputs, :]


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
        posterior_tokens = sample_tokens(chunk_logits, temperature)
        mx.eval(posterior_tokens, chunk_hidden)
        posterior_chunk = posterior_tokens[0].tolist()

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


def dflash_generate_stream(
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
    profile: bool = False,
) -> Iterator[DFlashRuntimeEvent]:
    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    profile_times: dict[str, float] | None = {} if profile else None
    total_max_tokens = int(prompt_tokens.shape[0]) + max_new_tokens
    prompt_len = int(prompt_tokens.shape[0])
    if speculative_tokens is None:
        block_size = draft.block_size
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
    streamed_len = prompt_len
    acceptance_lengths: list[int] = []

    if max_new_tokens > 0:
        yield DFlashRuntimeEvent(
            token_ids=output_tokens[streamed_len:],
            output_tokens=list(output_tokens),
        )
        streamed_len = len(output_tokens)

    decode_start = time.perf_counter()
    while start < total_max_tokens:
        draft_start = profile_start(profile_times)
        block_tokens = [output_tokens[start]] + [draft.mask_token_id] * (block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.embed_tokens(block_input)

        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = target.lm_head_logits(draft_hidden[:, 1:, :])
        drafted_tokens = sample_tokens(draft_logits, temperature)
        mx.eval(drafted_tokens)
        trim_draft_cache(draft_cache, block_size)
        drafted_suffix = drafted_tokens[0].tolist()
        block_tokens[1:] = drafted_suffix[: block_size - 1]
        add_profile_elapsed(profile_times, "draft_time_s", draft_start)

        verify_start = profile_start(profile_times)
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
        elif verify_mode == "parallel-lazy-logits":
            accepted_inputs, posterior_token, verifier_hidden = (
                verify_block_parallel_lazy_logits(
                    target=target,
                    target_cache=target_cache,
                    block_tokens=block_tokens,
                    draft_block_size=block_size,
                    temperature=temperature,
                    layer_ids=layer_ids,
                    logit_chunk_size=verify_chunk_size,
                    profile=profile_times,
                )
            )
        elif verify_mode == "parallel-greedy-argmax":
            accepted_inputs, posterior_token, verifier_hidden = (
                verify_block_parallel_greedy_argmax(
                    target=target,
                    target_cache=target_cache,
                    block_tokens=block_tokens,
                    draft_block_size=block_size,
                    temperature=temperature,
                    layer_ids=layer_ids,
                    profile=profile_times,
                )
            )
        else:
            accepted_inputs, posterior_token, verifier_hidden = verify_block_parallel_replay(
                target=target,
                target_cache=target_cache,
                block_tokens=block_tokens,
                draft_block_size=block_size,
                temperature=temperature,
                layer_ids=layer_ids,
                profile=profile_times,
            )
        acceptance_lengths.append(accepted_inputs)

        target_hidden = verifier_hidden[:, :accepted_inputs, :]
        hidden_start = profile_start(profile_times)
        if profile_times is not None:
            mx.eval(target_hidden)
        add_profile_elapsed(profile_times, "verify_target_hidden_time_s", hidden_start)
        add_profile_elapsed(profile_times, "verify_time_s", verify_start)

        bookkeeping_start = profile_start(profile_times)
        output_tokens = output_tokens[:start]
        output_tokens.extend(block_tokens[:accepted_inputs])
        output_tokens.append(posterior_token)
        start += accepted_inputs
        add_profile_elapsed(
            profile_times,
            "bookkeeping_time_s",
            bookkeeping_start,
        )

        stop_idx = stop_position(output_tokens, prompt_len, stop_token_ids)
        finished = False
        if stop_idx is not None:
            output_tokens = output_tokens[: stop_idx + 1]
            finished = True

        if len(output_tokens) > total_max_tokens:
            output_tokens = output_tokens[:total_max_tokens]
            finished = True

        token_ids = output_tokens[streamed_len:]
        if token_ids:
            yield DFlashRuntimeEvent(
                token_ids=token_ids,
                output_tokens=list(output_tokens),
            )
            streamed_len = len(output_tokens)

        if finished:
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    generated_tokens = generated_token_count(output_tokens, prompt_len)
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
    if profile_times is not None:
        profiled_time = sum(
            profile_times.get(key, 0.0)
            for key in ("draft_time_s", "verify_time_s", "bookkeeping_time_s")
        )
        metrics["profile"] = {
            **profile_times,
            "unattributed_decode_time_s": decode_time - profiled_time,
            "steps": len(acceptance_lengths),
        }
    yield DFlashRuntimeEvent(
        token_ids=[],
        output_tokens=list(output_tokens),
        metrics=metrics,
        finished=True,
    )


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
    profile: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    final_event: DFlashRuntimeEvent | None = None
    for event in dflash_generate_stream(
        target=target,
        draft=draft,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
        layer_ids=layer_ids,
        speculative_tokens=speculative_tokens,
        verify_mode=verify_mode,
        verify_chunk_size=verify_chunk_size,
        profile=profile,
    ):
        if event.finished:
            final_event = event
    if final_event is None or final_event.metrics is None:
        raise RuntimeError("DFlash generation did not produce a final event.")
    return final_event.output_tokens, final_event.metrics
