#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.generate import wired_limit
from mlx_lm.models import cache as cache_lib
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class DraftArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: dict | None = None
    block_size: int = 16
    dflash_config: dict | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DraftArgs":
        keys = {
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "rms_norm_eps",
            "vocab_size",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "head_dim",
            "tie_word_embeddings",
            "attention_bias",
            "attention_dropout",
            "rope_scaling",
            "block_size",
            "dflash_config",
        }
        return cls(**{key: config[key] for key in keys if key in config})


class DFlashAttention(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: cache_lib.KVCache | None = None,
    ) -> mx.array:
        batch_size, query_len, _ = hidden_states.shape
        context_len = target_hidden.shape[1]

        queries = self.q_proj(hidden_states)
        queries = self.q_norm(
            queries.reshape(batch_size, query_len, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

        keys = mx.concatenate(
            [self.k_proj(target_hidden), self.k_proj(hidden_states)],
            axis=1,
        )
        values = mx.concatenate(
            [self.v_proj(target_hidden), self.v_proj(hidden_states)],
            axis=1,
        )
        keys = self.k_norm(
            keys.reshape(batch_size, context_len + query_len, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, context_len + query_len, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset + context_len)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries, offset=context_len)
            keys = self.rope(keys)

        mask = None if query_len == 1 else "causal"
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
        return self.o_proj(output)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: cache_lib.KVCache | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.args = args
        self.layers = [DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.target_layer_ids = list(args.dflash_config["target_layer_ids"])
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = args.block_size
        self.mask_token_id = int(args.dflash_config["mask_token_id"])

    def make_cache(self) -> list[cache_lib.KVCache]:
        return [cache_lib.KVCache() for _ in self.layers]

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: list[cache_lib.KVCache] | None = None,
    ) -> mx.array:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, target_hidden, cache=layer_cache)
        return self.norm(hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal MLX DFlash prototype for Qwen3.5 on Apple Silicon."
    )
    parser.add_argument(
        "--target-model",
        default="mlx-community/Qwen3.5-4B-MLX-bf16",
        help="MLX target model repo or local path.",
    )
    parser.add_argument(
        "--draft-model",
        default="z-lab/Qwen3.5-4B-DFlash",
        help="Hugging Face repo or local path for the DFlash draft weights.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/functional_equation.txt"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--verify-mode",
        choices=["stream", "chunked", "parallel-replay"],
        default="parallel-replay",
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


def load_draft_model(path_or_repo: str) -> tuple[DFlashDraftModel, Path]:
    model_path = resolve_model_path(path_or_repo)
    config = json.loads((model_path / "config.json").read_text())
    draft = DFlashDraftModel(DraftArgs.from_dict(config))

    weights: list[tuple[str, mx.array]] = []
    for weight_file in sorted(model_path.glob("model*.safetensors")):
        weights.extend(mx.load(str(weight_file)).items())
    if not weights:
        raise FileNotFoundError(f"No draft weights found in {model_path}")
    draft.load_weights(weights)
    mx.eval(draft.parameters())
    return draft, model_path


def build_prompt(tokenizer, prompt_text: str) -> mx.array:
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(tokens, dtype=mx.uint32)


def sample_tokens(logits: mx.array, temperature: float) -> mx.array:
    vocab_size = logits.shape[-1]
    flat = logits.reshape(-1, vocab_size)
    if temperature < 1e-5:
        sampled = mx.argmax(flat, axis=-1)
    else:
        sampled = mx.random.categorical(flat / temperature)
    return sampled.reshape(logits.shape[:-1]).astype(mx.uint32)


def lm_head_logits(model, hidden_states: mx.array) -> mx.array:
    language_model = model.language_model
    text_model = language_model.model
    if language_model.args.tie_word_embeddings:
        return text_model.embed_tokens.as_linear(hidden_states)
    return language_model.lm_head(hidden_states)


def forward_target_with_hidden_states(
    model,
    inputs: mx.array,
    cache: list[Any],
    layer_ids: list[int],
) -> tuple[mx.array, mx.array]:
    language_model = model.language_model
    text_model = language_model.model

    hidden_states = text_model.embed_tokens(inputs)
    fa_mask = qwen3_5.create_attention_mask(hidden_states, cache[text_model.fa_idx])
    ssm_mask = qwen3_5.create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

    selected_hidden_states: list[mx.array] = []
    target_layer_ids = set(layer_ids)
    for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        if idx in target_layer_ids:
            selected_hidden_states.append(hidden_states)

    logits = lm_head_logits(model, text_model.norm(hidden_states))
    target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
    return logits, target_hidden


def advance_target_cache(
    target,
    inputs: mx.array,
    cache: list[Any],
) -> mx.array:
    language_model = target.language_model
    text_model = language_model.model

    hidden_states = text_model.embed_tokens(inputs)
    fa_mask = qwen3_5.create_attention_mask(hidden_states, cache[text_model.fa_idx])
    ssm_mask = qwen3_5.create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

    for layer, layer_cache in zip(text_model.layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

    return hidden_states


def snapshot_linear_caches(cache: list[Any]) -> dict[int, list[mx.array | None]]:
    snapshots: dict[int, list[mx.array | None]] = {}
    for idx, layer_cache in enumerate(cache):
        if isinstance(layer_cache, cache_lib.ArraysCache):
            snapshots[idx] = [
                None if value is None else mx.array(value) for value in layer_cache.cache
            ]
    return snapshots


def restore_linear_caches(
    cache: list[Any],
    snapshots: dict[int, list[mx.array | None]],
) -> None:
    for idx, values in snapshots.items():
        layer_cache = cache[idx]
        layer_cache.cache = [
            None if value is None else mx.array(value) for value in values
        ]
        layer_cache.left_padding = None
        layer_cache.lengths = None


def rewind_target_kv_caches(cache: list[Any], num_tokens: int) -> None:
    for layer_cache in cache:
        if isinstance(layer_cache, cache_lib.KVCache):
            layer_cache.trim(num_tokens)


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


def cache_summary(cache: list[Any]) -> str:
    parts: list[str] = []
    for idx, layer_cache in enumerate(cache):
        if isinstance(layer_cache, cache_lib.KVCache):
            parts.append(f"{idx}:kv={layer_cache.offset}")
        elif isinstance(layer_cache, cache_lib.ArraysCache):
            recurrent = None if layer_cache[1] is None else tuple(layer_cache[1].shape)
            parts.append(f"{idx}:ssm={recurrent}")
    return " ".join(parts)


def verify_block_stream(
    target,
    target_cache: list[Any],
    block_tokens: list[int],
    temperature: float,
    layer_ids: list[int],
) -> tuple[int, int, mx.array]:
    verified_hidden_steps: list[mx.array] = []
    for idx, token in enumerate(block_tokens):
        logits_step, hidden_step = forward_target_with_hidden_states(
            target,
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
    target,
    target_cache: list[Any],
    block_tokens: list[int],
    draft_block_size: int,
    temperature: float,
    layer_ids: list[int],
) -> tuple[int, int, mx.array]:
    linear_snapshots = snapshot_linear_caches(target_cache)
    verifier_logits, verifier_hidden = forward_target_with_hidden_states(
        target,
        mx.array(block_tokens, dtype=mx.uint32)[None],
        target_cache,
        layer_ids,
    )
    mx.eval(verifier_logits, verifier_hidden)

    posterior = sample_tokens(verifier_logits, temperature)[0].tolist()
    matched = longest_prefix_match(block_tokens[1:], posterior[:-1])
    accepted_inputs = matched + 1

    if accepted_inputs < draft_block_size:
        rewind_target_kv_caches(target_cache, draft_block_size)
        restore_linear_caches(target_cache, linear_snapshots)
        replay_states = advance_target_cache(
            target,
            mx.array(block_tokens[:accepted_inputs], dtype=mx.uint32)[None],
            target_cache,
        )
        mx.eval(replay_states)

    return accepted_inputs, posterior[matched], verifier_hidden[:, :accepted_inputs, :]


def verify_block_chunked(
    target,
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
        linear_snapshots = snapshot_linear_caches(target_cache)

        chunk_logits, chunk_hidden = forward_target_with_hidden_states(
            target,
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
            return draft_block_size, posterior_chunk[-1], mx.concatenate(verified_hidden_chunks, axis=1)

        accepted_local_inputs = local_matches + 1
        rewind_target_kv_caches(target_cache, len(chunk_tokens))
        restore_linear_caches(target_cache, linear_snapshots)
        replay_hidden = forward_target_with_hidden_states(
            target,
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
    target,
    draft: DFlashDraftModel,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    verify_mode: str,
    verify_chunk_size: int,
) -> tuple[list[int], dict[str, Any]]:
    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    total_max_tokens = int(prompt_tokens.shape[0]) + max_new_tokens
    prompt_len = int(prompt_tokens.shape[0])

    sync_start = time.perf_counter()
    logits, target_hidden = forward_target_with_hidden_states(
        target,
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
        block_tokens = [output_tokens[start]] + [draft.mask_token_id] * (draft.block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.language_model.model.embed_tokens(block_input)

        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        mx.eval(draft_hidden)
        trim_draft_cache(draft_cache, draft.block_size)

        draft_logits = lm_head_logits(target, draft_hidden[:, 1:, :])
        drafted_suffix = sample_tokens(draft_logits, temperature)[0].tolist()
        block_tokens[1:] = drafted_suffix

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
                draft_block_size=draft.block_size,
                temperature=temperature,
                layer_ids=layer_ids,
                verify_chunk_size=verify_chunk_size,
            )
        else:
            accepted_inputs, posterior_token, verifier_hidden = verify_block_parallel_replay(
                target=target,
                target_cache=target_cache,
                block_tokens=block_tokens,
                draft_block_size=draft.block_size,
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
        "target_cache_summary": cache_summary(target_cache),
    }
    return output_tokens, metrics


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)
    prompt_text = args.prompt_file.read_text()

    print(f"[load target] {args.target_model}")
    target, tokenizer = load(args.target_model)
    print(f"[load draft] {args.draft_model}")
    draft, draft_path = load_draft_model(args.draft_model)
    prompt_tokens = build_prompt(tokenizer, prompt_text)
    stop_token_ids = set(tokenizer.eos_token_ids)

    print(
        f"[run] prompt_tokens={prompt_tokens.shape[0]} block_size={draft.block_size} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature}"
    )
    with wired_limit(target):
        for warmup_idx in range(args.warmup_runs):
            _, warm_metrics = dflash_generate(
                target=target,
                draft=draft,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop_token_ids=stop_token_ids,
                layer_ids=draft.target_layer_ids,
                verify_mode=args.verify_mode,
                verify_chunk_size=args.verify_chunk_size,
            )
            print(
                f"[warmup {warmup_idx + 1}/{args.warmup_runs}] "
                f"gen_tps={warm_metrics['generation_tps']:.2f} "
                f"accept={warm_metrics['avg_acceptance_length']:.2f}"
            )

        mx.reset_peak_memory()
        output_tokens, metrics = dflash_generate(
            target=target,
            draft=draft,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_token_ids=stop_token_ids,
            layer_ids=draft.target_layer_ids,
            verify_mode=args.verify_mode,
            verify_chunk_size=args.verify_chunk_size,
        )

    generated_tokens = output_tokens[metrics["num_input_tokens"] :]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print(f"Target model:             {args.target_model}")
    print(f"Draft model:              {draft_path}")
    print(f"Verify mode:              {args.verify_mode}")
    if args.verify_mode == "chunked":
        print(f"Verify chunk size:        {args.verify_chunk_size}")
    print(f"Prompt tokens:            {metrics['num_input_tokens']}")
    print(f"Generated tokens:         {metrics['num_output_tokens']}")
    print(f"Prefill time:             {metrics['prefill_time_s']:.2f}s")
    print(f"Decode time:              {metrics['decode_time_s']:.2f}s")
    print(f"Total time:               {metrics['total_time_s']:.2f}s")
    print(f"Prompt TPS:               {metrics['prompt_tps']:.2f}")
    print(f"Generation TPS:           {metrics['generation_tps']:.2f}")
    print(f"End-to-end TPS:           {metrics['end_to_end_tps']:.2f}")
    print(f"Average acceptance:       {metrics['avg_acceptance_length']:.2f}")
    print(f"Acceptance lengths:       {metrics['acceptance_lengths']}")
    print(f"Peak memory:              {metrics['peak_memory_gb']:.2f} GB")
    print("=" * 60)

    if args.print_output:
        print(output_text)


if __name__ == "__main__":
    main()
