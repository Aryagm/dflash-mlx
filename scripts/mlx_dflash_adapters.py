from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models import cache as cache_lib
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_kernel,
    gated_delta_ops,
)

from prepare_custom_mlx_model import prepare_custom_model


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


def make_gated_delta_state_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto dv_idx = thread_position_in_grid.y;
        constexpr int n_per_t = Dk / 32;

        auto k_ = k + (b_idx * T * Hv + hv_idx) * Dk;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          auto g_t = g_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
            state[i] = state[i] * g_t;
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
          }

          k_ += Hv * Dk;
          v_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_state_update",
        input_names=["k", "v", "g", "beta", "state_in", "T"],
        output_names=["state_out"],
        source=source,
    )


GATED_DELTA_STATE_KERNEL = make_gated_delta_state_kernel()


def advance_gated_delta_states(
    initial_states: mx.array,
    keys: mx.array,
    values: mx.array,
    g: mx.array,
    beta: mx.array,
) -> mx.array:
    if (
        GATED_DELTA_STATE_KERNEL is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        batch_size, _, _, head_dim = keys.shape
        num_v_heads = values.shape[2]
        value_dim = values.shape[-1]
        output = GATED_DELTA_STATE_KERNEL(
            inputs=[keys, values, g, beta, initial_states, keys.shape[1]],
            template=[
                ("InT", initial_states.dtype),
                ("Dk", head_dim),
                ("Dv", value_dim),
                ("Hv", num_v_heads),
            ],
            grid=(32, value_dim, batch_size * num_v_heads),
            threadgroup=(32, 4, 1),
            output_shapes=[initial_states.shape],
            output_dtypes=[initial_states.dtype],
        )
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    state = initial_states.astype(mx.float32)
    keys_f = keys.astype(mx.float32)
    values_f = values.astype(mx.float32)
    g_f = g.astype(mx.float32)
    beta_f = beta.astype(mx.float32)
    for token_idx in range(keys.shape[1]):
        state = state * g_f[:, token_idx, :, None, None]
        kv_mem = mx.sum(state * keys_f[:, token_idx, :, None, :], axis=-1)
        delta = (values_f[:, token_idx] - kv_mem) * beta_f[:, token_idx, :, None]
        state = state + delta[..., None] * keys_f[:, token_idx, :, None, :]
    return state.astype(initial_states.dtype)


def forward_linear_layer_with_rollback_record(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: cache_lib.ArraysCache | None,
) -> tuple[mx.array, dict[str, mx.array]]:
    linear = layer.linear_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    batch_size, seq_len, _ = inputs.shape

    qkv = linear.in_proj_qkv(inputs)
    z = linear.in_proj_z(inputs).reshape(
        batch_size,
        seq_len,
        linear.num_v_heads,
        linear.head_v_dim,
    )
    b = linear.in_proj_b(inputs)
    a = linear.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        initial_conv_state = cache[0]
    else:
        initial_conv_state = mx.zeros(
            (batch_size, linear.conv_kernel_size - 1, linear.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
    if cache is not None:
        cache[0] = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))

    queries, keys, values = [
        tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        for tensor, num_heads, head_dim in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

    state = cache[1] if cache is not None else None
    if state is not None:
        initial_state = state
    else:
        initial_state = mx.zeros(
            (batch_size, linear.num_v_heads, linear.head_v_dim, linear.head_k_dim),
            dtype=inputs.dtype,
        )
    inv_scale = keys.shape[-1] ** -0.5
    queries = (inv_scale**2) * mx.fast.rms_norm(queries, None, 1e-6)
    keys = inv_scale * mx.fast.rms_norm(keys, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(linear.A_log, a, linear.dt_bias)

    use_kernel = (
        not linear.training
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )
    if use_kernel:
        out, state = gated_delta_kernel(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=state,
            mask=mask,
        )
    else:
        out, state = gated_delta_ops(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=state,
            mask=mask,
        )

    if cache is not None:
        cache[1] = state

    out = linear.norm(out, z)
    out = linear.out_proj(out.reshape(batch_size, seq_len, -1))
    hidden_states = residual + out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)

    rollback_record = {
        "initial_conv_state": initial_conv_state,
        "initial_state": initial_state,
        "qkv": qkv,
        "k": keys,
        "v": values,
        "g": g,
        "beta": beta,
        "repeat_factor": linear.num_v_heads // linear.num_k_heads,
    }
    return hidden_states, rollback_record


class MLXTargetAdapter:
    family: str = "unknown"

    def resolve_target_model_path(self, path_or_repo: str) -> Path:
        return resolve_model_path(path_or_repo)

    def build_prompt(self, tokenizer, prompt_text: str) -> mx.array:
        raise NotImplementedError

    def stop_token_ids(self, tokenizer) -> set[int]:
        raise NotImplementedError

    def make_cache(self, model) -> list[Any]:
        return model.make_cache()

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        raise NotImplementedError

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        raise NotImplementedError

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        raise NotImplementedError

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        logits, target_hidden = self.forward_with_hidden_states(
            model,
            inputs,
            cache,
            layer_ids,
            return_rollback_records=False,
        )
        return logits[:, -1:, :], target_hidden

    def snapshot_linear_caches(
        self,
        model,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        raise NotImplementedError

    def restore_linear_caches(
        self,
        model,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        raise NotImplementedError

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        raise NotImplementedError

    def rollback_linear_caches(
        self,
        model,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        raise NotImplementedError

    def cache_summary(self, cache: list[Any]) -> str:
        raise NotImplementedError


class Qwen35TargetAdapter(MLXTargetAdapter):
    family = "qwen3_5"

    def resolve_target_model_path(self, path_or_repo: str) -> Path:
        model_path = resolve_model_path(path_or_repo)
        config = json.loads((model_path / "config.json").read_text())
        if (
            config.get("model_type") == "qwen3_5"
            and config.get("model_file") != "custom_qwen35_dflash_model.py"
        ):
            source_id = path_or_repo if not Path(path_or_repo).exists() else str(model_path)
            return prepare_custom_model(source_id)
        return model_path

    def build_prompt(self, tokenizer, prompt_text: str) -> mx.array:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return mx.array(tokens, dtype=mx.uint32)

    def stop_token_ids(self, tokenizer) -> set[int]:
        eos_token_ids = tokenizer.eos_token_ids
        if isinstance(eos_token_ids, int):
            return {eos_token_ids}
        return set(eos_token_ids)

    def embed_tokens(self, model, tokens: mx.array) -> mx.array:
        return model.language_model.model.embed_tokens(tokens)

    def lm_head_logits(self, model, hidden_states: mx.array) -> mx.array:
        language_model = model.language_model
        text_model = language_model.model
        if language_model.args.tie_word_embeddings:
            return text_model.embed_tokens.as_linear(hidden_states)
        return language_model.lm_head(hidden_states)

    def forward_with_hidden_states(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        if hasattr(model, "forward_dflash"):
            return model.forward_dflash(
                inputs=inputs,
                cache=cache,
                layer_ids=layer_ids,
                return_rollback_records=return_rollback_records,
            )

        language_model = model.language_model
        text_model = language_model.model
        hidden_states = text_model.embed_tokens(inputs)
        fa_mask = qwen3_5.create_attention_mask(hidden_states, cache[text_model.fa_idx])
        ssm_mask = qwen3_5.create_ssm_mask(hidden_states, cache[text_model.ssm_idx])

        selected_hidden_states: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        rollback_records: dict[int, dict[str, mx.array]] = {}
        for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if return_rollback_records and layer.is_linear:
                hidden_states, rollback_record = forward_linear_layer_with_rollback_record(
                    layer,
                    hidden_states,
                    mask,
                    layer_cache,
                )
                rollback_records[idx] = rollback_record
            else:
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        logits = self.lm_head_logits(model, text_model.norm(hidden_states))
        target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
        if return_rollback_records:
            return logits, target_hidden, rollback_records
        return logits, target_hidden

    def forward_accept_all_block(
        self,
        model,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        if hasattr(model, "language_model") and hasattr(
            model.language_model.model,
            "forward_dflash",
        ):
            norm_hidden_states, target_hidden = model.language_model.model.forward_dflash(
                inputs=inputs,
                cache=cache,
                layer_ids=layer_ids,
                return_rollback_records=False,
            )
            return self.lm_head_logits(model, norm_hidden_states[:, -1:, :]), target_hidden

        return super().forward_accept_all_block(model, inputs, cache, layer_ids)

    def snapshot_linear_caches(
        self,
        model,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        if hasattr(model, "snapshot_linear_caches"):
            return model.snapshot_linear_caches(cache)

        snapshots: dict[int, list[mx.array | None]] = {}
        for idx, layer_cache in enumerate(cache):
            if isinstance(layer_cache, cache_lib.ArraysCache):
                snapshots[idx] = [
                    None if value is None else mx.array(value) for value in layer_cache.cache
                ]
        return snapshots

    def restore_linear_caches(
        self,
        model,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        if hasattr(model, "restore_linear_caches"):
            model.restore_linear_caches(cache, snapshots)
            return

        for idx, values in snapshots.items():
            layer_cache = cache[idx]
            layer_cache.cache = [
                None if value is None else mx.array(value) for value in values
            ]
            layer_cache.left_padding = None
            layer_cache.lengths = None

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        for layer_cache in cache:
            if isinstance(layer_cache, cache_lib.KVCache):
                layer_cache.trim(num_tokens)

    def rollback_linear_caches(
        self,
        model,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        if hasattr(model, "rollback_linear_caches"):
            model.rollback_linear_caches(cache, rollback_records, accepted_inputs)
            return

        layer_indices: list[int] = []
        initial_states: list[mx.array] = []
        keys: list[mx.array] = []
        values: list[mx.array] = []
        gs: list[mx.array] = []
        betas: list[mx.array] = []

        for idx, record in rollback_records.items():
            layer_cache = cache[idx]
            initial_conv_state = record["initial_conv_state"]
            qkv = record["qkv"]
            n_keep = initial_conv_state.shape[1]
            conv_prefix = mx.concatenate(
                [initial_conv_state, qkv[:, :accepted_inputs, :]],
                axis=1,
            )
            layer_cache[0] = conv_prefix[:, -n_keep:, :]
            layer_indices.append(idx)
            initial_states.append(record["initial_state"])
            record_keys = record["k"][:, :accepted_inputs]
            repeat_factor = int(record["repeat_factor"])
            if repeat_factor > 1:
                record_keys = mx.repeat(record_keys, repeat_factor, axis=2)
            keys.append(record_keys)
            values.append(record["v"][:, :accepted_inputs])
            gs.append(record["g"][:, :accepted_inputs])
            betas.append(record["beta"][:, :accepted_inputs])

        if not layer_indices:
            return

        rebuilt_states = advance_gated_delta_states(
            initial_states=mx.concatenate(initial_states, axis=0),
            keys=mx.concatenate(keys, axis=0),
            values=mx.concatenate(values, axis=0),
            g=mx.concatenate(gs, axis=0),
            beta=mx.concatenate(betas, axis=0),
        )
        for offset, idx in enumerate(layer_indices):
            cache[idx][1] = rebuilt_states[offset : offset + 1]

    def cache_summary(self, cache: list[Any]) -> str:
        parts: list[str] = []
        for idx, layer_cache in enumerate(cache):
            if isinstance(layer_cache, cache_lib.KVCache):
                parts.append(f"{idx}:kv={layer_cache.offset}")
            elif isinstance(layer_cache, cache_lib.ArraysCache):
                recurrent = None if layer_cache[1] is None else tuple(layer_cache[1].shape)
                parts.append(f"{idx}:ssm={recurrent}")
        return " ".join(parts)


ADAPTERS: dict[str, type[MLXTargetAdapter]] = {
    "qwen3_5": Qwen35TargetAdapter,
}


@dataclass
class LoadedTargetModel:
    requested_model: str
    resolved_model_path: Path
    model: Any
    tokenizer: Any
    adapter: MLXTargetAdapter

    def build_prompt(self, prompt_text: str) -> mx.array:
        return self.adapter.build_prompt(self.tokenizer, prompt_text)

    def stop_token_ids(self) -> set[int]:
        return self.adapter.stop_token_ids(self.tokenizer)

    def make_cache(self) -> list[Any]:
        return self.adapter.make_cache(self.model)

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        return self.adapter.embed_tokens(self.model, tokens)

    def lm_head_logits(self, hidden_states: mx.array) -> mx.array:
        return self.adapter.lm_head_logits(self.model, hidden_states)

    def forward_with_hidden_states(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        return self.adapter.forward_with_hidden_states(
            self.model,
            inputs,
            cache,
            layer_ids,
            return_rollback_records=return_rollback_records,
        )

    def forward_accept_all_block(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
    ) -> tuple[mx.array, mx.array]:
        return self.adapter.forward_accept_all_block(
            self.model,
            inputs,
            cache,
            layer_ids,
        )

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return self.adapter.snapshot_linear_caches(self.model, cache)

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        self.adapter.restore_linear_caches(self.model, cache, snapshots)

    def rewind_kv_caches(self, cache: list[Any], num_tokens: int) -> None:
        self.adapter.rewind_kv_caches(cache, num_tokens)

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        self.adapter.rollback_linear_caches(
            self.model,
            cache,
            rollback_records,
            accepted_inputs,
        )

    def cache_summary(self, cache: list[Any]) -> str:
        return self.adapter.cache_summary(cache)


def load_target_model(path_or_repo: str) -> LoadedTargetModel:
    base_path = resolve_model_path(path_or_repo)
    config = json.loads((base_path / "config.json").read_text())
    model_type = config.get("model_type")
    adapter_cls = ADAPTERS.get(model_type)
    if adapter_cls is None:
        raise NotImplementedError(
            f"No MLX DFlash adapter registered for model_type={model_type!r}"
        )

    adapter = adapter_cls()
    resolved_model_path = adapter.resolve_target_model_path(path_or_repo)
    model, tokenizer = load(str(resolved_model_path))
    return LoadedTargetModel(
        requested_model=path_or_repo,
        resolved_model_path=resolved_model_path,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
    )
