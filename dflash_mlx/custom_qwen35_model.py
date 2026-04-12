# Local fork of MLX Qwen3.5 for DFlash verifier experiments.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_kernel,
    gated_delta_ops,
    gated_delta_update,
)
from mlx_lm.models.qwen3_next import Qwen3NextAttention as Attention
from mlx_lm.models.qwen3_next import Qwen3NextMLP as MLP
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


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
FULL_ATTENTION_VERIFY_COMPILED_FNS: dict[int, Any] = {}
LINEAR_VERIFY_COMPILED_FNS: dict[int, Any] = {}
VERIFY_WITH_ROLLBACK_COMPILED_FNS: dict[tuple[int, tuple[int, ...], int], Any] = {}
ENABLE_EXPLICIT_CACHE_COMPILED_VERIFY = False
ENABLE_LINEAR_LAYER_COMPILED_VERIFY = True


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


def get_compiled_full_attention_verify_fn(layer):
    key = id(layer)
    compiled = FULL_ATTENTION_VERIFY_COMPILED_FNS.get(key)
    if compiled is not None:
        return compiled

    attn = layer.self_attn

    @mx.compile
    def compiled_full_attention_verify(
        hidden_states: mx.array,
        old_keys: mx.array,
        old_values: mx.array,
        offset: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        residual = hidden_states
        inputs = layer.input_layernorm(hidden_states)
        B, L, _ = inputs.shape

        q_proj_output = attn.q_proj(inputs)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, attn.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        new_keys = attn.k_proj(inputs)
        new_values = attn.v_proj(inputs)

        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        new_keys = attn.k_norm(
            new_keys.reshape(B, L, attn.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        new_values = new_values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        queries = attn.rope(queries, offset=offset)
        new_keys = attn.rope(new_keys, offset=offset)

        keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
        values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=attn.scale,
            mask="causal",
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = attn.o_proj(output * mx.sigmoid(gate))

        hidden_states = residual + output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, new_keys, new_values

    FULL_ATTENTION_VERIFY_COMPILED_FNS[key] = compiled_full_attention_verify
    return compiled_full_attention_verify


def forward_full_attention_layer_dflash(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: KVCache | None,
) -> mx.array:
    if (
        cache is not None
        and cache.keys is not None
        and hidden_states.shape[1] > 1
        and mask == "causal"
    ):
        compiled = get_compiled_full_attention_verify_fn(layer)
        hidden_states, new_keys, new_values = compiled(
            hidden_states,
            cache.keys,
            cache.values,
            cache.offset,
        )
        cache.update_and_fetch(new_keys, new_values)
        return hidden_states
    return layer(hidden_states, mask=mask, cache=cache)


def get_compiled_linear_verify_fn(layer):
    key = id(layer)
    compiled = LINEAR_VERIFY_COMPILED_FNS.get(key)
    if compiled is not None:
        return compiled

    # Verification repeatedly evaluates fixed-size draft blocks with warm SSM
    # caches, so compiling the linear-attention layer body avoids rebuilding the
    # same MLX graph while keeping the rollback tensors exact.
    @mx.compile
    def compiled_linear_verify(
        hidden_states: mx.array,
        initial_conv_state: mx.array,
        initial_state: mx.array,
    ) -> tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ]:
        return forward_linear_layer_explicit_with_record(
            layer,
            hidden_states,
            None,
            initial_conv_state,
            initial_state,
        )

    LINEAR_VERIFY_COMPILED_FNS[key] = compiled_linear_verify
    return compiled_linear_verify


def forward_linear_layer_with_rollback_record(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: ArraysCache | None,
) -> tuple[mx.array, dict[str, mx.array]]:
    if (
        ENABLE_LINEAR_LAYER_COMPILED_VERIFY
        and cache is not None
        and cache[0] is not None
        and cache[1] is not None
        and hidden_states.shape[1] > 1
        and mask is None
    ):
        initial_conv_state = cache[0]
        initial_state = cache[1]
        compiled = get_compiled_linear_verify_fn(layer)
        (
            hidden_states,
            new_conv_state,
            new_state,
            qkv,
            keys,
            values,
            g,
            beta,
        ) = compiled(hidden_states, initial_conv_state, initial_state)
        cache[0] = new_conv_state
        cache[1] = new_state
        linear = layer.linear_attn
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


def forward_linear_layer_explicit_with_record(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    initial_conv_state: mx.array,
    initial_state: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
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

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
    new_conv_state = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))

    queries, keys, values = [
        tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        for tensor, num_heads, head_dim in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

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
        out, new_state = gated_delta_kernel(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=initial_state,
            mask=mask,
        )
    else:
        out, new_state = gated_delta_ops(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=initial_state,
            mask=mask,
        )

    out = linear.norm(out, z)
    out = linear.out_proj(out.reshape(batch_size, seq_len, -1))
    hidden_states = residual + out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states, new_conv_state, new_state, qkv, keys, values, g, beta


def forward_full_attention_layer_explicit(
    layer,
    hidden_states: mx.array,
    old_keys: mx.array,
    old_values: mx.array,
    offset: int,
) -> tuple[mx.array, mx.array, mx.array]:
    attn = layer.self_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    batch_size, seq_len, _ = inputs.shape

    q_proj_output = attn.q_proj(inputs)
    queries, gate = mx.split(
        q_proj_output.reshape(batch_size, seq_len, attn.num_attention_heads, -1),
        2,
        axis=-1,
    )
    gate = gate.reshape(batch_size, seq_len, -1)

    new_keys = attn.k_proj(inputs)
    new_values = attn.v_proj(inputs)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    new_keys = attn.k_norm(
        new_keys.reshape(batch_size, seq_len, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    new_values = new_values.reshape(
        batch_size,
        seq_len,
        attn.num_key_value_heads,
        -1,
    ).transpose(0, 2, 1, 3)

    queries = attn.rope(queries, offset=offset)
    new_keys = attn.rope(new_keys, offset=offset)

    keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
    values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn.scale,
        mask="causal",
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = attn.o_proj(output * mx.sigmoid(gate))

    hidden_states = residual + output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states, new_keys, new_values


def get_compiled_verify_with_rollback_fn(
    text_model,
    layer_ids: tuple[int, ...],
    seq_len: int,
):
    key = (id(text_model), layer_ids, seq_len)
    compiled = VERIFY_WITH_ROLLBACK_COMPILED_FNS.get(key)
    if compiled is not None:
        return compiled

    target_layer_ids = set(layer_ids)

    @mx.compile
    def compiled_verify(
        inputs: mx.array,
        fa_offset: int,
        full_keys: list[mx.array],
        full_values: list[mx.array],
        linear_conv_states: list[mx.array],
        linear_states: list[mx.array],
    ):
        hidden_states = text_model.embed_tokens(inputs)
        selected_hidden_states: list[mx.array] = []
        new_full_keys: list[mx.array] = []
        new_full_values: list[mx.array] = []
        new_linear_conv_states: list[mx.array] = []
        new_linear_states: list[mx.array] = []
        record_qkv: list[mx.array] = []
        record_k: list[mx.array] = []
        record_v: list[mx.array] = []
        record_g: list[mx.array] = []
        record_beta: list[mx.array] = []

        full_idx = 0
        linear_idx = 0
        for layer_idx, layer in enumerate(text_model.layers):
            if layer.is_linear:
                (
                    hidden_states,
                    new_conv_state,
                    new_state,
                    qkv,
                    keys,
                    values,
                    g,
                    beta,
                ) = forward_linear_layer_explicit_with_record(
                    layer,
                    hidden_states,
                    None,
                    linear_conv_states[linear_idx],
                    linear_states[linear_idx],
                )
                new_linear_conv_states.append(new_conv_state)
                new_linear_states.append(new_state)
                record_qkv.append(qkv)
                record_k.append(keys)
                record_v.append(values)
                record_g.append(g)
                record_beta.append(beta)
                linear_idx += 1
            else:
                hidden_states, new_keys, new_values = forward_full_attention_layer_explicit(
                    layer,
                    hidden_states,
                    full_keys[full_idx],
                    full_values[full_idx],
                    fa_offset,
                )
                new_full_keys.append(new_keys)
                new_full_values.append(new_values)
                full_idx += 1

            if layer_idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        return (
            text_model.norm(hidden_states),
            mx.concatenate(selected_hidden_states, axis=-1),
            new_full_keys,
            new_full_values,
            new_linear_conv_states,
            new_linear_states,
            mx.concatenate(record_qkv, axis=0),
            mx.concatenate(record_k, axis=0),
            mx.concatenate(record_v, axis=0),
            mx.concatenate(record_g, axis=0),
            mx.concatenate(record_beta, axis=0),
        )

    VERIFY_WITH_ROLLBACK_COMPILED_FNS[key] = compiled_verify
    return compiled_verify


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)

    def forward_dflash(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        input_embeddings: Optional[mx.array] = None,
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        if (
            ENABLE_EXPLICIT_CACHE_COMPILED_VERIFY
            and (
            return_rollback_records
            and input_embeddings is None
            and inputs.shape[0] == 1
            and inputs.shape[1] > 1
            )
        ):
            full_layer_indices = [idx for idx, layer in enumerate(self.layers) if not layer.is_linear]
            linear_layer_indices = [idx for idx, layer in enumerate(self.layers) if layer.is_linear]
            if all(cache[idx].keys is not None and cache[idx].values is not None for idx in full_layer_indices) and all(
                cache[idx][0] is not None and cache[idx][1] is not None for idx in linear_layer_indices
            ):
                compiled = get_compiled_verify_with_rollback_fn(
                    self,
                    tuple(layer_ids),
                    int(inputs.shape[1]),
                )
                full_keys = [cache[idx].keys for idx in full_layer_indices]
                full_values = [cache[idx].values for idx in full_layer_indices]
                linear_initial_conv_states = [cache[idx][0] for idx in linear_layer_indices]
                linear_initial_states = [cache[idx][1] for idx in linear_layer_indices]
                (
                    norm_hidden_states,
                    target_hidden,
                    new_full_keys,
                    new_full_values,
                    new_linear_conv_states,
                    new_linear_states,
                    record_qkv,
                    record_k,
                    record_v,
                    record_g,
                    record_beta,
                ) = compiled(
                    inputs,
                    cache[self.fa_idx].offset,
                    full_keys,
                    full_values,
                    linear_initial_conv_states,
                    linear_initial_states,
                )
                for idx, new_keys, new_values in zip(
                    full_layer_indices,
                    new_full_keys,
                    new_full_values,
                ):
                    cache[idx].update_and_fetch(new_keys, new_values)
                for idx, new_conv_state, new_state in zip(
                    linear_layer_indices,
                    new_linear_conv_states,
                    new_linear_states,
                ):
                    cache[idx][0] = new_conv_state
                    cache[idx][1] = new_state

                rollback_records: dict[int, dict[str, mx.array]] = {}
                for bundle_idx, layer_idx in enumerate(linear_layer_indices):
                    linear = self.layers[layer_idx].linear_attn
                    rollback_records[layer_idx] = {
                        "initial_conv_state": linear_initial_conv_states[bundle_idx],
                        "initial_state": linear_initial_states[bundle_idx],
                        "qkv": record_qkv[bundle_idx : bundle_idx + 1],
                        "k": record_k[bundle_idx : bundle_idx + 1],
                        "v": record_v[bundle_idx : bundle_idx + 1],
                        "g": record_g[bundle_idx : bundle_idx + 1],
                        "beta": record_beta[bundle_idx : bundle_idx + 1],
                        "repeat_factor": linear.num_v_heads // linear.num_k_heads,
                    }
                return norm_hidden_states, target_hidden, rollback_records

        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        selected_hidden_states: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        rollback_records: dict[int, dict[str, mx.array]] = {}

        for idx, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
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
                if layer.is_linear:
                    hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
                else:
                    hidden_states = forward_full_attention_layer_dflash(
                        layer,
                        hidden_states,
                        mask,
                        layer_cache,
                    )
            if idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        norm_hidden_states = self.norm(hidden_states)
        target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
        if return_rollback_records:
            return norm_hidden_states, target_hidden, rollback_records
        return norm_hidden_states, target_hidden

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        snapshots: dict[int, list[mx.array | None]] = {}
        for idx, layer_cache in enumerate(cache):
            if isinstance(layer_cache, ArraysCache):
                snapshots[idx] = [
                    None if value is None else mx.array(value) for value in layer_cache.cache
                ]
        return snapshots

    def restore_linear_caches(
        self,
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

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
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


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def forward_dflash(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        input_embeddings: Optional[mx.array] = None,
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        outputs = self.model.forward_dflash(
            inputs=inputs,
            cache=cache,
            layer_ids=layer_ids,
            input_embeddings=input_embeddings,
            return_rollback_records=return_rollback_records,
        )
        if return_rollback_records:
            norm_hidden_states, target_hidden, rollback_records = outputs
        else:
            norm_hidden_states, target_hidden = outputs

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(norm_hidden_states)
        else:
            logits = self.lm_head(norm_hidden_states)

        if return_rollback_records:
            return logits, target_hidden, rollback_records
        return logits, target_hidden

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return self.model.snapshot_linear_caches(cache)

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        self.model.restore_linear_caches(cache, snapshots)

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        self.model.rollback_linear_caches(cache, rollback_records, accepted_inputs)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_mtp_weights = any("mtp." in k for k in weights)
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def forward_dflash(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        input_embeddings: Optional[mx.array] = None,
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        return self.language_model.forward_dflash(
            inputs=inputs,
            cache=cache,
            layer_ids=layer_ids,
            input_embeddings=input_embeddings,
            return_rollback_records=return_rollback_records,
        )

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return self.language_model.snapshot_linear_caches(cache)

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        self.language_model.restore_linear_caches(cache, snapshots)

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        self.language_model.rollback_linear_caches(cache, rollback_records, accepted_inputs)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
