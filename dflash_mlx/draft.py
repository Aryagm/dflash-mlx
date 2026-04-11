from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models import cache as cache_lib
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.utils import quantize_model


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


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
        self.mask_mode = "none"

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
            keys.reshape(
                batch_size,
                context_len + query_len,
                self.n_kv_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size,
            context_len + query_len,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset + context_len)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries, offset=context_len)
            keys = self.rope(keys)

        mask = "causal" if self.mask_mode == "causal" and query_len > 1 else None
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
            args.hidden_size,
            eps=args.rms_norm_eps,
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
        self.attention_mask_mode = "none"

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
            layer.self_attn.mask_mode = self.attention_mask_mode
            hidden_states = layer(hidden_states, target_hidden, cache=layer_cache)
        return self.norm(hidden_states)


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


def maybe_quantize_draft_model(
    draft: DFlashDraftModel,
    bits: int | None,
    group_size: int,
) -> dict[str, Any] | None:
    if bits is None:
        return None
    _, quantized_config = quantize_model(
        model=draft,
        config={},
        group_size=group_size,
        bits=bits,
    )
    mx.eval(draft.parameters())
    return quantized_config.get("quantization")
