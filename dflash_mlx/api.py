from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.generate import wired_limit

from .adapters import LoadedTargetModel, load_target_model
from .draft import DFlashDraftModel, load_draft_model, maybe_quantize_draft_model
from .runtime import dflash_generate, dflash_generate_stream


DEFAULT_TARGET_MODEL = "mlx-community/Qwen3-4B-bf16"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3-4B-DFlash-b16"


@dataclass
class DFlashResult:
    text: str
    output_tokens: list[int]
    generated_tokens: list[int]
    metrics: dict[str, Any]


@dataclass
class DFlashStreamEvent:
    delta: str
    text: str
    token_ids: list[int]
    output_tokens: list[int]
    generated_tokens: list[int]
    metrics: dict[str, Any] | None = None
    finished: bool = False


class DFlashGenerator:
    def __init__(
        self,
        target_model: str = DEFAULT_TARGET_MODEL,
        draft_model: str = DEFAULT_DRAFT_MODEL,
        draft_attention_mask: str = "auto",
        draft_quant_bits: int | None = None,
        draft_quant_group_size: int = 64,
        seed: int = 0,
    ):
        mx.random.seed(seed)
        self.requested_target_model = target_model
        self.requested_draft_model = draft_model
        self.target: LoadedTargetModel = load_target_model(target_model)
        self.draft: DFlashDraftModel
        self.draft, self.draft_path = load_draft_model(draft_model)

        if draft_attention_mask == "auto":
            draft_attention_mask = (
                "causal" if self.target.adapter.family == "qwen3_5" else "none"
            )
        self.draft_attention_mask = draft_attention_mask
        self.draft.attention_mask_mode = draft_attention_mask
        self.draft_quantization = maybe_quantize_draft_model(
            self.draft,
            bits=draft_quant_bits,
            group_size=draft_quant_group_size,
        )

    @property
    def target_model_path(self) -> Path:
        return self.target.resolved_model_path

    def encode_prompt(self, prompt_text: str) -> mx.array:
        return self.target.build_prompt(prompt_text)

    def generate_from_tokens(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "parallel-replay",
        verify_chunk_size: int = 4,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
    ) -> DFlashResult:
        with wired_limit(self.target.model):
            if reset_peak_memory:
                mx.reset_peak_memory()
            output_tokens, metrics = dflash_generate(
                target=self.target,
                draft=self.draft,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=self.target.stop_token_ids(),
                layer_ids=self.draft.target_layer_ids,
                speculative_tokens=speculative_tokens,
                verify_mode=verify_mode,
                verify_chunk_size=verify_chunk_size,
                profile=profile,
            )

        generated_tokens = output_tokens[metrics["num_input_tokens"] :]
        text = self.target.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=skip_special_tokens,
        )
        return DFlashResult(
            text=text,
            output_tokens=output_tokens,
            generated_tokens=generated_tokens,
            metrics=metrics,
        )

    def stream_from_tokens(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "parallel-replay",
        verify_chunk_size: int = 4,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
    ) -> Iterator[DFlashStreamEvent]:
        prompt_len = int(prompt_tokens.shape[0])
        decoded_text = ""
        with wired_limit(self.target.model):
            if reset_peak_memory:
                mx.reset_peak_memory()
            for event in dflash_generate_stream(
                target=self.target,
                draft=self.draft,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=self.target.stop_token_ids(),
                layer_ids=self.draft.target_layer_ids,
                speculative_tokens=speculative_tokens,
                verify_mode=verify_mode,
                verify_chunk_size=verify_chunk_size,
                profile=profile,
            ):
                generated_tokens = event.output_tokens[prompt_len:]
                text = self.target.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=skip_special_tokens,
                )
                if text.startswith(decoded_text):
                    delta = text[len(decoded_text) :]
                else:
                    delta = text
                decoded_text = text
                yield DFlashStreamEvent(
                    delta=delta,
                    text=text,
                    token_ids=list(event.token_ids),
                    output_tokens=list(event.output_tokens),
                    generated_tokens=list(generated_tokens),
                    metrics=event.metrics,
                    finished=event.finished,
                )

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "parallel-replay",
        verify_chunk_size: int = 4,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
    ) -> DFlashResult:
        return self.generate_from_tokens(
            prompt_tokens=self.encode_prompt(prompt_text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            reset_peak_memory=reset_peak_memory,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
        )

    def stream(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        speculative_tokens: int | None = None,
        verify_mode: str = "parallel-replay",
        verify_chunk_size: int = 4,
        reset_peak_memory: bool = True,
        skip_special_tokens: bool = False,
        profile: bool = False,
    ) -> Iterator[DFlashStreamEvent]:
        return self.stream_from_tokens(
            prompt_tokens=self.encode_prompt(prompt_text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            reset_peak_memory=reset_peak_memory,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
        )
