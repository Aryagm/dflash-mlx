"""MLX runtime for DFlash speculative decoding on Apple Silicon."""

from .adapters import LoadedTargetModel, adapter_for_model_type, load_target_model
from .api import DFlashGenerator, DFlashResult, DFlashStreamEvent
from .draft import DFlashDraftModel, load_draft_model
from .runtime import dflash_generate, dflash_generate_stream, longest_prefix_match, sample_tokens
from . import openai_server

__all__ = [
    "DFlashGenerator",
    "DFlashResult",
    "DFlashStreamEvent",
    "DFlashDraftModel",
    "LoadedTargetModel",
    "adapter_for_model_type",
    "dflash_generate",
    "dflash_generate_stream",
    "load_draft_model",
    "load_target_model",
    "longest_prefix_match",
    "sample_tokens",
]
