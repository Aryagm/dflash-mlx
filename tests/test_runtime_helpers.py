from __future__ import annotations

from dflash_mlx.adapters import (
    Qwen3TargetAdapter,
    Qwen35TargetAdapter,
    adapter_for_model_type,
)
from dflash_mlx.runtime import (
    generated_token_count,
    longest_prefix_match,
    stop_position,
    trim_draft_cache,
)


class FakeDraftLayerCache:
    def __init__(self) -> None:
        self.trimmed: list[int] = []

    def trim(self, num_tokens: int) -> None:
        self.trimmed.append(num_tokens)


def test_longest_prefix_match_stops_at_first_mismatch() -> None:
    assert longest_prefix_match([1, 2, 3], [1, 2, 4]) == 2
    assert longest_prefix_match([1, 2], [9, 2]) == 0
    assert longest_prefix_match([1, 2, 3], [1, 2, 3, 4]) == 3


def test_stop_position_searches_only_generated_suffix() -> None:
    tokens = [99, 1, 2, 7, 3, 7]
    assert stop_position(tokens, start_idx=2, stop_token_ids={7}) == 3
    assert stop_position(tokens, start_idx=4, stop_token_ids={7}) == 5
    assert stop_position(tokens, start_idx=0, stop_token_ids={42}) is None


def test_generated_token_count_never_goes_negative() -> None:
    assert generated_token_count([10, 11, 12, 13], prompt_len=2) == 2
    assert generated_token_count([10], prompt_len=3) == 0


def test_trim_draft_cache_trims_every_layer() -> None:
    cache = [FakeDraftLayerCache(), FakeDraftLayerCache()]
    trim_draft_cache(cache, num_tokens=16)
    assert [layer.trimmed for layer in cache] == [[16], [16]]


def test_adapter_selection_is_explicit() -> None:
    assert adapter_for_model_type("qwen3") is Qwen3TargetAdapter
    assert adapter_for_model_type("qwen3_5") is Qwen35TargetAdapter
    assert adapter_for_model_type("unknown") is None
