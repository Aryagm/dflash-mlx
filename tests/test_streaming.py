from __future__ import annotations

from contextlib import nullcontext

import mlx.core as mx


def test_generator_stream_from_tokens_decodes_incremental_deltas(monkeypatch):
    from dflash_mlx.api import DFlashGenerator
    from dflash_mlx.runtime import DFlashRuntimeEvent

    def fake_dflash_generate_stream(**kwargs):
        yield DFlashRuntimeEvent(token_ids=[1], output_tokens=[99, 1])
        yield DFlashRuntimeEvent(token_ids=[2, 3], output_tokens=[99, 1, 2, 3])
        yield DFlashRuntimeEvent(
            token_ids=[],
            output_tokens=[99, 1, 2, 3],
            metrics={"num_input_tokens": 1, "num_output_tokens": 3},
            finished=True,
        )

    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            pieces = {1: "Hel", 2: "lo", 3: "!"}
            return "".join(pieces[token] for token in tokens)

    class FakeTarget:
        model = object()
        tokenizer = FakeTokenizer()

        def stop_token_ids(self):
            return set()

    class FakeDraft:
        target_layer_ids: list[int] = []

    monkeypatch.setattr(
        "dflash_mlx.api.dflash_generate_stream",
        fake_dflash_generate_stream,
    )
    monkeypatch.setattr("dflash_mlx.api.wired_limit", lambda model: nullcontext())

    generator = object.__new__(DFlashGenerator)
    generator.target = FakeTarget()
    generator.draft = FakeDraft()

    events = list(
        generator.stream_from_tokens(
            mx.array([99], dtype=mx.uint32),
            max_new_tokens=3,
        )
    )

    assert [event.delta for event in events] == ["Hel", "lo!", ""]
    assert events[-1].finished is True
    assert events[-1].text == "Hello!"
    assert events[-1].metrics == {"num_input_tokens": 1, "num_output_tokens": 3}
