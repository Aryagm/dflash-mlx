from __future__ import annotations


def test_server_module_importable_and_has_main():
    from dflash_mlx import openai_server

    assert callable(openai_server.main)


def test_chat_messages_join_text_segments_into_prompt():
    from dflash_mlx.openai_server import messages_to_prompt

    prompt = messages_to_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this."},
                    {"type": "text", "text": "Keep it short."},
                ],
            },
        ]
    )

    assert "You are helpful." in prompt
    assert "Summarize this." in prompt
    assert "Keep it short." in prompt


def test_chat_messages_reject_non_text_content_parts():
    from dflash_mlx.openai_server import messages_to_prompt

    try:
        messages_to_prompt(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    ],
                }
            ]
        )
    except ValueError as exc:
        assert "text-only" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for non-text content")


def test_build_chat_response_has_openai_shape():
    from dflash_mlx.openai_server import build_chat_response

    payload = build_chat_response(
        model="local/qwen-dflash",
        content="Hello from DFlash.",
        prompt_tokens=12,
        completion_tokens=4,
    )

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "local/qwen-dflash"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == "Hello from DFlash."
    assert payload["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }


def test_build_chat_stream_chunk_has_openai_shape():
    from dflash_mlx.openai_server import build_chat_stream_chunk

    payload = build_chat_stream_chunk(
        chunk_id="chatcmpl-test",
        created=123,
        model="local/qwen-dflash",
        delta={"content": "Hel"},
    )

    assert payload == {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "local/qwen-dflash",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hel"},
                "finish_reason": None,
            }
        ],
    }


def test_models_response_lists_single_configured_model():
    from dflash_mlx.openai_server import build_models_response

    payload = build_models_response(model_id="local/qwen-dflash")

    assert payload == {
        "object": "list",
        "data": [
            {
                "id": "local/qwen-dflash",
                "object": "model",
                "owned_by": "dflash-mlx",
            }
        ],
    }


def test_health_response_shape():
    from dflash_mlx.openai_server import build_health_response

    assert build_health_response() == {"status": "ok"}
