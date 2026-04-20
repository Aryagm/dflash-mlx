from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def build_health_response() -> dict[str, str]:
    return {"status": "ok"}


def build_models_response(model_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "dflash-mlx",
            }
        ],
    }


def build_chat_response(
    *,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_chat_stream_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Unsupported message content format for text-only server.")
            item_type = item.get("type", "text")
            if item_type != "text":
                raise ValueError("This DFlash OpenAI-compatible server is text-only.")
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("Text content parts must include a string 'text' field.")
            parts.append(text)
        return "\n".join(part for part in parts if part)
    raise ValueError("Unsupported message content format for text-only server.")


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ValueError("messages must not be empty")

    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message must include a string role.")
        content = _extract_text_content(message.get("content", ""))
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


@dataclass
class GenerationChunk:
    delta: str
    text: str
    completion_tokens: int
    prompt_tokens: int = 0
    finish_reason: str | None = None
    finished: bool = False


class RunnerProtocol:
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        raise NotImplementedError

    def stream(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError


class DFlashRunner(RunnerProtocol):
    def __init__(
        self,
        *,
        target_model: str,
        draft_model: str,
        speculative_tokens: int | None = None,
        verify_mode: str = "parallel-replay",
        verify_chunk_size: int = 4,
        seed: int = 0,
    ):
        from .api import DFlashGenerator

        self.generator = DFlashGenerator(
            target_model=target_model,
            draft_model=draft_model,
            seed=seed,
        )
        self.speculative_tokens = speculative_tokens
        self.verify_mode = verify_mode
        self.verify_chunk_size = verify_chunk_size

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        result = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=self.speculative_tokens,
            verify_mode=self.verify_mode,
            verify_chunk_size=self.verify_chunk_size,
            skip_special_tokens=True,
        )
        prompt_tokens = int(result.metrics.get("num_input_tokens", 0))
        completion_tokens = len(result.generated_tokens)
        finish_reason = str(result.metrics.get("finish_reason", "stop"))
        if finish_reason == "max_tokens":
            finish_reason = "length"
        return GenerationResult(
            text=result.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def stream(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[GenerationChunk]:
        for event in self.generator.stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=self.speculative_tokens,
            verify_mode=self.verify_mode,
            verify_chunk_size=self.verify_chunk_size,
            skip_special_tokens=True,
        ):
            if event.finished:
                if event.delta:
                    yield GenerationChunk(
                        delta=event.delta,
                        text=event.text,
                        completion_tokens=len(event.generated_tokens),
                    )
                metrics = event.metrics or {}
                prompt_tokens = int(metrics.get("num_input_tokens", 0))
                finish_reason = str(metrics.get("finish_reason", "stop"))
                if finish_reason == "max_tokens":
                    finish_reason = "length"
                yield GenerationChunk(
                    delta="",
                    text=event.text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(event.generated_tokens),
                    finish_reason=finish_reason,
                    finished=True,
                )
            else:
                yield GenerationChunk(
                    delta=event.delta,
                    text=event.text,
                    completion_tokens=len(event.generated_tokens),
                )


@dataclass
class ServerConfig:
    host: str
    port: int
    model_id: str
    runner: RunnerProtocol


def make_handler(config: ServerConfig):
    class OpenAIHandler(BaseHTTPRequestHandler):
        server_version = "dflash-mlx-openai/0.1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_sse_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

        def _write_sse(self, payload: dict[str, Any] | str) -> None:
            data = payload if isinstance(payload, str) else json.dumps(payload)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not raw:
                raise ValueError("Request body is required.")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, build_health_response())
                return
            if self.path == "/v1/models":
                self._send_json(HTTPStatus.OK, build_models_response(config.model_id))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found", "type": "not_found_error"}})

        def _send_streaming_chat(
            self,
            *,
            prompt: str,
            model: str,
            max_new_tokens: int,
            temperature: float,
        ) -> None:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            self._send_sse_headers()
            self._write_sse(
                build_chat_stream_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"role": "assistant"},
                )
            )

            final_chunk: GenerationChunk | None = None
            try:
                for chunk in config.runner.stream(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    if chunk.finished:
                        final_chunk = chunk
                        continue
                    if not chunk.delta:
                        continue
                    self._write_sse(
                        build_chat_stream_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            delta={"content": chunk.delta},
                        )
                    )
                if final_chunk is None:
                    raise RuntimeError("Streaming generation did not produce a final chunk.")
                usage = {
                    "prompt_tokens": final_chunk.prompt_tokens,
                    "completion_tokens": final_chunk.completion_tokens,
                    "total_tokens": final_chunk.prompt_tokens + final_chunk.completion_tokens,
                }
                self._write_sse(
                    build_chat_stream_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={},
                        finish_reason=final_chunk.finish_reason or "stop",
                        usage=usage,
                    )
                )
            except Exception as exc:  # pragma: no cover - sent after headers
                self._write_sse(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "server_error",
                        }
                    }
                )
            finally:
                self._write_sse("[DONE]")

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found", "type": "not_found_error"}})
                return
            try:
                payload = self._read_json()
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("'messages' must be a list.")
                prompt = messages_to_prompt(messages)
                model = str(payload.get("model") or config.model_id)
                max_new_tokens = int(payload.get("max_tokens", 256))
                temperature = float(payload.get("temperature", 0.0))
                if payload.get("stream"):
                    self._send_streaming_chat(
                        prompt=prompt,
                        model=model,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return
                generation = config.runner.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                response = build_chat_response(
                    model=model,
                    content=generation.text,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    finish_reason=generation.finish_reason,
                )
                self._send_json(HTTPStatus.OK, response)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(exc), "type": "invalid_request_error"}},
                )
            except Exception as exc:  # pragma: no cover - safety net for runtime errors
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": str(exc), "type": "server_error"}},
                )

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("[dflash-openai] " + format % args + "\n")

    return OpenAIHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible HTTP server for dflash-mlx.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--model-id", default="dflash-mlx")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--max-speculative-tokens", type=int, default=None)
    parser.add_argument(
        "--verify-mode",
        choices=["stream", "chunked", "parallel-replay", "parallel-lazy-logits", "parallel-greedy-argmax"],
        default="parallel-replay",
    )
    parser.add_argument("--verify-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runner = DFlashRunner(
        target_model=args.target_model,
        draft_model=args.draft_model,
        speculative_tokens=args.max_speculative_tokens,
        verify_mode=args.verify_mode,
        verify_chunk_size=args.verify_chunk_size,
        seed=args.seed,
    )
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        runner=runner,
    )
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    print(f"Serving DFlash OpenAI-compatible API on http://{config.host}:{config.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
