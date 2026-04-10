#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


_ORIGINAL_QWEN3_5_GDN_FORWARD = Qwen3_5GatedDeltaNet.forward
_QWEN3_5_CACHE_PATCHED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DFlash prototype for macOS using PyTorch MPS.")
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3.5-4B-DFlash")
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/functional_equation.txt"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--verify-mode", choices=["full", "cached"], default="full")
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def patch_qwen3_5_multitoken_cache() -> None:
    global _QWEN3_5_CACHE_PATCHED
    if _QWEN3_5_CACHE_PATCHED:
        return

    def patched_forward(self, hidden_states, cache_params=None, attention_mask=None):
        if cache_params is not None and hidden_states.shape[1] > 1 and cache_params.has_previous_state(self.layer_idx):
            # Qwen3.5's stock cached path only handles seq_len == 1 correctly for linear-attention layers.
            # Step through the verification block token-by-token so recurrent/conv state is updated causally.
            outputs = []
            state_history = []
            record_history = bool(getattr(cache_params, "_dflash_record_history", False))
            for i in range(hidden_states.shape[1]):
                mask_i = None
                if attention_mask is not None:
                    mask_i = attention_mask[:, i : i + 1]
                outputs.append(
                    _ORIGINAL_QWEN3_5_GDN_FORWARD(
                        self,
                        hidden_states[:, i : i + 1, :],
                        cache_params=cache_params,
                        attention_mask=mask_i,
                    )
                )
                if record_history:
                    layer_cache = cache_params.layers[self.layer_idx]
                    conv_state = None
                    recurrent_state = None
                    if layer_cache.conv_states is not None:
                        conv_state = layer_cache.conv_states.clone()
                    if layer_cache.recurrent_states is not None:
                        recurrent_state = layer_cache.recurrent_states.clone()
                    state_history.append((conv_state, recurrent_state))
            if record_history:
                cache_params.layers[self.layer_idx]._dflash_state_history = state_history
            return torch.cat(outputs, dim=1)
        return _ORIGINAL_QWEN3_5_GDN_FORWARD(
            self,
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
        )

    Qwen3_5GatedDeltaNet.forward = patched_forward
    _QWEN3_5_CACHE_PATCHED = True


def restore_qwen3_5_linear_cache_prefix(target_cache: DynamicCache, accepted_tokens: int) -> None:
    for layer in target_cache.layers:
        state_history = getattr(layer, "_dflash_state_history", None)
        if state_history is None:
            continue
        conv_state, recurrent_state = state_history[accepted_tokens - 1]
        if conv_state is not None:
            layer.conv_states.copy_(conv_state)
            layer.has_previous_state = True
        if recurrent_state is not None:
            layer.recurrent_states.copy_(recurrent_state)
            layer.has_previous_state = True
        delattr(layer, "_dflash_state_history")


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = logits.shape
    probs = torch.softmax(logits.reshape(-1, vocab_size) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(batch_size, seq_len)


def extract_context_feature(hidden_states: tuple[torch.Tensor, ...], layer_ids: list[int]) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def build_prompt(tokenizer, prompt_text: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)


def stop_reached(output_ids: torch.Tensor, num_input_tokens: int, stop_token_ids: list[int], device: torch.device) -> bool:
    if not stop_token_ids:
        return False
    stop_ids = torch.tensor(stop_token_ids, device=device)
    return bool(torch.isin(output_ids[0, num_input_tokens:], stop_ids).any().item())


@torch.inference_mode()
def dflash_generate(
    draft,
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int],
    verify_mode: str,
) -> tuple[torch.Tensor, dict]:
    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = draft.block_size

    output_ids = torch.full(
        (1, max_length + block_size),
        draft.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    target_cache = None
    if verify_mode == "cached":
        target_cache = DynamicCache(config=target.model.language_model.config)
    draft_cache = DynamicCache(config=draft.config)

    sync(device)
    wall_start = time.perf_counter()

    prefill_start = time.perf_counter()
    output = target(
        input_ids=input_ids,
        past_key_values=target_cache,
        use_cache=verify_mode == "cached",
        logits_to_keep=1,
        output_hidden_states=True,
    )
    sync(device)
    prefill_time = time.perf_counter() - prefill_start

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

    decode_start = time.perf_counter()
    acceptance_lengths: list[int] = []
    start = num_input_tokens

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        noise_embedding = target.get_input_embeddings()(block_output_ids)
        draft_logits = target.lm_head(
            draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, draft_cache.get_seq_length() : start + block_size],
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1 :, :]
        )
        draft_cache.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits, temperature)

        if verify_mode == "cached":
            target_cache._dflash_record_history = True
            output = target(
                input_ids=block_output_ids,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            target_cache._dflash_record_history = False
            posterior = sample(output.logits, temperature)
            verified_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)
        else:
            # Qwen3.5's multi-token cached path diverges from full-sequence logits on stock PyTorch,
            # so use a full verifier pass for correctness in this rough macOS prototype.
            verify_input_ids = torch.cat([output_ids[:, :start], block_output_ids], dim=1)
            output = target(
                input_ids=verify_input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            posterior = sample(output.logits[:, start : start + block_size], temperature)
            verified_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)[
                :, start : start + block_size, :
            ]

        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        if target_cache is not None and acceptance_length >= 0:
            restore_qwen3_5_linear_cache_prefix(target_cache, acceptance_length + 1)
            target_cache.crop(start)
        target_hidden = verified_hidden[:, : acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)

        if stop_reached(output_ids, num_input_tokens, stop_token_ids, device):
            break

    sync(device)
    total_time = time.perf_counter() - wall_start
    decode_time = time.perf_counter() - decode_start

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != draft.mask_token_id]
    if stop_token_ids:
        stop_ids = torch.tensor(stop_token_ids, device=device)
        stop_positions = torch.isin(output_ids[0][num_input_tokens:], stop_ids).nonzero(as_tuple=True)[0]
        if stop_positions.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_positions[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    metrics = {
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "generation_tps": num_output_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": num_output_tokens / max(total_time, 1e-9),
        "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
        "acceptance_lengths": acceptance_lengths,
    }
    return output_ids, metrics


def main() -> None:
    args = parse_args()
    requested_device = torch.device(args.device)
    if requested_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this environment.")

    dtype = get_dtype(args.dtype)
    prompt_text = args.prompt_file.read_text()

    if args.verify_mode == "cached":
        patch_qwen3_5_multitoken_cache()

    print(f"[load tokenizer] {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    print(f"[load target] {args.target_model} on {requested_device} ({args.dtype})")
    target = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(requested_device).eval()

    print(f"[load draft] {args.draft_model} on {requested_device} ({args.dtype})")
    draft = AutoModel.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(requested_device).eval()

    input_ids = build_prompt(tokenizer, prompt_text).to(requested_device)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    print(
        f"[run] prompt_tokens={input_ids.shape[1]} block_size={draft.block_size} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} "
        f"verify_mode={args.verify_mode}"
    )
    output_ids, metrics = dflash_generate(
        draft=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop_token_ids=stop_token_ids,
        verify_mode=args.verify_mode,
    )

    generated_ids = output_ids[0, metrics["num_input_tokens"] :]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print(f"Prompt tokens:            {metrics['num_input_tokens']}")
    print(f"Generated tokens:         {metrics['num_output_tokens']}")
    print(f"Prefill time:             {metrics['prefill_time_s']:.2f}s")
    print(f"Decode time:              {metrics['decode_time_s']:.2f}s")
    print(f"Total time:               {metrics['total_time_s']:.2f}s")
    print(f"Generation TPS:           {metrics['generation_tps']:.2f}")
    print(f"End-to-end TPS:           {metrics['end_to_end_tps']:.2f}")
    print(f"Average acceptance:       {metrics['avg_acceptance_length']:.2f}")
    print(f"Acceptance lengths:       {metrics['acceptance_lengths']}")
    print("=" * 60)

    if args.print_output:
        print(output_text)


if __name__ == "__main__":
    main()
