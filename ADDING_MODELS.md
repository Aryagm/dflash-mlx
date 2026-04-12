# Adding Models

`dflash-mlx` can only run target models that have both:

- A matching DFlash draft checkpoint.
- An MLX target adapter that can expose verifier internals.

The draft model predicts a block. The target model verifies that block and returns the hidden states needed to condition the next draft.

Current exact support is centered on Qwen3-4B. Qwen3.5-4B remains supported, but is no longer the default benchmark target. Other upstream DFlash models are not automatically supported until their MLX target family has an adapter and exactness validation.

## 1. Check the Pair

Pick a pair from the upstream DFlash model list:

```text
target model + z-lab/<target>-DFlash
```

Then inspect the target model's `config.json`:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
import json

path = snapshot_download("mlx-community/Qwen3-4B-bf16")
config = json.load(open(f"{path}/config.json"))
print(config["model_type"])
print(config.get("model_file"))
PY
```

If `model_type` is already registered in `dflash_mlx/adapters.py`, start by reusing the existing adapter.

## 2. Add an Adapter

Add a subclass of `MLXTargetAdapter` in `dflash_mlx/adapters.py` and register it in `ADAPTERS`.

The adapter must implement:

```python
class NewTargetAdapter(MLXTargetAdapter):
    family = "new_family"

    def build_prompt(self, tokenizer, prompt_text): ...
    def stop_token_ids(self, tokenizer): ...
    def make_cache(self, model): ...
    def embed_tokens(self, model, tokens): ...
    def lm_head_logits(self, model, hidden_states): ...
    def forward_with_hidden_states(self, model, inputs, cache, layer_ids, return_rollback_records=False): ...
    def forward_verifier_states(self, model, inputs, cache, layer_ids): ...
    def forward_accept_all_block(self, model, inputs, cache, layer_ids): ...
    def rewind_kv_caches(self, cache, num_tokens): ...
    def rollback_linear_caches(self, model, cache, rollback_records, accepted_inputs): ...
    def cache_summary(self, cache): ...
```

The critical method is `forward_with_hidden_states`: it must run the target model and return `(logits, target_hidden)`, where `target_hidden` is the concatenation of the target layers listed in the draft checkpoint's `dflash_config.target_layer_ids`.

## 3. Decide If a Custom Model Is Needed

You usually do not need a custom model fork for plain decoder-only transformer models with normal KV caches. The adapter can run the layers directly and call `KVCache.trim(...)` on rejection.

You may need a custom model file when the target has cache state that cannot be rolled back generically:

- Hybrid attention plus linear attention.
- SSM/Mamba-style recurrent state.
- MoE or architecture-specific execution paths that MLX-LM does not expose cleanly.
- Any model where exact verification needs intermediate tensors that the public MLX model does not return.

Qwen3.5 uses a custom model because it has Qwen3-Next-style gated-delta/linear-attention caches. Exact DFlash needs to accept only part of a verified block and then rebuild the recurrent linear-attention state for the accepted prefix.

If a custom model is needed, copy the closest MLX-LM model implementation into `dflash_mlx/`, add `forward_dflash(...)` and rollback helpers, then point `resolve_target_model_path(...)` at `prepare_custom_model(...)`.

## 4. Validate Exactness

Use this as the concrete pass/fail gate before calling a new pair supported.

Before marking a model supported:

```bash
uv run python -m py_compile dflash_mlx/*.py

uv run python - <<'PY'
from argparse import Namespace

from dflash_mlx import DFlashGenerator
from dflash_mlx.benchmark_cli import run_one_prompt

target = "<target>"
draft = "<draft>"
prompt = "Write a quicksort in Python."

runner = DFlashGenerator(target_model=target, draft_model=draft)
prompt_tokens = runner.encode_prompt(prompt)

dflash = runner.generate_from_tokens(
    prompt_tokens,
    max_new_tokens=64,
    temperature=0.0,
    verify_mode="parallel-replay",
)

plain_args = Namespace(
    max_new_tokens=64,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    min_tokens_to_keep=1,
)
plain = run_one_prompt(
    runner.target.model,
    runner.target.tokenizer,
    prompt_tokens.tolist(),
    plain_args,
)

if dflash.text != plain.output_text:
    raise SystemExit("FAIL: DFlash output differs from plain greedy target output.")
if dflash.metrics["avg_acceptance_length"] <= 0:
    raise SystemExit("FAIL: DFlash did not accept any draft tokens.")

print("PASS: exact greedy smoke test matched plain target output.")
PY
```

Pass means all of the following are true:

- `py_compile` exits cleanly.
- `dflash-mlx` exits cleanly in exact `parallel-replay` mode.
- Plain target generation exits cleanly for the same prompt and target.
- For greedy decoding, the DFlash decoded output exactly matches plain target decoded output on the same prompt at `--temperature 0`.
- The DFlash run reports a positive average acceptance length and does not use any inexact verifier mode.

Fail means the decoded output differs, any command exits non-zero, or the adapter needs an inexact verifier path to run. If it fails, do not list the model as supported.

## 5. Add the README Row

Add the target/draft pair to the Supported Models table in `README.md` only after exact mode works.
