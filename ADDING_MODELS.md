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

Before marking a model supported:

```bash
uv run python -m py_compile dflash_mlx/*.py
uv run dflash-mlx --target-model <target> --draft-model <draft> --max-new-tokens 64 --print-output --no-history
uv run dflash-mlx-bench --model <target> --prompt "Write a quicksort in Python." --max-new-tokens 64 --no-history
```

For greedy decoding, DFlash output must match the target verifier's accepted path. `--verify-mode accept-all` is not exact and must not be used to claim model support.

## 5. Add the README Row

Add the target/draft pair to the Supported Models table in `README.md` only after exact mode works.
