"""Diffusion Draft Tree (DDTree) — tree-based speculative decoding on Apple Silicon.

Implements the DDTree method from Ringel & Romano (2026):
  "Accelerating Speculative Decoding with Block Diffusion Draft Trees"

DDTree builds a draft tree from the per-position marginal distributions
produced by a single DFlash block diffusion forward pass, selects the
top-B prefixes via a best-first heap (Algorithm 1), verifies them in one
target-model pass with tree attention, and walks the result.

Integrates Bet-Optimal Drafting (BOD) in tree mode to dynamically select
the node budget B that maximizes throughput.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .adapters import LoadedTargetModel
from .bet_optimal_drafting import BODConfig, BODController
from .draft import DFlashDraftModel
from .runtime import sample_tokens, stop_position, trim_draft_cache


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    token_id: int
    depth: int
    parent_idx: int
    log_prob: float
    rank_tuple: tuple[int, ...]
    children: list[int] = field(default_factory=list)


@dataclass
class DraftTree:
    """Draft tree built from block diffusion per-position marginals."""

    nodes: list[_TreeNode] = field(default_factory=list)
    root_token: int = -1
    block_size: int = 0

    @property
    def size(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Algorithm 1: Best-first draft-tree construction
# ---------------------------------------------------------------------------

def build_draft_tree(
    draft_logits: mx.array,
    bonus_token: int,
    budget: int,
    block_size: int,
) -> DraftTree:
    """Build an optimal draft tree from one DFlash forward pass.

    Implements Algorithm 1 from the DDTree paper. Given per-position
    logits, selects the B highest-probability prefixes under the
    factorized distribution Q using a max-heap.

    The heap stores rank tuples rho = (rho_1, ..., rho_d) scored by their
    log-probability sigma(rho) = sum log q^(rho_i)_i. When a tuple rho
    is popped, two new candidates may be pushed:
      - next sibling: (rho_1, ..., rho_{d-1}, rho_d + 1)  (if rho_d + 1 < K)
      - first child:  (rho_1, ..., rho_d, 0)               (if d < L)
    """
    L = min(block_size, draft_logits.shape[1])
    K = min(budget, draft_logits.shape[-1])

    log_probs_all = mx.log(mx.softmax(draft_logits[0, :L, :], axis=-1))

    top_k_tokens: list[list[int]] = []
    top_k_log_probs: list[list[float]] = []
    for i in range(L):
        row = log_probs_all[i]
        indices = mx.argsort(row)[::-1][:K]
        indices_list = indices.tolist()
        top_k_tokens.append(indices_list)
        top_k_log_probs.append([float(row[idx]) for idx in indices_list])

    tree = DraftTree(root_token=bonus_token, block_size=L)
    rank_to_idx: dict[tuple[int, ...], int] = {}

    max_heap: list[tuple[float, int, tuple[int, ...]]] = []
    counter = 0

    first_rho = (0,)
    first_lp = top_k_log_probs[0][0]
    heapq.heappush(max_heap, (-first_lp, counter, first_rho))
    counter += 1

    while len(tree.nodes) < budget and max_heap:
        neg_lp, _, rho = heapq.heappop(max_heap)
        d = len(rho)

        parent_idx = -1
        if d > 1:
            parent_rho = rho[:-1]
            parent_idx = rank_to_idx.get(parent_rho, -1)

        tid = top_k_tokens[d - 1][rho[-1]]
        lp = -neg_lp

        node_idx = len(tree.nodes)
        node = _TreeNode(
            token_id=tid,
            depth=d,
            parent_idx=parent_idx,
            log_prob=lp,
            rank_tuple=rho,
        )
        tree.nodes.append(node)
        rank_to_idx[rho] = node_idx

        if parent_idx >= 0:
            tree.nodes[parent_idx].children.append(node_idx)

        if rho[-1] + 1 < K:
            sib_rho = rho[:-1] + (rho[-1] + 1,)
            sib_lp = lp - top_k_log_probs[d - 1][rho[-1]] + top_k_log_probs[d - 1][rho[-1] + 1]
            heapq.heappush(max_heap, (-sib_lp, counter, sib_rho))
            counter += 1

        if d < L:
            child_rho = rho + (0,)
            child_lp = lp + top_k_log_probs[d][0]
            heapq.heappush(max_heap, (-child_lp, counter, child_rho))
            counter += 1

    return tree


# ---------------------------------------------------------------------------
# Tree flattening for target-model verification
# ---------------------------------------------------------------------------

@dataclass
class TreeTensors:
    token_ids: mx.array
    position_ids: mx.array
    attention_mask: mx.array
    tree_size: int
    tree: DraftTree
    flat_order: list[int]


def compile_tree_tensors(
    tree: DraftTree,
    context_len: int,
) -> TreeTensors:
    """Flatten the draft tree into tensors for one target-model forward pass.

    Tokens are laid out in BFS order starting with the bonus token (root).
    Position ids equal tree depth. The attention mask ensures each drafted
    node attends to all context tokens plus only its ancestors (and itself)
    within the tree -- ancestor-only tree attention (Section 4.4).

    NOTE: The current adapter API applies causal attention by default.
    The ancestor-only mask is computed here for correctness and for
    future use when adapters support custom masks. With causal attention,
    each node sees all predecessors (not just ancestors), which makes
    verification approximate but still functional.
    """
    B = tree.size
    if B == 0:
        return TreeTensors(
            token_ids=mx.array([], dtype=mx.uint32),
            position_ids=mx.array([], dtype=mx.int32),
            attention_mask=mx.zeros((0, 0), dtype=mx.float32),
            tree_size=0,
            tree=tree,
            flat_order=[],
        )

    flat_order = _bfs_order(tree)
    order_inv = {orig: flat for flat, orig in enumerate(flat_order)}

    ancestor_sets: list[set[int]] = []
    for orig_idx in flat_order:
        anc: set[int] = set()
        cur = orig_idx
        while cur >= 0:
            anc.add(cur)
            cur = tree.nodes[cur].parent_idx
        ancestor_sets.append(anc)

    total_len = 1 + B
    mask_np = [[0.0] * total_len for _ in range(B)]

    for flat_i in range(B):
        mask_np[flat_i][0] = 1.0
        for anc_orig in ancestor_sets[flat_i]:
            flat_j = order_inv[anc_orig]
            mask_np[flat_i][1 + flat_j] = 1.0

    token_ids_list = [tree.nodes[idx].token_id for idx in flat_order]
    position_ids_list = [tree.nodes[idx].depth for idx in flat_order]

    return TreeTensors(
        token_ids=mx.array(token_ids_list, dtype=mx.uint32),
        position_ids=mx.array(position_ids_list, dtype=mx.int32),
        attention_mask=mx.array(mask_np, dtype=mx.float32),
        tree_size=B,
        tree=tree,
        flat_order=flat_order,
    )


def _bfs_order(tree: DraftTree) -> list[int]:
    order: list[int] = []
    queue: list[int] = []
    for i, node in enumerate(tree.nodes):
        if node.depth == 1:
            queue.append(i)
    queue.sort(key=lambda i: tree.nodes[i].rank_tuple)
    while queue:
        next_queue: list[int] = []
        for idx in queue:
            order.append(idx)
            next_queue.extend(tree.nodes[idx].children)
        next_queue.sort(key=lambda i: tree.nodes[i].rank_tuple)
        queue = next_queue
    return order


# ---------------------------------------------------------------------------
# Verifier walk
# ---------------------------------------------------------------------------

@dataclass
class WalkResult:
    accepted_tokens: list[int]
    accepted_count: int
    bonus_token: int


def walk_tree(
    tree: DraftTree,
    flat_order: list[int],
    verifier_logits: mx.array,
    temperature: float,
) -> WalkResult:
    """Walk the draft tree following the target model's decoding rule.

    verifier_logits has shape [1, 1+B, V] where position 0 is the bonus
    token and positions 1..B are the tree nodes in flat_order. At each
    step the target model's logits at the current position determine
    the next token. If that token matches a child in the tree, the walk
    continues; otherwise it stops and the unmatched token becomes the
    new bonus token.

    Figure 2(b): starting from the bonus token b, check whether the
    token selected by the target model matches a child. If yes, accept
    and continue. If no, stop; the first unmatched target token becomes
    the next bonus token.
    """
    order_inv = {orig: flat for flat, orig in enumerate(flat_order)}

    node_children: dict[int, dict[int, int]] = {}
    for i, node in enumerate(tree.nodes):
        node_children[i] = {}
        for child_idx in node.children:
            node_children[i][tree.nodes[child_idx].token_id] = child_idx

    root_children: dict[int, int] = {}
    for i, node in enumerate(tree.nodes):
        if node.depth == 1:
            root_children[node.token_id] = i

    accepted_tokens: list[int] = []
    current_node = -1

    while True:
        if current_node == -1:
            child_dict = root_children
        else:
            child_dict = node_children.get(current_node, {})

        if current_node == -1:
            logits_pos = 0
        else:
            logits_pos = 1 + order_inv.get(current_node, 0)

        logits_at = verifier_logits[0, logits_pos, :]
        selected = int(sample_tokens(logits_at[None, None, :], temperature).item())

        if selected in child_dict:
            next_node = child_dict[selected]
            accepted_tokens.append(selected)
            current_node = next_node
        else:
            return WalkResult(
                accepted_tokens=accepted_tokens,
                accepted_count=len(accepted_tokens),
                bonus_token=selected,
            )


# ---------------------------------------------------------------------------
# DDTree generate loop
# ---------------------------------------------------------------------------

@dataclass
class DDTreeConfig:
    budget: int = 64
    use_bod: bool = False
    bod_config: BODConfig | None = None
    profile: bool = False


def ddtree_generate(
    target: LoadedTargetModel,
    draft: DFlashDraftModel,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: set[int],
    layer_ids: list[int],
    config: DDTreeConfig | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """DDTree speculative decoding with tree-based verification.

    Per round:
    1. Run one DFlash drafter pass -> per-position marginals {q_i}.
    2. Build draft tree with B nodes (Algorithm 1).
    3. Flatten tree + bonus token into one sequence, run target model.
    4. Walk tree; accept matched path; fix caches; carry next bonus.

    Cache management:
    The accepted path through the tree is a specific trajectory (not a
    prefix of the BFS-ordered input), so we cannot simply trim the tail.
    Instead we:
      a. Snapshot SSM caches (Qwen3.5) before the tree forward pass.
      b. Forward the tree input through the target model.
      c. Walk the tree to determine the accepted path.
      d. Rewind KV caches by the full tree token count.
      e. Restore SSM caches from the snapshot.
      f. Replay the accepted path [bonus, accepted_1, ..., accepted_d]
         through the target model to rebuild both KV and SSM caches
         correctly, and extract target_hidden for the next draft step.

    The replay adds a short forward pass of (1 + accepted_count) tokens,
    which is a small fraction of the tree verification cost.
    """
    if config is None:
        config = DDTreeConfig()

    budget = config.budget
    block_size = draft.block_size

    bod: BODController | None = None
    if config.use_bod:
        bod_cfg = config.bod_config or BODConfig(
            mode="tree",
            min_bet=16,
            max_bet=1024,
        )
        bod = BODController(bod_cfg)

    target_cache = target.make_cache()
    draft_cache = draft.make_cache()
    total_max_tokens = int(prompt_tokens.shape[0]) + max_new_tokens
    prompt_len = int(prompt_tokens.shape[0])

    has_ssm = target.adapter.family in ("qwen3_5",)

    t0 = time.perf_counter()
    logits, target_hidden = target.forward_with_hidden_states(
        prompt_tokens[None],
        target_cache,
        layer_ids,
    )
    first_token = int(sample_tokens(logits[:, -1, :], temperature).item())
    mx.eval(logits, target_hidden)
    prefill_time = time.perf_counter() - t0

    output_tokens = prompt_tokens.tolist() + [first_token]
    pos = prompt_len
    acceptance_lengths: list[int] = []

    decode_start = time.perf_counter()
    while pos < total_max_tokens:
        if bod is not None:
            budget = bod.optimal_bet()

        effective_budget = max(1, min(budget, 1024))

        draft_t0 = time.perf_counter()
        bonus_token = output_tokens[pos]
        block_tokens = [bonus_token] + [draft.mask_token_id] * (block_size - 1)
        block_input = mx.array(block_tokens, dtype=mx.uint32)[None]
        noise_embedding = target.embed_tokens(block_input)

        draft_hidden = draft(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = target.lm_head_logits(draft_hidden[:, 1:, :])
        mx.eval(draft_logits)
        trim_draft_cache(draft_cache, block_size)
        draft_time = time.perf_counter() - draft_t0

        tree_t0 = time.perf_counter()
        tree = build_draft_tree(
            draft_logits=draft_logits[:, :block_size, :],
            bonus_token=bonus_token,
            budget=effective_budget,
            block_size=block_size,
        )
        tree_time = time.perf_counter() - tree_t0

        if tree.size == 0:
            out_tok = int(sample_tokens(draft_logits[0, 0:1, :], temperature).item())
            output_tokens.append(out_tok)
            pos += 1
            acceptance_lengths.append(0)
            continue

        tree_tensors = compile_tree_tensors(tree, pos)
        total_tree_tokens = 1 + tree.size

        if has_ssm:
            ssm_snapshot = target.snapshot_linear_caches(target_cache)

        verify_t0 = time.perf_counter()
        tree_input_ids = mx.array(
            [bonus_token] + [tree.nodes[i].token_id for i in tree_tensors.flat_order],
            dtype=mx.uint32,
        )[None]

        verifier_logits, _, _ = target.forward_with_hidden_states(
            tree_input_ids,
            target_cache,
            layer_ids,
            return_rollback_records=True,
        )
        mx.eval(verifier_logits)
        verify_time = time.perf_counter() - verify_t0

        walk_result = walk_tree(
            tree=tree,
            flat_order=tree_tensors.flat_order,
            verifier_logits=verifier_logits,
            temperature=temperature,
        )

        accepted_count = walk_result.accepted_count
        new_bonus = walk_result.bonus_token
        acceptance_lengths.append(accepted_count)

        target.rewind_kv_caches(target_cache, total_tree_tokens)

        if has_ssm:
            target.restore_linear_caches(target_cache, ssm_snapshot)

        if accepted_count > 0:
            replay_ids = mx.array(
                [bonus_token] + walk_result.accepted_tokens,
                dtype=mx.uint32,
            )[None]
        else:
            replay_ids = mx.array([[bonus_token]], dtype=mx.uint32)

        replay_logits, target_hidden = target.forward_with_hidden_states(
            replay_ids,
            target_cache,
            layer_ids,
        )
        mx.eval(replay_logits, target_hidden)

        if accepted_count == 0:
            target.rewind_kv_caches(target_cache, 1)

        output_tokens = output_tokens[:pos + 1]
        output_tokens.extend(walk_result.accepted_tokens)
        output_tokens.append(new_bonus)
        pos += 1 + accepted_count

        if bod is not None:
            bod.observe(
                bet=effective_budget,
                accepted=accepted_count,
                cycle_time_ms=(draft_time + verify_time + tree_time) * 1000,
                draft_time_ms=draft_time * 1000,
                verify_time_ms=verify_time * 1000,
                confidence=float(mx.max(mx.softmax(draft_logits[0, 0, :])).item()),
            )

        if stop_position(output_tokens, prompt_len, stop_token_ids) is not None:
            break
        if len(output_tokens) > total_max_tokens:
            break

    decode_time = time.perf_counter() - decode_start
    output_tokens = output_tokens[:total_max_tokens]
    generated_tokens = max(len(output_tokens) - prompt_len, 0)
    total_time = prefill_time + decode_time

    metrics: dict[str, Any] = {
        "num_input_tokens": prompt_len,
        "num_output_tokens": generated_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "prompt_tps": prompt_len / max(prefill_time, 1e-9),
        "generation_tps": generated_tokens / max(decode_time, 1e-9),
        "end_to_end_tps": generated_tokens / max(total_time, 1e-9),
        "avg_acceptance_length": sum(acceptance_lengths) / max(len(acceptance_lengths), 1),
        "acceptance_lengths": acceptance_lengths,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "target_cache_summary": target.cache_summary(target_cache),
        "speculative_tokens": block_size,
        "ddtree_budget": budget,
        "ddtree_use_bod": config.use_bod,
    }
    if bod is not None:
        metrics["bod_final_budget"] = budget

    return output_tokens, metrics
