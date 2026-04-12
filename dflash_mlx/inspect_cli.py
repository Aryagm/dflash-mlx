#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .adapters import adapter_for_model_type
from .api import DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL


def read_config(path_or_repo: str) -> tuple[dict[str, Any], str]:
    path = Path(path_or_repo)
    if path.exists():
        config_path = path / "config.json"
        return json.loads(config_path.read_text()), str(config_path.resolve())

    config_path = hf_hub_download(path_or_repo, "config.json")
    return json.loads(Path(config_path).read_text()), config_path


def support_status(model_type: str | None) -> str:
    if model_type in {"qwen3", "qwen3_5"}:
        return "supported"
    return "unsupported"


def inspect_pair(target_model: str, draft_model: str) -> dict[str, Any]:
    target_config, target_config_path = read_config(target_model)
    draft_config, draft_config_path = read_config(draft_model)
    target_model_type = target_config.get("model_type")
    adapter_cls = adapter_for_model_type(target_model_type)
    dflash_config = draft_config.get("dflash_config") or {}
    custom_wrapper_needed = (
        target_model_type == "qwen3_5"
        and target_config.get("model_file") != "custom_qwen35_dflash_model.py"
    )

    messages: list[str] = []
    if adapter_cls is None:
        messages.append(
            "No MLX DFlash adapter is registered for this target model_type. "
            "A DFlash draft checkpoint alone is not enough; add an adapter first."
        )
    if not dflash_config:
        messages.append("Draft config has no dflash_config; it may not be a DFlash draft.")
    if custom_wrapper_needed:
        messages.append(
            "Qwen3.5 exact verification uses a local custom MLX model wrapper "
            "for hidden-state extraction and linear-attention cache rollback."
        )

    return {
        "target_model": target_model,
        "target_config_path": target_config_path,
        "target_model_type": target_model_type,
        "target_model_file": target_config.get("model_file", ""),
        "adapter": adapter_cls.__name__ if adapter_cls is not None else "",
        "status": support_status(target_model_type),
        "custom_wrapper_needed": custom_wrapper_needed,
        "draft_model": draft_model,
        "draft_config_path": draft_config_path,
        "draft_model_type": draft_config.get("model_type", ""),
        "draft_block_size": draft_config.get("block_size", ""),
        "draft_mask_token_id": dflash_config.get("mask_token_id", ""),
        "draft_target_layer_ids": dflash_config.get("target_layer_ids", []),
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect DFlash target/draft support without loading weights."
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = inspect_pair(args.target_model, args.draft_model)
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
        return

    print(f"Target:        {info['target_model']}")
    print(f"Model type:    {info['target_model_type']}")
    print(f"Adapter:       {info['adapter'] or 'none'}")
    print(f"Status:        {info['status']}")
    print(f"Custom model:  {'yes' if info['custom_wrapper_needed'] else 'no'}")
    print(f"Draft:         {info['draft_model']}")
    print(f"Draft type:    {info['draft_model_type']}")
    print(f"Block size:    {info['draft_block_size']}")
    print(f"Target layers: {info['draft_target_layer_ids']}")
    for message in info["messages"]:
        print(f"Note:          {message}")


if __name__ == "__main__":
    main()
