"""Smoke test: the package imports and public surface is wired correctly.

Intentionally does not load any model weights. A contributor can run this
without an Apple Silicon GPU and without a ~12 GB model download.
"""
from __future__ import annotations

import sys


def test_public_api_importable():
    import dflash_mlx

    expected = {
        "DDTreeConfig",
        "DFlashGenerator",
        "DFlashResult",
        "DFlashDraftModel",
        "DraftTree",
        "LoadedTargetModel",
        "adapter_for_model_type",
        "build_draft_tree",
        "dflash_generate",
        "ddtree_generate",
        "load_draft_model",
        "load_target_model",
        "longest_prefix_match",
        "sample_tokens",
        "walk_tree",
    }
    assert expected.issubset(set(dflash_mlx.__all__))
    for name in expected:
        assert hasattr(dflash_mlx, name), f"missing export: {name}"


def test_generator_is_callable_class():
    from dflash_mlx import DFlashGenerator

    assert callable(DFlashGenerator)


def test_cli_entrypoints_importable():
    from dflash_mlx import benchmark_cli, chat_cli, cli, inspect_cli, model_prep

    for module in (cli, benchmark_cli, chat_cli, inspect_cli, model_prep):
        assert callable(module.main)


def test_cli_verify_mode_choices_exclude_unsafe_modes():
    from dflash_mlx import chat_cli, cli

    saved_argv = sys.argv
    try:
        for module in (cli, chat_cli):
            sys.argv = [module.__name__]
            args = module.parse_args()
            assert args.verify_mode == "parallel-replay"
    finally:
        sys.argv = saved_argv
