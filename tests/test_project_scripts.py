from __future__ import annotations

import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chart_scripts_compile() -> None:
    scripts = [
        REPO_ROOT / "scripts" / "generate_benchmark_chart.py",
        REPO_ROOT / "scripts" / "generate_qwen_comparison_chart.py",
    ]
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


def test_public_cli_wrappers_compile() -> None:
    scripts = [
        REPO_ROOT / "scripts" / "run_dflash_mlx.py",
        REPO_ROOT / "scripts" / "benchmark_mlx.py",
        REPO_ROOT / "scripts" / "diagnose_dflash_acceptance.py",
    ]
    for script in scripts:
        py_compile.compile(str(script), doraise=True)
