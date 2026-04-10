#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUSTOM_MODEL_ROOT = REPO_ROOT / "cache" / "custom_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local MLX model directory that uses a custom model_file."
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="Source MLX model repo or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to create for the custom model.",
    )
    parser.add_argument(
        "--model-file-source",
        type=Path,
        default=REPO_ROOT / "scripts" / "custom_qwen35_dflash_model.py",
        help="Custom model Python file to copy into the output directory.",
    )
    parser.add_argument(
        "--model-file-name",
        default="custom_qwen35_dflash_model.py",
        help="Filename to use inside the custom model directory.",
    )
    return parser.parse_args()


def resolve_source(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path.resolve()
    return Path(snapshot_download(path_or_repo)).resolve()


def default_output_dir(source_repo: str) -> Path:
    slug = source_repo.replace("/", "__").replace(":", "__")
    return DEFAULT_CUSTOM_MODEL_ROOT / f"{slug}__custom_qwen35"


def prepare_custom_model(
    source_repo: str,
    output_dir: Path | None = None,
    model_file_source: Path = REPO_ROOT / "scripts" / "custom_qwen35_dflash_model.py",
    model_file_name: str = "custom_qwen35_dflash_model.py",
) -> Path:
    source_dir = resolve_source(source_repo)
    output_dir = (output_dir or default_output_dir(source_repo)).resolve()

    needs_refresh = True
    if output_dir.exists():
        config_path = output_dir / "config.json"
        model_file_path = output_dir / model_file_name
        if config_path.exists() and model_file_path.exists():
            existing_config = json.loads(config_path.read_text())
            if existing_config.get("model_file") == model_file_name:
                try:
                    if model_file_path.read_text() == model_file_source.read_text():
                        needs_refresh = False
                except FileNotFoundError:
                    needs_refresh = True

    if not needs_refresh:
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for source_path in source_dir.iterdir():
        if source_path.name == "config.json":
            continue
        target_path = output_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, symlinks=True)
        else:
            target_path.symlink_to(source_path)

    config = json.loads((source_dir / "config.json").read_text())
    config["model_file"] = model_file_name
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(model_file_source, output_dir / model_file_name)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = prepare_custom_model(
        source_repo=args.source_repo,
        output_dir=args.output_dir,
        model_file_source=args.model_file_source,
        model_file_name=args.model_file_name,
    )

    print(f"Prepared custom model at {output_dir}")


if __name__ == "__main__":
    main()
