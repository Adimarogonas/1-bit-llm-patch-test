from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from bankai_poc.utils.artifacts import save_run_manifest
from bankai_poc.utils.io import dump_json, ensure_dir, load_yaml

from .registry import config_path, data_dir


def _download_hf_dataset(source: dict[str, Any], destination: Path) -> dict[str, Any]:
    dataset = load_dataset(
        source["path"],
        source.get("name"),
        split=source.get("split", "train"),
    )
    records = [dict(row) for row in dataset]
    dump_json(destination / "dataset.json", records)
    return {
        "rows": len(records),
        "raw_files": [str(destination / "dataset.json")],
        "features": list(dataset.features.keys()),
    }


def _download_hf_json_files(source: dict[str, Any], destination: Path) -> dict[str, Any]:
    repo_id = source["path"]
    repo_files = list_repo_files(repo_id, repo_type="dataset")
    requested = source.get("files", [])
    downloaded: list[str] = []
    preview: dict[str, Any] = {}
    for filename in requested:
        if filename not in repo_files:
            raise FileNotFoundError(f"{filename} not found in dataset repo {repo_id}")
        cached = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
        local_path = destination / filename
        ensure_dir(local_path.parent)
        local_path.write_bytes(Path(cached).read_bytes())
        downloaded.append(str(local_path))
        if local_path.suffix == ".json":
            with local_path.open("r", encoding="utf-8") as handle:
                preview[local_path.name] = handle.readline()[:1000]
    return {
        "rows": None,
        "raw_files": downloaded,
        "repo_file_count": len(repo_files),
        "preview": preview,
    }


def download_benchmark(benchmark: str) -> Path:
    config = load_yaml(config_path(benchmark))
    source = config["source"]
    destination = ensure_dir(data_dir(benchmark) / "raw")

    if source["kind"] == "huggingface_dataset":
        details = _download_hf_dataset(source, destination)
    elif source["kind"] == "huggingface_json_files":
        details = _download_hf_json_files(source, destination)
    else:
        raise ValueError(f"Unsupported source kind: {source['kind']}")

    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "source": source,
        **details,
    }
    save_run_manifest(destination / "download_manifest.json", payload)
    return destination
