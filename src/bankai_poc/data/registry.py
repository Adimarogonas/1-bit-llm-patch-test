from __future__ import annotations

from pathlib import Path

BENCHMARKS = ("gsm8k", "humaneval_plus", "ifeval", "bfcl")


def root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def config_path(benchmark: str) -> Path:
    return root_dir() / "configs" / f"{benchmark}.yaml"


def data_dir(benchmark: str) -> Path:
    return root_dir() / "data" / benchmark


def probes_path(benchmark: str) -> Path:
    return root_dir() / "probes" / benchmark / "probes.jsonl"


def patch_path(benchmark: str) -> Path:
    return root_dir() / "patches" / f"{benchmark}_patch.json"

