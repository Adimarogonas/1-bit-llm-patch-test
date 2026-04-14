from __future__ import annotations

from pathlib import Path

import pandas as pd

from bankai_poc.data.registry import BENCHMARKS, patch_path
from bankai_poc.model.patching import load_patch
from bankai_poc.utils.artifacts import save_run_manifest
from bankai_poc.utils.io import ensure_dir

from .benchmarks import evaluate_benchmark


def build_cross_benchmark_matrix(output_dir: Path) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    rows: list[dict[str, object]] = []

    for benchmark in BENCHMARKS:
        base = evaluate_benchmark(benchmark, patch=None)
        rows.append(
            {
                "system": "base",
                "patch": "base",
                "benchmark": benchmark,
                "score": base.score,
                "delta_vs_base": 0.0,
            }
        )

    for patch_name in BENCHMARKS:
        patch_file = patch_path(patch_name)
        if not patch_file.exists():
            continue
        patch = load_patch(patch_file)
        for benchmark in BENCHMARKS:
            base_score = evaluate_benchmark(benchmark, patch=None).score
            scored = evaluate_benchmark(benchmark, patch=patch)
            rows.append(
                {
                    "system": f"{patch_name}_patch",
                    "patch": patch_name,
                    "benchmark": benchmark,
                    "score": scored.score,
                    "delta_vs_base": scored.score - base_score,
                }
            )

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_matrix.csv"
    json_path = output_dir / "storage_metrics.json"
    frame.to_csv(csv_path, index=False)

    patches = []
    for name in BENCHMARKS:
        file = patch_path(name)
        if file.exists():
            patch = load_patch(file)
            patches.append(
                {
                    "benchmark": name,
                    "bytes": file.stat().st_size,
                    "logical_patch_bytes": patch.size_bytes,
                    "flips": len(patch.flips),
                }
            )
    save_run_manifest(
        json_path,
        {
            "base_backend": "mock_bonsai_backend",
            "patches": patches,
            "total_patch_bytes": sum(item["bytes"] for item in patches),
            "total_logical_patch_bytes": sum(item["logical_patch_bytes"] for item in patches),
        },
    )
    return csv_path, json_path
