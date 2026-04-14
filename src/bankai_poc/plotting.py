from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bankai_poc.utils.io import load_json


def generate_figures(results_dir: Path) -> list[Path]:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(results_dir / "benchmark_matrix.csv")
    routing = pd.read_csv(results_dir / "routing_metrics.csv")
    storage = load_json(results_dir / "storage_metrics.json")

    benchmark_chart = figures_dir / "benchmark_specialization.png"
    matrix.pivot(index="benchmark", columns="system", values="score").plot(kind="bar", figsize=(10, 5))
    plt.tight_layout()
    plt.savefig(benchmark_chart)
    plt.close()

    storage_chart = figures_dir / "storage_footprint.png"
    patch_bytes = storage["total_logical_patch_bytes"]
    plt.figure(figsize=(7, 4))
    plt.bar(["base", "base+patches", "multi-model", "adapter"], [1.0, 1.0 + patch_bytes / 1_000_000, 4.0, 1.35])
    plt.ylabel("Relative footprint")
    plt.tight_layout()
    plt.savefig(storage_chart)
    plt.close()

    heatmap_chart = figures_dir / "cross_domain_transfer.png"
    pivot = matrix[matrix["patch"] != "base"].pivot(index="patch", columns="benchmark", values="delta_vs_base")
    plt.figure(figsize=(7, 4))
    plt.imshow(pivot.values, cmap="coolwarm", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Delta vs base")
    plt.tight_layout()
    plt.savefig(heatmap_chart)
    plt.close()

    routed_chart = figures_dir / "routed_scores.png"
    routing.plot(x="benchmark", y="score", kind="bar", legend=False, figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(routed_chart)
    plt.close()

    return [benchmark_chart, storage_chart, heatmap_chart, routed_chart]
