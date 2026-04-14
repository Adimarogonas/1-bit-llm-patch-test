from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bankai_poc.data.registry import config_path, patch_path, probes_path
from bankai_poc.model.backend import MockBonsaiBackend
from bankai_poc.model.patching import BankaiPatch, PatchFlip, save_patch
from bankai_poc.utils.artifacts import save_run_manifest
from bankai_poc.utils.io import load_yaml, read_jsonl


@dataclass
class SearchResult:
    patch: BankaiPatch
    best_score: float
    trajectory: list[dict[str, Any]]


def _probe_score(
    probe: dict[str, Any], patch_rows: set[tuple[int, str, int]], benchmark: str
) -> float:
    base = 0.15
    if probe["benchmark"] == benchmark:
        base += 0.35
    overlap = sum(1 for row in patch_rows if row[2] % 7 == len(probe["correct_token"]) % 7)
    return min(1.0, base + (overlap * 0.03))


def run_search(benchmark: str, output_path: Path | None = None) -> SearchResult:
    config = load_yaml(config_path(benchmark))
    probes = read_jsonl(probes_path(benchmark))
    target_subset = probes[:128]
    model = MockBonsaiBackend.from_seed()
    rng = random.Random(benchmark)

    candidates: list[tuple[int, str, int, float]] = []
    for layer in config["search"]["search_layers"]:
        for proj in config["search"]["search_projs"]:
            module = model.modules[(layer, proj)]
            for row in range(min(config["search"]["candidate_rows"], module.weight.shape[0])):
                candidates.append((layer, proj, row, float(module.scales[row].mean())))
    candidates.sort(key=lambda item: item[3], reverse=True)

    chosen: list[PatchFlip] = []
    chosen_rows: set[tuple[int, str, int]] = set()
    best_score = 0.0
    trajectory: list[dict[str, Any]] = []
    screened_out = 0

    for iteration in range(config["search"]["iterations"]):
        layer, proj, row, scale = rng.choice(candidates)
        key = (layer, proj, row)
        if key in chosen_rows and len(chosen) >= config["search"]["max_flips"]:
            continue

        trial_rows = set(chosen_rows)
        trial_rows.add(key)
        score = sum(_probe_score(probe, trial_rows, benchmark) for probe in target_subset) / max(1, len(target_subset))

        screen_probes = sorted(target_subset, key=lambda probe: len(probe["correct_token"]))[:2]
        screen_score = sum(_probe_score(probe, trial_rows, benchmark) for probe in screen_probes) / max(1, len(screen_probes))
        baseline_screen_score = sum(_probe_score(probe, chosen_rows, benchmark) for probe in screen_probes) / max(1, len(screen_probes))
        if screen_score < baseline_screen_score:
            screened_out += 1
            trajectory.append(
                {
                    "iteration": iteration,
                    "layer": layer,
                    "proj": proj,
                    "row": row,
                    "scale_mean": scale,
                    "screened_out": True,
                }
            )
            continue

        accepted = score >= best_score
        trajectory.append(
            {
                "iteration": iteration,
                "layer": layer,
                "proj": proj,
                "row": row,
                "scale_mean": scale,
                "candidate_score": score,
                "accepted": accepted,
            }
        )
        if accepted and len(chosen) < config["search"]["max_flips"]:
            chosen_rows.add(key)
            chosen.append(PatchFlip(layer=layer, proj=proj, row=row))
            best_score = score

    patch = BankaiPatch(
        name=f"{benchmark}_patch",
        description=f"Benchmark-specific Bankai-style row XOR patch for {benchmark}.",
        base_model="prism-ml/Bonsai-8B-mlx-1bit",
        flips=chosen,
        metadata={
            "benchmark": benchmark,
            "created_at": time.time(),
            "base_checksum": model.checksum(),
            "final_fitness": best_score,
            "rows_flipped": len(chosen),
            "candidate_rows": config["search"]["candidate_rows"],
            "search_iterations": config["search"]["iterations"],
            "search_layers": config["search"]["search_layers"],
            "search_projs": config["search"]["search_projs"],
            "control_penalty": config["search"]["control_penalty"],
            "screened_out": screened_out,
        },
    )
    output_path = output_path or patch_path(benchmark)
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.parent.parent / "results" / f"{benchmark}_search.json",
        {
            "benchmark": benchmark,
            "patch_path": str(output_path),
            "best_score": best_score,
            "trajectory": trajectory,
            "screened_out": screened_out,
        },
    )
    return SearchResult(patch=patch, best_score=best_score, trajectory=trajectory)
