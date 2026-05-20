from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bankai_poc.eval.dynamic_probes import build_dynamic_probes_from_details
from bankai_poc.eval.pipeline import JudgeConfig, ModelConfig, PipelineConfig, run_pipeline, timestamp_id
from bankai_poc.search.real_runner import run_real_greedy_search_from_probes, run_real_shortlist_search_from_probes
from bankai_poc.utils.io import dump_json, load_json, read_jsonl


@dataclass
class AdaptiveEvalConfig:
    dataset_path: Path
    model_ref: str
    output_dir: Path
    max_tokens: int = 160
    rounds: int = 4
    pool: int = 8
    topk: int = 2
    accept_per_round: int = 1
    search_mode: str = "shortlist"
    max_iters: int = 200
    fitness_mode: str = "mean"
    max_target_probes: int = 12
    max_control_probes: int = 6
    candidate_rows: int = 48
    max_flips: int = 16
    control_penalty: float = 2.0
    search_layers: list[int] | None = None
    layer_profile: str | None = "stable"
    impact_weighted: bool = True
    judge: JudgeConfig | None = None


def run_adaptive_eval(config: AdaptiveEvalConfig, log: Any | None = None) -> dict[str, str]:
    log = log or (lambda message: None)
    run_dir = config.output_dir / f"adaptive_{timestamp_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"Adaptive run: dataset={config.dataset_path} model={config.model_ref} output={run_dir} "
        f"rounds={config.rounds} pool={config.pool} topk={config.topk} "
        f"accept_per_round={config.accept_per_round} search_mode={config.search_mode} "
        f"max_iters={config.max_iters} fitness_mode={config.fitness_mode} "
        f"target_probes={config.max_target_probes} control_probes={config.max_control_probes} "
        f"candidate_rows={config.candidate_rows} max_flips={config.max_flips} "
        f"search_layers={config.search_layers} layer_profile={config.layer_profile} "
        f"impact_weighted={config.impact_weighted}"
    )

    log("Step 1/4: running the base model on the CSV eval")
    base_summary, base_details = run_pipeline(
        PipelineConfig(
            dataset_path=config.dataset_path,
            models=[ModelConfig(name="base-model", kind="mlx", model_ref=config.model_ref)],
            max_tokens=config.max_tokens,
            output_dir=run_dir / "base",
            judge=config.judge or JudgeConfig(),
        ),
        log=log,
    )

    log("Step 2/4: building dynamic probes from base failures")
    probes_path = build_dynamic_probes_from_details(
        details_path=base_details,
        dataset_path=config.dataset_path,
        output_path=run_dir / "dynamic_probes.jsonl",
        model_name="base-model",
        max_target=config.max_target_probes,
        max_control=config.max_control_probes,
    )
    probes = read_jsonl(probes_path)
    search_count = sum(1 for probe in probes if (probe.get("metadata") or {}).get("partition") == "search")
    control_count = sum(1 for probe in probes if (probe.get("metadata") or {}).get("partition") == "control")
    validation_count = sum(1 for probe in probes if (probe.get("metadata") or {}).get("partition") == "validation")
    log(
        "Dynamic probes ready: "
        f"{search_count} failure probes, {control_count} control probes, {validation_count} validation probes"
    )
    for probe in probes[: min(3, len(probes))]:
        meta = probe.get("metadata") or {}
        log(
            "Probe sample: "
            f"row={meta.get('source_row_id')} partition={meta.get('partition')} "
            f"correct={probe.get('correct_token')} wrong={probe.get('wrong_token')}"
        )

    log(
        "Step 3/4: searching for a patch using dynamic probes "
        f"(mode={config.search_mode}, rounds={config.rounds}, pool={config.pool}, topk={config.topk}, "
        f"accept_per_round={config.accept_per_round}, "
        f"max_iters={config.max_iters}, fitness_mode={config.fitness_mode}, "
        f"candidate_rows={config.candidate_rows}, max_flips={config.max_flips}, "
        f"control_penalty={config.control_penalty})"
    )
    patch_path = run_dir / "dynamic_patch.json"
    task_name = config.dataset_path.stem
    if config.search_mode == "greedy":
        search_result = run_real_greedy_search_from_probes(
            task_name=task_name,
            model_ref=config.model_ref,
            probes=probes,
            output_path=patch_path,
            max_iters=config.max_iters,
            fitness_mode=config.fitness_mode,
            max_target_probes=config.max_target_probes,
            max_control_probes=config.max_control_probes,
            candidate_rows=config.candidate_rows,
            max_flips=config.max_flips,
            control_penalty=config.control_penalty,
            search_layers=config.search_layers,
            layer_profile=config.layer_profile,
            impact_weighted=config.impact_weighted,
            verbose=True,
        )
    elif config.search_mode == "shortlist":
        search_result = run_real_shortlist_search_from_probes(
            task_name=task_name,
            model_ref=config.model_ref,
            probes=probes,
            output_path=patch_path,
            rounds=config.rounds,
            shortlist_pool=config.pool,
            shortlist_topk=config.topk,
            accept_per_round=config.accept_per_round,
            max_target_probes=config.max_target_probes,
            max_control_probes=config.max_control_probes,
            candidate_rows=config.candidate_rows,
            max_flips=config.max_flips,
            control_penalty=config.control_penalty,
            search_layers=config.search_layers,
            layer_profile=config.layer_profile,
            impact_weighted=config.impact_weighted,
            verbose=True,
        )
    else:
        raise ValueError("search_mode must be 'shortlist' or 'greedy'")
    log(
        f"Patch search complete: flips={len(search_result.patch.flips)} "
        f"best_probe_score={search_result.best_score:+.6f}"
    )
    for flip in search_result.patch.flips:
        log(f"Accepted flip: layer={flip.layer} proj={flip.proj} row={flip.row}")

    log("Step 4/4: re-running the eval with the learned patch")
    patched_summary, patched_details = run_pipeline(
        PipelineConfig(
            dataset_path=config.dataset_path,
            models=[
                ModelConfig(
                    name="patched-model",
                    kind="bankai_mlx",
                    model_ref=config.model_ref,
                    patch_path=patch_path,
                )
            ],
            max_tokens=config.max_tokens,
            output_dir=run_dir / "patched",
            judge=config.judge or JudgeConfig(),
        ),
        log=log,
    )

    manifest = {
        "run_dir": str(run_dir),
        "dataset_path": str(config.dataset_path),
        "model_ref": config.model_ref,
        "base_summary": str(base_summary),
        "base_details": str(base_details),
        "dynamic_probes": str(probes_path),
        "patch": str(patch_path),
        "patched_summary": str(patched_summary),
        "patched_details": str(patched_details),
        "best_probe_score": search_result.best_score,
        "patch_flips": len(search_result.patch.flips),
        "search_mode": config.search_mode,
        "max_iters": config.max_iters,
        "fitness_mode": config.fitness_mode,
        "candidate_rows": config.candidate_rows,
        "max_flips": config.max_flips,
        "control_penalty": config.control_penalty,
        "search_layers": config.search_layers,
        "layer_profile": config.layer_profile,
        "impact_weighted": config.impact_weighted,
        "changed_generations": _changed_generation_count(base_details, patched_details),
    }
    dump_json(run_dir / "adaptive_manifest.json", manifest)
    log(f"Done. Adaptive manifest: {run_dir / 'adaptive_manifest.json'}")
    return manifest


def _changed_generation_count(base_details: Path, patched_details: Path) -> int:
    base_rows = load_json(base_details)["rows"]
    patched_rows = load_json(patched_details)["rows"]
    return sum(
        1
        for base_row, patched_row in zip(base_rows, patched_rows)
        if (base_row.get("prediction") or "") != (patched_row.get("prediction") or "")
    )
