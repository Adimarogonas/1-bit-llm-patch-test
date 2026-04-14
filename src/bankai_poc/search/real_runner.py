from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from bankai_poc.data.registry import config_path, patch_path, probes_path
from bankai_poc.model.patching import BankaiPatch, PatchFlip, save_patch
from bankai_poc.model.real_mlx import apply_real_patch, flip_row, get_module, load_real_model, model_patchable_summary
from bankai_poc.utils.artifacts import save_run_manifest
from bankai_poc.utils.io import load_yaml, read_jsonl


LAYER_PROFILES: dict[str, list[int]] = {
    "stable": [8, 12, 16, 17, 18, 19, 20, 21, 22, 24, 28],
    "balanced": [5, 8, 12, 16, 17, 18, 19, 20, 21, 22, 24, 28, 32, 35],
    "aggressive": [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 34, 35],
}

LAYER_IMPACT: dict[int, float] = {
    **{layer: 5.2 for layer in range(0, 5)},
    **{layer: 1.9 for layer in range(5, 17)},
    **{layer: 1.1 for layer in range(17, 22)},
    **{layer: 2.5 for layer in range(22, 34)},
    34: 9.0,
    35: 3.2,
}


@dataclass
class RealSearchResult:
    patch: BankaiPatch
    best_score: float
    trajectory: list[dict[str, Any]]


def _pre_tokenize(tokenizer: Any, probes: list[dict[str, Any]]) -> list[tuple[mx.array, int, int]]:
    packed = []
    for probe in probes:
        tokens = mx.array(tokenizer.encode(probe["prompt"]))
        correct_id = tokenizer.encode(probe["correct_token"])[-1]
        wrong_id = tokenizer.encode(probe["wrong_token"])[-1]
        packed.append((tokens, correct_id, wrong_id))
    return packed


def _measure_fast(model: Any, packed: list[tuple[mx.array, int, int]], names: list[str]) -> dict[str, float]:
    gaps: dict[str, float] = {}
    for (tokens, correct_id, wrong_id), name in zip(packed, names):
        logits = model(tokens[None, :])
        last = logits[0, -1, :]
        mx.eval(last)
        gaps[name] = float(last[correct_id].item() - last[wrong_id].item())
    return gaps


def _fitness(
    target_gaps: dict[str, float],
    control_gaps: dict[str, float],
    target_baseline: dict[str, float],
    control_baseline: dict[str, float],
    penalty: float,
) -> float:
    target_improvement = sum(target_gaps[n] - target_baseline[n] for n in target_baseline) / len(target_baseline)
    control_degradation = sum(max(0.0, control_baseline[n] - control_gaps[n]) for n in control_baseline) / max(1, len(control_baseline))
    return target_improvement - penalty * control_degradation


def _mean_gain(gaps: dict[str, float], baseline: dict[str, float]) -> float:
    return sum(gaps[n] - baseline[n] for n in baseline) / max(1, len(baseline))


def _control_loss(gaps: dict[str, float], baseline: dict[str, float]) -> float:
    return sum(max(0.0, baseline[n] - gaps[n]) for n in baseline) / max(1, len(baseline))


def _gain_std(gaps: dict[str, float], baseline: dict[str, float]) -> float:
    if not baseline:
        return 0.0
    gains = np.array([gaps[n] - baseline[n] for n in baseline], dtype=np.float64)
    return float(np.std(gains))


def _score_screen(
    target_gaps: dict[str, float],
    control_gaps: dict[str, float],
    target_baseline: dict[str, float],
    control_baseline: dict[str, float],
    control_penalty: float,
    consistency_penalty: float,
) -> float:
    return (
        _mean_gain(target_gaps, target_baseline)
        - control_penalty * _control_loss(control_gaps, control_baseline)
        - consistency_penalty * _gain_std(target_gaps, target_baseline)
    )


def _select_probe_partitions(
    probes: list[dict[str, Any]],
    max_target_probes: int,
    max_control_probes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    search = [probe for probe in probes if (probe.get("metadata") or {}).get("partition") == "search"]
    control = [probe for probe in probes if (probe.get("metadata") or {}).get("partition") == "control"]
    validation = [probe for probe in probes if (probe.get("metadata") or {}).get("partition") == "validation"]

    if not search:
        search = probes
    if not control:
        control = probes[len(search) : len(search) + max_control_probes] or probes[:max_control_probes]
    if not validation:
        validation = probes[: min(max_control_probes, len(probes))]

    return (
        search[: min(max_target_probes, len(search))],
        control[: min(max_control_probes, len(control))],
        validation[: min(max_control_probes, len(validation))],
    )


def _build_candidates(
    model: Any,
    active_layers: list[int],
    search_projs: list[str],
    candidate_rows: int,
) -> list[tuple[int, str, int, float]]:
    candidates: list[tuple[int, str, int, float]] = []
    for layer in active_layers:
        for proj in search_projs:
            mod = get_module(model, f"model.layers.{layer}.mlp.{proj}")
            row_scales = np.array(mx.mean(mx.abs(mod.scales), axis=1))
            for row in range(min(candidate_rows, mod.weight.shape[0])):
                candidates.append((layer, proj, row, float(row_scales[row])))
    return candidates


def _candidate_weights(candidates: list[tuple[int, str, int, float]], impact_weighted: bool = False) -> np.ndarray:
    if impact_weighted:
        weights = np.array([max(item[3], 1e-6) / LAYER_IMPACT.get(item[0], 2.5) for item in candidates], dtype=np.float64)
    else:
        weights = np.array([max(item[3], 1e-6) for item in candidates], dtype=np.float64)
    weights /= weights.sum()
    return weights


def _resolve_search_layers(config: dict[str, Any], search_layers: list[int] | None, layer_profile: str | None) -> list[int]:
    if search_layers:
        return search_layers
    if layer_profile:
        if layer_profile not in LAYER_PROFILES:
            raise ValueError(f"Unknown layer profile: {layer_profile}. Choose one of: {', '.join(sorted(LAYER_PROFILES))}")
        return LAYER_PROFILES[layer_profile]
    return config["search"]["search_layers"]


def _sample_pool(
    candidates: list[tuple[int, str, int, float]],
    candidate_weights: np.ndarray,
    rng: np.random.Generator,
    tried: set[tuple[int, str, int]],
    pool_size: int,
) -> list[tuple[int, str, int, float]]:
    pool: list[tuple[int, str, int, float]] = []
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 40:
        idx = int(rng.choice(len(candidates), p=candidate_weights))
        candidate = candidates[idx]
        key = candidate[:3]
        if key not in tried:
            tried.add(key)
            pool.append(candidate)
        attempts += 1
    return pool


def _sample_pool_excluding(
    candidates: list[tuple[int, str, int, float]],
    candidate_weights: np.ndarray,
    rng: np.random.Generator,
    excluded: set[tuple[int, str, int]],
    pool_size: int,
) -> list[tuple[int, str, int, float]]:
    pool: list[tuple[int, str, int, float]] = []
    seen = set(excluded)
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 40:
        idx = int(rng.choice(len(candidates), p=candidate_weights))
        candidate = candidates[idx]
        key = candidate[:3]
        if key not in seen:
            seen.add(key)
            pool.append(candidate)
        attempts += 1
    return pool


def _candidate_to_flip(candidate: tuple[int, str, int, float]) -> PatchFlip:
    layer, proj, row, _ = candidate
    return PatchFlip(layer=layer, proj=proj, row=row)


def _flip_key(flip: PatchFlip) -> tuple[int, str, int]:
    return (flip.layer, flip.proj, flip.row)


def run_real_search(benchmark: str, model_ref: str, output_path: Path | None = None, max_iters: int | None = None) -> RealSearchResult:
    return _run_real_search(benchmark, model_ref, output_path=output_path, max_iters=max_iters)


def _run_real_search(
    benchmark: str,
    model_ref: str,
    output_path: Path | None = None,
    max_iters: int | None = None,
    max_target_probes: int = 64,
    max_control_probes: int = 8,
    search_layers: list[int] | None = None,
    layer_profile: str | None = None,
    impact_weighted: bool = False,
    verbose: bool = True,
) -> RealSearchResult:
    config = load_yaml(config_path(benchmark))
    probes = read_jsonl(probes_path(benchmark))
    handle = load_real_model(model_ref)
    summary = model_patchable_summary(handle.model)
    if not summary["patchable"]:
        raise RuntimeError("Loaded model does not expose Bankai-compatible uint32 row-packed MLP weights.")

    target_probes, control_probes, validation_probes = _select_probe_partitions(probes, max_target_probes, max_control_probes)
    packed_target = _pre_tokenize(handle.tokenizer, target_probes)
    packed_control = _pre_tokenize(handle.tokenizer, control_probes)
    target_names = [probe["name"] for probe in target_probes]
    control_names = [probe["name"] for probe in control_probes]
    validation_names = [probe["name"] for probe in validation_probes]
    packed_validation = _pre_tokenize(handle.tokenizer, validation_probes)

    target_baseline = _measure_fast(handle.model, packed_target, target_names)
    control_baseline = _measure_fast(handle.model, packed_control, control_names)
    validation_baseline = _measure_fast(handle.model, packed_validation, validation_names)

    active_layers = _resolve_search_layers(config, search_layers, layer_profile)

    candidates = _build_candidates(handle.model, active_layers, config["search"]["search_projs"], config["search"]["candidate_rows"])
    candidates.sort(key=lambda item: item[3], reverse=True)
    candidate_weights = _candidate_weights(candidates, impact_weighted=impact_weighted)
    rng = np.random.default_rng(benchmark.__hash__() & 0xFFFFFFFF)

    accepted: list[PatchFlip] = []
    current_fitness = 0.0
    trajectory: list[dict[str, Any]] = []
    screened_out = 0
    iterations = max_iters or config["search"]["iterations"]
    tried: set[tuple[int, str, int]] = set()

    screen_names = [name for name, _ in sorted(target_baseline.items(), key=lambda kv: kv[1])[: min(2, len(target_baseline))]]
    screen_indices = [target_names.index(name) for name in screen_names]
    screen_packed = [packed_target[i] for i in screen_indices]

    if verbose:
        print(
            f"real-search benchmark={benchmark} target_probes={len(target_probes)} "
            f"control_probes={len(control_probes)} candidates={len(candidates)} iterations={iterations}"
        )

    for step in range(iterations):
        attempts = 0
        while True:
            idx = int(rng.choice(len(candidates), p=candidate_weights))
            layer, proj, row, scale = candidates[idx]
            key = (layer, proj, row)
            if key not in tried:
                tried.add(key)
                break
            attempts += 1
            if attempts > len(candidates):
                raise RuntimeError("Exhausted candidate pool during real search.")

        flip_row(handle.model, layer, proj, row)
        mx.eval(handle.model.parameters())

        screen_gaps = _measure_fast(handle.model, screen_packed, screen_names)
        if not any(screen_gaps[name] > target_baseline[name] for name in screen_names):
            flip_row(handle.model, layer, proj, row)
            screened_out += 1
            trajectory.append({"iteration": step, "layer": layer, "proj": proj, "row": row, "scale_mean": scale, "screened_out": True})
            if verbose:
                print(f"[{step+1}/{iterations}] screen-reject L{layer}.{proj}[{row}] scale={scale:.6f}")
            continue

        target_gaps = _measure_fast(handle.model, packed_target, target_names)
        control_gaps = _measure_fast(handle.model, packed_control, control_names)
        validation_gaps = _measure_fast(handle.model, packed_validation, validation_names)
        fitness = _fitness(target_gaps, control_gaps, target_baseline, control_baseline, config["search"]["control_penalty"])
        validation_gain = sum(validation_gaps[n] - validation_baseline[n] for n in validation_baseline) / max(1, len(validation_baseline))
        accepted_flag = fitness > current_fitness
        trajectory.append(
            {
                "iteration": step,
                "layer": layer,
                "proj": proj,
                "row": row,
                "scale_mean": scale,
                "candidate_score": fitness,
                "validation_gain": validation_gain,
                "accepted": accepted_flag,
            }
        )

        if accepted_flag and len(accepted) < config["search"]["max_flips"]:
            accepted.append(PatchFlip(layer=layer, proj=proj, row=row))
            current_fitness = fitness
            if verbose:
                print(
                    f"[{step+1}/{iterations}] ACCEPT L{layer}.{proj}[{row}] "
                    f"scale={scale:.6f} fitness={fitness:+.6f} flips={len(accepted)}"
                )
        else:
            flip_row(handle.model, layer, proj, row)
            if verbose:
                print(
                    f"[{step+1}/{iterations}] reject L{layer}.{proj}[{row}] "
                    f"scale={scale:.6f} fitness={fitness:+.6f}"
                )

    patch = BankaiPatch(
        name=f"{benchmark}_real_patch",
        description=f"Real MLX Bankai-style search result for {benchmark}.",
        base_model=model_ref,
        flips=accepted,
        metadata={
            "benchmark": benchmark,
            "search_algorithm": "real_greedy_hill_climbing_screened",
            "search_layers": active_layers,
            "layer_profile": layer_profile,
            "impact_weighted": impact_weighted,
            "search_projs": config["search"]["search_projs"],
            "control_penalty": config["search"]["control_penalty"],
            "final_fitness": current_fitness,
            "screened_out": screened_out,
            "target_probe_count": len(target_probes),
            "control_probe_count": len(control_probes),
            "validation_probe_count": len(validation_probes),
            "model_summary": summary,
        },
    )
    output_path = output_path or patch_path(benchmark).with_name(f"{benchmark}_real_patch.json")
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.parent.parent / "results" / f"{benchmark}_real_search.json",
        {"benchmark": benchmark, "patch_path": str(output_path), "best_score": current_fitness, "trajectory": trajectory},
    )
    return RealSearchResult(patch=patch, best_score=current_fitness, trajectory=trajectory)


def run_real_shortlist_search(
    benchmark: str,
    model_ref: str,
    output_path: Path | None = None,
    rounds: int = 6,
    shortlist_pool: int = 16,
    shortlist_topk: int = 4,
    max_target_probes: int = 6,
    max_control_probes: int = 3,
    search_layers: list[int] | None = None,
    layer_profile: str | None = None,
    impact_weighted: bool = False,
    verbose: bool = True,
) -> RealSearchResult:
    config = load_yaml(config_path(benchmark))
    probes = read_jsonl(probes_path(benchmark))
    handle = load_real_model(model_ref)
    summary = model_patchable_summary(handle.model)
    if not summary["patchable"]:
        raise RuntimeError("Loaded model does not expose Bankai-compatible uint32 row-packed MLP weights.")

    target_probes, control_probes, validation_probes = _select_probe_partitions(probes, max_target_probes, max_control_probes)
    packed_target = _pre_tokenize(handle.tokenizer, target_probes)
    packed_control = _pre_tokenize(handle.tokenizer, control_probes)
    target_names = [probe["name"] for probe in target_probes]
    control_names = [probe["name"] for probe in control_probes]
    validation_names = [probe["name"] for probe in validation_probes]
    packed_validation = _pre_tokenize(handle.tokenizer, validation_probes)

    target_baseline = _measure_fast(handle.model, packed_target, target_names)
    control_baseline = _measure_fast(handle.model, packed_control, control_names)
    validation_baseline = _measure_fast(handle.model, packed_validation, validation_names)

    active_layers = _resolve_search_layers(config, search_layers, layer_profile)

    candidates = _build_candidates(handle.model, active_layers, config["search"]["search_projs"], config["search"]["candidate_rows"])
    candidate_weights = _candidate_weights(candidates, impact_weighted=impact_weighted)
    rng = np.random.default_rng(abs(hash((benchmark, "shortlist"))) & 0xFFFFFFFF)

    screen_names = [name for name, _ in sorted(target_baseline.items(), key=lambda kv: kv[1])[: min(2, len(target_baseline))]]
    screen_indices = [target_names.index(name) for name in screen_names]
    screen_packed = [packed_target[i] for i in screen_indices]

    accepted: list[PatchFlip] = []
    tried: set[tuple[int, str, int]] = set()
    trajectory: list[dict[str, Any]] = []
    current_fitness = 0.0

    if verbose:
        print(
            f"shortlist-search benchmark={benchmark} target_probes={len(target_probes)} "
            f"control_probes={len(control_probes)} rounds={rounds} pool={shortlist_pool} topk={shortlist_topk}"
        )

    for round_idx in range(rounds):
        pool = _sample_pool(candidates, candidate_weights, rng, tried, shortlist_pool)
        if not pool:
            break

        screened: list[tuple[float, tuple[int, str, int, float]]] = []
        for layer, proj, row, scale in pool:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            screen_gaps = _measure_fast(handle.model, screen_packed, screen_names)
            screen_gain = sum(screen_gaps[name] - target_baseline[name] for name in screen_names) / len(screen_names)
            flip_row(handle.model, layer, proj, row)
            screened.append((screen_gain, (layer, proj, row, scale)))

        screened.sort(key=lambda item: item[0], reverse=True)
        finalists = screened[: min(shortlist_topk, len(screened))]
        if verbose:
            best_screen_gain = finalists[0][0] if finalists else float("-inf")
            print(f"[round {round_idx+1}/{rounds}] screened {len(pool)} candidates best_screen_gain={best_screen_gain:+.6f}")

        best_round_candidate: tuple[int, str, int, float] | None = None
        best_round_fitness = current_fitness
        for screen_gain, (layer, proj, row, scale) in finalists:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            target_gaps = _measure_fast(handle.model, packed_target, target_names)
            control_gaps = _measure_fast(handle.model, packed_control, control_names)
            validation_gaps = _measure_fast(handle.model, packed_validation, validation_names)
            fitness = _fitness(target_gaps, control_gaps, target_baseline, control_baseline, config["search"]["control_penalty"])
            validation_gain = sum(validation_gaps[n] - validation_baseline[n] for n in validation_baseline) / max(1, len(validation_baseline))
            flip_row(handle.model, layer, proj, row)

            accepted_flag = fitness > best_round_fitness
            trajectory.append(
                {
                    "round": round_idx,
                    "layer": layer,
                    "proj": proj,
                    "row": row,
                    "scale_mean": scale,
                    "screen_gain": screen_gain,
                    "candidate_score": fitness,
                    "validation_gain": validation_gain,
                    "accepted": accepted_flag,
                }
            )
            if verbose:
                state = "ACCEPT-CANDIDATE" if accepted_flag else "reject"
                print(
                    f"  {state} L{layer}.{proj}[{row}] scale={scale:.6f} "
                    f"screen={screen_gain:+.6f} fitness={fitness:+.6f}"
                )
            if accepted_flag:
                best_round_fitness = fitness
                best_round_candidate = (layer, proj, row, scale)

        if best_round_candidate is not None and len(accepted) < config["search"]["max_flips"]:
            layer, proj, row, scale = best_round_candidate
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            accepted.append(PatchFlip(layer=layer, proj=proj, row=row))
            current_fitness = best_round_fitness
            if verbose:
                print(
                    f"[round {round_idx+1}/{rounds}] ACCEPTED L{layer}.{proj}[{row}] "
                    f"fitness={current_fitness:+.6f} flips={len(accepted)}"
                )
        elif verbose:
            print(f"[round {round_idx+1}/{rounds}] no improving candidate")

    patch = BankaiPatch(
        name=f"{benchmark}_real_patch",
        description=f"Real MLX shortlist search result for {benchmark}.",
        base_model=model_ref,
        flips=accepted,
        metadata={
            "benchmark": benchmark,
            "search_algorithm": "real_shortlist_search",
            "search_layers": active_layers,
            "layer_profile": layer_profile,
            "impact_weighted": impact_weighted,
            "search_projs": config["search"]["search_projs"],
            "control_penalty": config["search"]["control_penalty"],
            "final_fitness": current_fitness,
            "target_probe_count": len(target_probes),
            "control_probe_count": len(control_probes),
            "validation_probe_count": len(validation_probes),
            "rounds": rounds,
            "shortlist_pool": shortlist_pool,
            "shortlist_topk": shortlist_topk,
            "model_summary": summary,
        },
    )
    output_path = output_path or patch_path(benchmark).with_name(f"{benchmark}_real_patch.json")
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.parent.parent / "results" / f"{benchmark}_real_search.json",
        {
            "benchmark": benchmark,
            "patch_path": str(output_path),
            "best_score": current_fitness,
            "trajectory": trajectory,
            "mode": "shortlist",
        },
    )
    return RealSearchResult(patch=patch, best_score=current_fitness, trajectory=trajectory)


def run_real_two_pass_search(
    benchmark: str,
    model_ref: str,
    output_path: Path | None = None,
    rounds: int = 4,
    shortlist_pool: int = 32,
    mid_topk: int = 8,
    shortlist_topk: int = 2,
    max_target_probes: int = 6,
    max_control_probes: int = 3,
    pass2_target_probes: int = 4,
    pass2_control_probes: int = 2,
    search_layers: list[int] | None = None,
    layer_profile: str | None = None,
    impact_weighted: bool = False,
    consistency_penalty: float = 0.5,
    verbose: bool = True,
) -> RealSearchResult:
    config = load_yaml(config_path(benchmark))
    probes = read_jsonl(probes_path(benchmark))
    handle = load_real_model(model_ref)
    summary = model_patchable_summary(handle.model)
    if not summary["patchable"]:
        raise RuntimeError("Loaded model does not expose Bankai-compatible uint32 row-packed MLP weights.")

    target_probes, control_probes, validation_probes = _select_probe_partitions(probes, max_target_probes, max_control_probes)
    packed_target = _pre_tokenize(handle.tokenizer, target_probes)
    packed_control = _pre_tokenize(handle.tokenizer, control_probes)
    packed_validation = _pre_tokenize(handle.tokenizer, validation_probes)
    target_names = [probe["name"] for probe in target_probes]
    control_names = [probe["name"] for probe in control_probes]
    validation_names = [probe["name"] for probe in validation_probes]

    target_baseline = _measure_fast(handle.model, packed_target, target_names)
    control_baseline = _measure_fast(handle.model, packed_control, control_names)
    validation_baseline = _measure_fast(handle.model, packed_validation, validation_names)

    screen_names = [name for name, _ in sorted(target_baseline.items(), key=lambda kv: kv[1])[: min(2, len(target_baseline))]]
    screen_indices = [target_names.index(name) for name in screen_names]
    screen_packed = [packed_target[i] for i in screen_indices]
    screen_baseline = {name: target_baseline[name] for name in screen_names}

    pass2_target_names = target_names[: min(pass2_target_probes, len(target_names))]
    pass2_control_names = control_names[: min(pass2_control_probes, len(control_names))]
    pass2_target_packed = packed_target[: len(pass2_target_names)]
    pass2_control_packed = packed_control[: len(pass2_control_names)]
    pass2_target_baseline = {name: target_baseline[name] for name in pass2_target_names}
    pass2_control_baseline = {name: control_baseline[name] for name in pass2_control_names}

    active_layers = _resolve_search_layers(config, search_layers, layer_profile)
    candidates = _build_candidates(handle.model, active_layers, config["search"]["search_projs"], config["search"]["candidate_rows"])
    candidate_weights = _candidate_weights(candidates, impact_weighted=impact_weighted)
    rng = np.random.default_rng(abs(hash((benchmark, "two-pass"))) & 0xFFFFFFFF)

    accepted: list[PatchFlip] = []
    tried: set[tuple[int, str, int]] = set()
    trajectory: list[dict[str, Any]] = []
    current_fitness = 0.0
    control_penalty = config["search"]["control_penalty"]

    if verbose:
        print(
            f"two-pass-search benchmark={benchmark} target_probes={len(target_probes)} "
            f"control_probes={len(control_probes)} rounds={rounds} pool={shortlist_pool} "
            f"mid_topk={mid_topk} topk={shortlist_topk}"
        )

    for round_idx in range(rounds):
        pool = _sample_pool(candidates, candidate_weights, rng, tried, shortlist_pool)
        if not pool:
            break

        first_pass: list[tuple[float, tuple[int, str, int, float]]] = []
        for layer, proj, row, scale in pool:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            gaps = _measure_fast(handle.model, screen_packed, screen_names)
            flip_row(handle.model, layer, proj, row)
            score = _mean_gain(gaps, screen_baseline)
            first_pass.append((score, (layer, proj, row, scale)))

        first_pass.sort(key=lambda item: item[0], reverse=True)
        mid_candidates = first_pass[: min(mid_topk, len(first_pass))]

        second_pass: list[tuple[float, float, tuple[int, str, int, float]]] = []
        for pass1_score, (layer, proj, row, scale) in mid_candidates:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            target_gaps = _measure_fast(handle.model, pass2_target_packed, pass2_target_names)
            control_gaps = _measure_fast(handle.model, pass2_control_packed, pass2_control_names)
            flip_row(handle.model, layer, proj, row)
            score = _score_screen(
                target_gaps,
                control_gaps,
                pass2_target_baseline,
                pass2_control_baseline,
                control_penalty,
                consistency_penalty,
            )
            second_pass.append((score, pass1_score, (layer, proj, row, scale)))

        second_pass.sort(key=lambda item: item[0], reverse=True)
        finalists = second_pass[: min(shortlist_topk, len(second_pass))]
        if verbose:
            best_first = first_pass[0][0] if first_pass else float("-inf")
            best_second = finalists[0][0] if finalists else float("-inf")
            print(
                f"[round {round_idx+1}/{rounds}] screened {len(pool)} candidates "
                f"best_pass1={best_first:+.6f} best_pass2={best_second:+.6f}"
            )

        best_round_candidate: tuple[int, str, int, float] | None = None
        best_round_fitness = current_fitness
        for pass2_score, pass1_score, (layer, proj, row, scale) in finalists:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            target_gaps = _measure_fast(handle.model, packed_target, target_names)
            control_gaps = _measure_fast(handle.model, packed_control, control_names)
            validation_gaps = _measure_fast(handle.model, packed_validation, validation_names)
            fitness = _fitness(target_gaps, control_gaps, target_baseline, control_baseline, control_penalty)
            validation_gain = _mean_gain(validation_gaps, validation_baseline)
            flip_row(handle.model, layer, proj, row)

            accepted_flag = fitness > best_round_fitness
            trajectory.append(
                {
                    "round": round_idx,
                    "layer": layer,
                    "proj": proj,
                    "row": row,
                    "scale_mean": scale,
                    "pass1_score": pass1_score,
                    "pass2_score": pass2_score,
                    "candidate_score": fitness,
                    "validation_gain": validation_gain,
                    "accepted": accepted_flag,
                }
            )
            if verbose:
                state = "ACCEPT-CANDIDATE" if accepted_flag else "reject"
                print(
                    f"  {state} L{layer}.{proj}[{row}] scale={scale:.6f} "
                    f"p1={pass1_score:+.6f} p2={pass2_score:+.6f} fitness={fitness:+.6f}"
                )
            if accepted_flag:
                best_round_fitness = fitness
                best_round_candidate = (layer, proj, row, scale)

        if best_round_candidate is not None and len(accepted) < config["search"]["max_flips"]:
            layer, proj, row, scale = best_round_candidate
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            accepted.append(PatchFlip(layer=layer, proj=proj, row=row))
            current_fitness = best_round_fitness
            if verbose:
                print(
                    f"[round {round_idx+1}/{rounds}] ACCEPTED L{layer}.{proj}[{row}] "
                    f"fitness={current_fitness:+.6f} flips={len(accepted)}"
                )
        elif verbose:
            print(f"[round {round_idx+1}/{rounds}] no improving candidate")

    patch = BankaiPatch(
        name=f"{benchmark}_real_patch",
        description=f"Real MLX two-pass shortlist search result for {benchmark}.",
        base_model=model_ref,
        flips=accepted,
        metadata={
            "benchmark": benchmark,
            "search_algorithm": "real_two_pass_shortlist_search",
            "search_layers": active_layers,
            "layer_profile": layer_profile,
            "impact_weighted": impact_weighted,
            "search_projs": config["search"]["search_projs"],
            "control_penalty": control_penalty,
            "consistency_penalty": consistency_penalty,
            "final_fitness": current_fitness,
            "target_probe_count": len(target_probes),
            "control_probe_count": len(control_probes),
            "validation_probe_count": len(validation_probes),
            "rounds": rounds,
            "shortlist_pool": shortlist_pool,
            "mid_topk": mid_topk,
            "shortlist_topk": shortlist_topk,
            "pass2_target_probes": len(pass2_target_names),
            "pass2_control_probes": len(pass2_control_names),
            "model_summary": summary,
        },
    )
    output_path = output_path or patch_path(benchmark).with_name(f"{benchmark}_real_patch.json")
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.parent.parent / "results" / f"{benchmark}_real_search.json",
        {
            "benchmark": benchmark,
            "patch_path": str(output_path),
            "best_score": current_fitness,
            "trajectory": trajectory,
            "mode": "two_pass_shortlist",
        },
    )
    return RealSearchResult(patch=patch, best_score=current_fitness, trajectory=trajectory)


def run_real_anneal_shortlist_search(
    benchmark: str,
    model_ref: str,
    output_path: Path | None = None,
    steps: int = 24,
    shortlist_pool: int = 16,
    shortlist_topk: int = 4,
    max_target_probes: int = 6,
    max_control_probes: int = 3,
    search_layers: list[int] | None = None,
    layer_profile: str | None = None,
    impact_weighted: bool = False,
    start_temp: float = 0.02,
    end_temp: float = 0.001,
    remove_prob: float = 0.15,
    swap_prob: float = 0.45,
    verbose: bool = True,
) -> RealSearchResult:
    config = load_yaml(config_path(benchmark))
    probes = read_jsonl(probes_path(benchmark))
    handle = load_real_model(model_ref)
    summary = model_patchable_summary(handle.model)
    if not summary["patchable"]:
        raise RuntimeError("Loaded model does not expose Bankai-compatible uint32 row-packed MLP weights.")

    target_probes, control_probes, validation_probes = _select_probe_partitions(probes, max_target_probes, max_control_probes)
    packed_target = _pre_tokenize(handle.tokenizer, target_probes)
    packed_control = _pre_tokenize(handle.tokenizer, control_probes)
    packed_validation = _pre_tokenize(handle.tokenizer, validation_probes)
    target_names = [probe["name"] for probe in target_probes]
    control_names = [probe["name"] for probe in control_probes]
    validation_names = [probe["name"] for probe in validation_probes]

    target_baseline = _measure_fast(handle.model, packed_target, target_names)
    control_baseline = _measure_fast(handle.model, packed_control, control_names)
    validation_baseline = _measure_fast(handle.model, packed_validation, validation_names)

    screen_names = [name for name, _ in sorted(target_baseline.items(), key=lambda kv: kv[1])[: min(2, len(target_baseline))]]
    screen_indices = [target_names.index(name) for name in screen_names]
    screen_packed = [packed_target[i] for i in screen_indices]
    screen_baseline = {name: target_baseline[name] for name in screen_names}

    active_layers = _resolve_search_layers(config, search_layers, layer_profile)
    candidates = _build_candidates(handle.model, active_layers, config["search"]["search_projs"], config["search"]["candidate_rows"])
    candidate_weights = _candidate_weights(candidates, impact_weighted=impact_weighted)
    rng = np.random.default_rng(abs(hash((benchmark, "anneal-shortlist"))) & 0xFFFFFFFF)

    max_flips = int(config["search"]["max_flips"])
    control_penalty = float(config["search"]["control_penalty"])
    active: list[PatchFlip] = []
    current_fitness = 0.0
    best: list[PatchFlip] = []
    best_fitness = 0.0
    trajectory: list[dict[str, Any]] = []

    def choose_add_candidate(excluded: set[tuple[int, str, int]]) -> tuple[float, tuple[int, str, int, float]] | None:
        pool = _sample_pool_excluding(candidates, candidate_weights, rng, excluded, shortlist_pool)
        if not pool:
            return None
        screened: list[tuple[float, tuple[int, str, int, float]]] = []
        for candidate in pool:
            layer, proj, row, scale = candidate
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            gaps = _measure_fast(handle.model, screen_packed, screen_names)
            flip_row(handle.model, layer, proj, row)
            screened.append((_mean_gain(gaps, screen_baseline), (layer, proj, row, scale)))
        screened.sort(key=lambda item: item[0], reverse=True)
        finalists = screened[: min(shortlist_topk, len(screened))]
        return finalists[int(rng.integers(0, len(finalists)))] if finalists else None

    def measure_state() -> tuple[float, float]:
        target_gaps = _measure_fast(handle.model, packed_target, target_names)
        control_gaps = _measure_fast(handle.model, packed_control, control_names)
        validation_gaps = _measure_fast(handle.model, packed_validation, validation_names)
        fitness = _fitness(target_gaps, control_gaps, target_baseline, control_baseline, control_penalty)
        validation_gain = _mean_gain(validation_gaps, validation_baseline)
        return fitness, validation_gain

    if verbose:
        print(
            f"anneal-shortlist-search benchmark={benchmark} target_probes={len(target_probes)} "
            f"control_probes={len(control_probes)} steps={steps} pool={shortlist_pool} topk={shortlist_topk} "
            f"temp={start_temp}->{end_temp}"
        )

    for step in range(steps):
        progress = step / max(1, steps - 1)
        temperature = start_temp * ((end_temp / start_temp) ** progress) if start_temp > 0 and end_temp > 0 else 0.0
        active_keys = {_flip_key(flip) for flip in active}
        roll = float(rng.random())
        if not active or (len(active) < max_flips and roll >= remove_prob + swap_prob):
            move = "add"
        elif len(active) >= max_flips or roll < swap_prob:
            move = "swap"
        else:
            move = "remove"

        removed: PatchFlip | None = None
        added_candidate: tuple[int, str, int, float] | None = None
        screen_gain: float | None = None

        if move in {"remove", "swap"} and active:
            removed_idx = int(rng.integers(0, len(active)))
            removed = active.pop(removed_idx)
            flip_row(handle.model, removed.layer, removed.proj, removed.row)
            mx.eval(handle.model.parameters())
            active_keys.remove(_flip_key(removed))

        if move in {"add", "swap"}:
            chosen = choose_add_candidate(active_keys)
            if chosen is None:
                if removed is not None:
                    flip_row(handle.model, removed.layer, removed.proj, removed.row)
                    mx.eval(handle.model.parameters())
                    active.append(removed)
                trajectory.append({"step": step, "move": move, "skipped": True, "reason": "no_candidate"})
                continue
            screen_gain, added_candidate = chosen
            layer, proj, row, _ = added_candidate
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            active.append(_candidate_to_flip(added_candidate))

        proposed_fitness, validation_gain = measure_state()
        delta = proposed_fitness - current_fitness
        accept_prob = 1.0 if delta >= 0 else float(np.exp(delta / max(temperature, 1e-12)))
        accepted_flag = float(rng.random()) < accept_prob

        if accepted_flag:
            current_fitness = proposed_fitness
            if proposed_fitness > best_fitness:
                best_fitness = proposed_fitness
                best = [PatchFlip(layer=flip.layer, proj=flip.proj, row=flip.row) for flip in active]
        else:
            if added_candidate is not None:
                layer, proj, row, _ = added_candidate
                flip_row(handle.model, layer, proj, row)
                mx.eval(handle.model.parameters())
                active = [flip for flip in active if _flip_key(flip) != (layer, proj, row)]
            if removed is not None:
                flip_row(handle.model, removed.layer, removed.proj, removed.row)
                mx.eval(handle.model.parameters())
                active.append(removed)

        trajectory.append(
            {
                "step": step,
                "move": move,
                "temperature": temperature,
                "screen_gain": screen_gain,
                "candidate_score": proposed_fitness,
                "current_fitness": current_fitness,
                "best_fitness": best_fitness,
                "validation_gain": validation_gain,
                "delta": delta,
                "accept_prob": accept_prob,
                "accepted": accepted_flag,
                "active_flips": len(active),
                "added": None
                if added_candidate is None
                else {"layer": added_candidate[0], "proj": added_candidate[1], "row": added_candidate[2], "scale_mean": added_candidate[3]},
                "removed": None if removed is None else {"layer": removed.layer, "proj": removed.proj, "row": removed.row},
            }
        )
        if verbose:
            added_label = "" if added_candidate is None else f" +L{added_candidate[0]}.{added_candidate[1]}[{added_candidate[2]}]"
            removed_label = "" if removed is None else f" -L{removed.layer}.{removed.proj}[{removed.row}]"
            state = "ACCEPT" if accepted_flag else "reject"
            print(
                f"[{step+1}/{steps}] {state} {move}{removed_label}{added_label} "
                f"score={proposed_fitness:+.6f} current={current_fitness:+.6f} "
                f"best={best_fitness:+.6f} temp={temperature:.6f} p={accept_prob:.3f}"
            )

    patch = BankaiPatch(
        name=f"{benchmark}_real_patch",
        description=f"Real MLX simulated-annealing shortlist search result for {benchmark}.",
        base_model=model_ref,
        flips=best,
        metadata={
            "benchmark": benchmark,
            "search_algorithm": "real_anneal_shortlist_search",
            "search_layers": active_layers,
            "layer_profile": layer_profile,
            "impact_weighted": impact_weighted,
            "search_projs": config["search"]["search_projs"],
            "control_penalty": control_penalty,
            "final_fitness": best_fitness,
            "target_probe_count": len(target_probes),
            "control_probe_count": len(control_probes),
            "validation_probe_count": len(validation_probes),
            "steps": steps,
            "shortlist_pool": shortlist_pool,
            "shortlist_topk": shortlist_topk,
            "start_temp": start_temp,
            "end_temp": end_temp,
            "remove_prob": remove_prob,
            "swap_prob": swap_prob,
            "model_summary": summary,
        },
    )
    output_path = output_path or patch_path(benchmark).with_name(f"{benchmark}_real_patch.json")
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.parent.parent / "results" / f"{benchmark}_real_search.json",
        {
            "benchmark": benchmark,
            "patch_path": str(output_path),
            "best_score": best_fitness,
            "trajectory": trajectory,
            "mode": "anneal_shortlist",
        },
    )
    return RealSearchResult(patch=patch, best_score=best_fitness, trajectory=trajectory)
