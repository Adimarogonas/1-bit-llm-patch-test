from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from bankai_poc.data.registry import config_path, patch_path, probes_path
from bankai_poc.model.patching import BankaiPatch, PatchFlip, save_patch
from bankai_poc.model.real_mlx import apply_real_patch, flip_row, get_module, load_real_model, model_patchable_summary
from bankai_poc.search.probe_eval import compute_fitness, compute_fitness_min
from bankai_poc.utils.artifacts import save_run_manifest
from bankai_poc.utils.io import load_yaml, read_jsonl


DEFAULT_SEARCH_LAYERS: list[int] = [0, 1, 2, 3, 4, 34, 35]
DEFAULT_SEARCH_PROJS: list[str] = ["gate_proj", "up_proj", "down_proj"]

LAYER_PROFILES: dict[str, list[int]] = {
    "stable": DEFAULT_SEARCH_LAYERS,
    "balanced": [0, 1, 2, 3, 4, 22, 24, 28, 32, 34, 35],
    "aggressive": [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 34, 35],
}

LAYER_IMPACT: dict[int, float] = {
    **{layer: 5.2 for layer in range(0, 5)},
    **{layer: 1.9 for layer in range(5, 17)},
    **{layer: 1.1 for layer in range(17, 22)},
    **{layer: 2.5 for layer in range(22, 34)},
    34: 9.0,
    35: 3.2,
}

DEFAULT_DYNAMIC_SEARCH_CONFIG: dict[str, Any] = {
    "search": {
        "candidate_rows": 48,
        "max_flips": 16,
        "search_layers": DEFAULT_SEARCH_LAYERS,
        "search_projs": DEFAULT_SEARCH_PROJS,
        "control_penalty": 2.0,
    }
}


@dataclass
class RealSearchResult:
    patch: BankaiPatch
    best_score: float
    trajectory: list[dict[str, Any]]


PackedProbe = tuple[mx.array, tuple[int, ...], tuple[int, ...]]


def _pre_tokenize(tokenizer: Any, probes: list[dict[str, Any]]) -> list[PackedProbe]:
    packed = []
    for probe in probes:
        prompt = probe["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        correct_completion = probe.get("correct_completion")
        wrong_completion = probe.get("wrong_completion")
        if correct_completion is not None and wrong_completion is not None:
            prefix_ids, correct_ids, wrong_ids = _divergent_token_suffixes(
                tokenizer,
                prompt,
                str(correct_completion),
                str(wrong_completion),
            )
            tokens = mx.array(prefix_ids)
        else:
            tokens = mx.array(prompt_ids)
            correct_ids = tuple(tokenizer.encode(probe["correct_token"])[:1])
            wrong_ids = tuple(tokenizer.encode(probe["wrong_token"])[:1])
        packed.append((tokens, tuple(correct_ids), tuple(wrong_ids)))
    return packed


def _divergent_token_suffixes(
    tokenizer: Any,
    prompt: str,
    correct_completion: str,
    wrong_completion: str,
) -> tuple[list[int], tuple[int, ...], tuple[int, ...]]:
    prompt_ids = tokenizer.encode(prompt)
    correct_ids = _continuation_ids(tokenizer, prompt, correct_completion, prompt_ids)
    wrong_ids = _continuation_ids(tokenizer, prompt, wrong_completion, prompt_ids)
    if not correct_ids:
        correct_ids = tokenizer.encode(correct_completion)
    if not wrong_ids:
        wrong_ids = tokenizer.encode(wrong_completion)

    common = 0
    while common < min(len(correct_ids), len(wrong_ids)) and correct_ids[common] == wrong_ids[common]:
        common += 1
    if common >= len(correct_ids) or common >= len(wrong_ids):
        common = 0
    return prompt_ids + correct_ids[:common], tuple(correct_ids[common:]), tuple(wrong_ids[common:])


def _continuation_ids(tokenizer: Any, prompt: str, completion: str, prompt_ids: list[int]) -> list[int]:
    combined_ids = tokenizer.encode(prompt + completion)
    if combined_ids[: len(prompt_ids)] == prompt_ids:
        return combined_ids[len(prompt_ids) :]
    return tokenizer.encode(completion)


def _measure_fast(model: Any, packed: list[PackedProbe], names: list[str]) -> dict[str, float]:
    gaps: dict[str, float] = {}
    for (tokens, correct_ids, wrong_ids), name in zip(packed, names):
        correct_score = _sequence_mean_logprob(model, tokens, correct_ids)
        wrong_score = _sequence_mean_logprob(model, tokens, wrong_ids)
        gaps[name] = correct_score - wrong_score
    return gaps


def _sequence_mean_logprob(model: Any, prefix_tokens: mx.array, continuation_ids: tuple[int, ...]) -> float:
    if not continuation_ids:
        return 0.0
    if len(continuation_ids) == 1:
        logits = model(prefix_tokens[None, :])
        last = logits[0, -1, :]
        log_probs = last - mx.logsumexp(last, axis=-1)
        mx.eval(log_probs)
        return float(log_probs[continuation_ids[0]].item())

    continuation = mx.array(continuation_ids[:-1])
    tokens = mx.concatenate([prefix_tokens, continuation], axis=0)
    logits = model(tokens[None, :])
    start = int(prefix_tokens.shape[0]) - 1
    selected = logits[0, start : start + len(continuation_ids), :]
    log_probs = selected - mx.logsumexp(selected, axis=-1, keepdims=True)
    mx.eval(log_probs)
    total = 0.0
    for index, token_id in enumerate(continuation_ids):
        total += float(log_probs[index, token_id].item())
    return total / len(continuation_ids)


def _fitness(
    target_gaps: dict[str, float],
    control_gaps: dict[str, float],
    target_baseline: dict[str, float],
    control_baseline: dict[str, float],
    penalty: float,
) -> float:
    return compute_fitness(target_gaps, control_gaps, target_baseline, control_baseline, penalty)


def _fitness_min(
    target_gaps: dict[str, float],
    control_gaps: dict[str, float],
    target_baseline: dict[str, float],
    control_baseline: dict[str, float],
    penalty: float,
) -> float:
    return compute_fitness_min(target_gaps, control_gaps, target_baseline, control_baseline, penalty)


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
            if candidate_rows <= 0:
                rows = np.arange(mod.weight.shape[0])
            else:
                row_limit = min(candidate_rows, mod.weight.shape[0])
                rows = np.argsort(row_scales)[-row_limit:][::-1]
            for row in rows:
                row_index = int(row)
                candidates.append((int(layer), proj, row_index, float(row_scales[row_index])))
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


def _save_patch_checkpoint(
    output_path: Path,
    name: str,
    description: str,
    model_ref: str,
    accepted: list[PatchFlip],
    metadata: dict[str, Any],
) -> None:
    checkpoint_path = output_path.with_name(f"{output_path.stem}.checkpoint.json")
    checkpoint = BankaiPatch(
        name=name,
        description=description,
        base_model=model_ref,
        flips=list(accepted),
        metadata={**metadata, "checkpoint": True},
    )
    save_patch(checkpoint_path, checkpoint)


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


def run_real_shortlist_search_from_probes(
    task_name: str,
    model_ref: str,
    probes: list[dict[str, Any]],
    output_path: Path,
    rounds: int = 4,
    shortlist_pool: int = 8,
    shortlist_topk: int = 2,
    accept_per_round: int = 1,
    max_target_probes: int = 12,
    max_control_probes: int = 6,
    candidate_rows: int = 48,
    max_flips: int = 16,
    control_penalty: float = 2.0,
    search_layers: list[int] | None = None,
    layer_profile: str | None = None,
    impact_weighted: bool = True,
    verbose: bool = True,
) -> RealSearchResult:
    config = {
        "search": {
            **DEFAULT_DYNAMIC_SEARCH_CONFIG["search"],
            "candidate_rows": candidate_rows,
            "max_flips": max_flips,
            "control_penalty": control_penalty,
            "accept_per_round": max(1, accept_per_round),
        }
    }
    if verbose:
        print(f"loading model for dynamic patch search: {model_ref}", flush=True)
    handle = load_real_model(model_ref)
    if verbose:
        print("model loaded; checking patchable MLX row layout", flush=True)
    summary = model_patchable_summary(handle.model)
    if not summary["patchable"]:
        raise RuntimeError("Loaded model does not expose Bankai-compatible uint32 row-packed MLP weights.")

    if verbose:
        print(f"partitioning {len(probes)} dynamic probes", flush=True)
    target_probes, control_probes, validation_probes = _select_probe_partitions(probes, max_target_probes, max_control_probes)
    if verbose:
        print("tokenizing dynamic probes", flush=True)
    packed_target = _pre_tokenize(handle.tokenizer, target_probes)
    packed_control = _pre_tokenize(handle.tokenizer, control_probes)
    packed_validation = _pre_tokenize(handle.tokenizer, validation_probes)
    target_names = [probe["name"] for probe in target_probes]
    control_names = [probe["name"] for probe in control_probes]
    validation_names = [probe["name"] for probe in validation_probes]

    if verbose:
        print("measuring baseline logit gaps for target/control/validation probes", flush=True)
    target_baseline = _measure_fast(handle.model, packed_target, target_names)
    control_baseline = _measure_fast(handle.model, packed_control, control_names)
    validation_baseline = _measure_fast(handle.model, packed_validation, validation_names)

    active_layers = _resolve_search_layers(config, search_layers, layer_profile)
    if verbose:
        print(f"building row-flip candidates from layers={active_layers}", flush=True)
    candidates = _build_candidates(handle.model, active_layers, config["search"]["search_projs"], config["search"]["candidate_rows"])
    candidate_weights = _candidate_weights(candidates, impact_weighted=impact_weighted)
    rng = np.random.default_rng(abs(hash((task_name, "dynamic-shortlist"))) & 0xFFFFFFFF)

    screen_names = [name for name, _ in sorted(target_baseline.items(), key=lambda kv: kv[1])[: min(2, len(target_baseline))]]
    screen_indices = [target_names.index(name) for name in screen_names]
    screen_packed = [packed_target[i] for i in screen_indices]

    accepted: list[PatchFlip] = []
    tried: set[tuple[int, str, int]] = set()
    trajectory: list[dict[str, Any]] = []
    current_fitness = 0.0
    accept_limit = config["search"]["accept_per_round"]

    if verbose:
        print(
            f"dynamic-shortlist-search task={task_name} target_probes={len(target_probes)} "
            f"control_probes={len(control_probes)} validation_probes={len(validation_probes)} "
            f"rounds={rounds} pool={shortlist_pool} topk={shortlist_topk} accept_per_round={accept_limit} "
            f"candidate_rows={candidate_rows} max_flips={max_flips} candidates={len(candidates)} "
            f"control_penalty={control_penalty}",
            flush=True,
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
            print(f"[round {round_idx+1}/{rounds}] screened {len(pool)} candidates best_screen_gain={best_screen_gain:+.6f}", flush=True)

        remaining_finalists = list(finalists)
        accepted_this_round = 0
        while (
            remaining_finalists
            and accepted_this_round < accept_limit
            and len(accepted) < config["search"]["max_flips"]
        ):
            best_eval: dict[str, Any] | None = None
            rescored: list[dict[str, Any]] = []
            for screen_gain, (layer, proj, row, scale) in remaining_finalists:
                flip_row(handle.model, layer, proj, row)
                mx.eval(handle.model.parameters())
                target_gaps = _measure_fast(handle.model, packed_target, target_names)
                control_gaps = _measure_fast(handle.model, packed_control, control_names)
                validation_gaps = _measure_fast(handle.model, packed_validation, validation_names)
                fitness = _fitness(target_gaps, control_gaps, target_baseline, control_baseline, config["search"]["control_penalty"])
                validation_gain = _mean_gain(validation_gaps, validation_baseline)
                flip_row(handle.model, layer, proj, row)

                candidate_eval = {
                    "round": round_idx,
                    "accept_slot": accepted_this_round,
                    "layer": layer,
                    "proj": proj,
                    "row": row,
                    "scale_mean": scale,
                    "screen_gain": screen_gain,
                    "candidate_score": fitness,
                    "validation_gain": validation_gain,
                    "accepted": False,
                }
                rescored.append(candidate_eval)
                if fitness > current_fitness and (best_eval is None or fitness > best_eval["candidate_score"]):
                    best_eval = candidate_eval

            if best_eval is None:
                trajectory.extend(rescored)
                for candidate_eval in rescored:
                    if verbose:
                        print(
                            f"  reject L{candidate_eval['layer']}.{candidate_eval['proj']}[{candidate_eval['row']}] "
                            f"scale={candidate_eval['scale_mean']:.6f} "
                            f"screen={candidate_eval['screen_gain']:+.6f} "
                            f"fitness={candidate_eval['candidate_score']:+.6f}",
                            flush=True,
                        )
                break

            best_eval["accepted"] = True
            trajectory.extend(rescored)
            for candidate_eval in rescored:
                if verbose:
                    state = "ACCEPT-CANDIDATE" if candidate_eval is best_eval else "reject"
                    print(
                        f"  {state} L{candidate_eval['layer']}.{candidate_eval['proj']}[{candidate_eval['row']}] "
                        f"scale={candidate_eval['scale_mean']:.6f} "
                        f"screen={candidate_eval['screen_gain']:+.6f} "
                        f"fitness={candidate_eval['candidate_score']:+.6f}",
                        flush=True,
                    )

            layer = best_eval["layer"]
            proj = best_eval["proj"]
            row = best_eval["row"]
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            accepted.append(PatchFlip(layer=layer, proj=proj, row=row))
            _save_patch_checkpoint(
                output_path,
                f"{task_name}_dynamic_patch",
                f"Checkpoint for dynamic practical-eval Bankai patch for {task_name}.",
                model_ref,
                accepted,
                {
                    "task": task_name,
                    "search_algorithm": "dynamic_real_shortlist_search",
                    "search_layers": active_layers,
                    "layer_profile": layer_profile,
                    "impact_weighted": impact_weighted,
                    "final_fitness": best_eval["candidate_score"],
                    "round": round_idx,
                },
            )
            current_fitness = best_eval["candidate_score"]
            accepted_this_round += 1
            remaining_finalists = [
                item for item in remaining_finalists if item[1][:3] != (layer, proj, row)
            ]
            if verbose:
                print(
                    f"[round {round_idx+1}/{rounds}] ACCEPTED L{layer}.{proj}[{row}] "
                    f"fitness={current_fitness:+.6f} accepted_this_round={accepted_this_round}/{accept_limit} "
                    f"flips={len(accepted)}/{max_flips}",
                    flush=True,
                )

        if accepted_this_round == 0 and verbose:
            print(f"[round {round_idx+1}/{rounds}] no improving candidate", flush=True)

    patch = BankaiPatch(
        name=f"{task_name}_dynamic_patch",
        description=f"Dynamic practical-eval Bankai patch for {task_name}.",
        base_model=model_ref,
        flips=accepted,
        metadata={
            "benchmark": task_name,
            "task": task_name,
            "search_algorithm": "dynamic_real_shortlist_search",
            "search_layers": active_layers,
            "layer_profile": layer_profile,
            "impact_weighted": impact_weighted,
            "search_projs": config["search"]["search_projs"],
            "control_penalty": config["search"]["control_penalty"],
            "final_fitness": current_fitness,
            "candidate_rows": candidate_rows,
            "max_flips": max_flips,
            "target_probe_count": len(target_probes),
            "control_probe_count": len(control_probes),
            "validation_probe_count": len(validation_probes),
            "rounds": rounds,
            "shortlist_pool": shortlist_pool,
            "shortlist_topk": shortlist_topk,
            "accept_per_round": accept_limit,
            "model_summary": summary,
        },
    )
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.with_name("dynamic_search.json"),
        {
            "task": task_name,
            "patch_path": str(output_path),
            "best_score": current_fitness,
            "trajectory": trajectory,
            "mode": "dynamic_shortlist",
            "accept_per_round": accept_limit,
        },
    )
    return RealSearchResult(patch=patch, best_score=current_fitness, trajectory=trajectory)


def run_real_greedy_search_from_probes(
    task_name: str,
    model_ref: str,
    probes: list[dict[str, Any]],
    output_path: Path,
    max_iters: int = 200,
    fitness_mode: str = "mean",
    max_target_probes: int = 12,
    max_control_probes: int = 6,
    candidate_rows: int = 48,
    max_flips: int = 16,
    control_penalty: float = 2.0,
    search_layers: list[int] | None = None,
    layer_profile: str | None = None,
    impact_weighted: bool = True,
    verbose: bool = True,
) -> RealSearchResult:
    if fitness_mode not in {"mean", "min"}:
        raise ValueError("fitness_mode must be 'mean' or 'min'")

    config = {
        "search": {
            **DEFAULT_DYNAMIC_SEARCH_CONFIG["search"],
            "candidate_rows": candidate_rows,
            "max_flips": max_flips,
            "control_penalty": control_penalty,
        }
    }
    if verbose:
        print(f"loading model for dynamic greedy patch search: {model_ref}", flush=True)
    handle = load_real_model(model_ref)
    if verbose:
        print("model loaded; checking patchable MLX row layout", flush=True)
    summary = model_patchable_summary(handle.model)
    if not summary["patchable"]:
        raise RuntimeError("Loaded model does not expose Bankai-compatible uint32 row-packed MLP weights.")

    if verbose:
        print(f"partitioning {len(probes)} dynamic probes", flush=True)
    target_probes, control_probes, validation_probes = _select_probe_partitions(probes, max_target_probes, max_control_probes)
    if not target_probes:
        raise RuntimeError("Greedy search needs at least one target probe.")

    if verbose:
        print("tokenizing dynamic probes", flush=True)
    packed_target = _pre_tokenize(handle.tokenizer, target_probes)
    packed_control = _pre_tokenize(handle.tokenizer, control_probes)
    packed_validation = _pre_tokenize(handle.tokenizer, validation_probes)
    target_names = [probe["name"] for probe in target_probes]
    control_names = [probe["name"] for probe in control_probes]
    validation_names = [probe["name"] for probe in validation_probes]

    if verbose:
        print("measuring baseline logit gaps for target/control/validation probes", flush=True)
    target_baseline = _measure_fast(handle.model, packed_target, target_names)
    control_baseline = _measure_fast(handle.model, packed_control, control_names)
    validation_baseline = _measure_fast(handle.model, packed_validation, validation_names)

    sorted_targets = sorted(target_baseline.items(), key=lambda item: item[1])
    screen_names = [name for name, _ in sorted_targets[: min(2, len(sorted_targets))]]
    screen_indices = [target_names.index(name) for name in screen_names]
    screen_packed = [packed_target[index] for index in screen_indices]

    active_layers = _resolve_search_layers(config, search_layers, layer_profile)
    if verbose:
        print(f"building row-flip candidates from layers={active_layers}", flush=True)
    candidates = _build_candidates(handle.model, active_layers, config["search"]["search_projs"], config["search"]["candidate_rows"])
    candidate_weights = _candidate_weights(candidates, impact_weighted=impact_weighted)
    rng = np.random.default_rng(abs(hash((task_name, "dynamic-greedy", fitness_mode))) & 0xFFFFFFFF)
    fitness_fn = _fitness_min if fitness_mode == "min" else _fitness

    accepted: list[PatchFlip] = []
    tried: set[tuple[int, str, int]] = set()
    trajectory: list[dict[str, Any]] = []
    current_fitness = 0.0
    screened_out = 0

    if verbose:
        row_label = "all" if candidate_rows <= 0 else str(candidate_rows)
        print(
            f"dynamic-greedy-search task={task_name} target_probes={len(target_probes)} "
            f"control_probes={len(control_probes)} validation_probes={len(validation_probes)} "
            f"max_iters={max_iters} fitness_mode={fitness_mode} screen_probes={screen_names} "
            f"candidate_rows={row_label} max_flips={max_flips} candidates={len(candidates)} "
            f"control_penalty={control_penalty}",
            flush=True,
        )

    for step in range(max_iters):
        if len(accepted) >= config["search"]["max_flips"]:
            if verbose:
                print(f"[greedy {step}/{max_iters}] reached max_flips={max_flips}", flush=True)
            break

        pool = _sample_pool(candidates, candidate_weights, rng, tried, 1)
        if not pool:
            if verbose:
                print("[greedy] exhausted candidate pool", flush=True)
            break

        layer, proj, row, scale = pool[0]
        flip_row(handle.model, layer, proj, row)
        mx.eval(handle.model.parameters())

        screen_gaps = _measure_fast(handle.model, screen_packed, screen_names)
        screen_improved = any(screen_gaps[name] > target_baseline[name] for name in screen_names)
        screen_gain = sum(screen_gaps[name] - target_baseline[name] for name in screen_names) / max(1, len(screen_names))

        if not screen_improved:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            screened_out += 1
            trajectory.append(
                {
                    "step": step,
                    "layer": layer,
                    "proj": proj,
                    "row": row,
                    "scale_mean": scale,
                    "screen_gain": screen_gain,
                    "accepted": False,
                    "screened_out": True,
                }
            )
            if verbose and (step + 1) % 25 == 0:
                print(
                    f"[greedy {step+1}/{max_iters}] screened_out={screened_out} "
                    f"accepted={len(accepted)} fitness={current_fitness:+.6f}",
                    flush=True,
                )
            continue

        target_gaps = _measure_fast(handle.model, packed_target, target_names)
        control_gaps = _measure_fast(handle.model, packed_control, control_names)
        validation_gaps = _measure_fast(handle.model, packed_validation, validation_names)
        fitness = fitness_fn(target_gaps, control_gaps, target_baseline, control_baseline, config["search"]["control_penalty"])
        validation_gain = _mean_gain(validation_gaps, validation_baseline)
        accepted_candidate = fitness > current_fitness

        trajectory.append(
            {
                "step": step,
                "layer": layer,
                "proj": proj,
                "row": row,
                "scale_mean": scale,
                "screen_gain": screen_gain,
                "candidate_score": fitness,
                "validation_gain": validation_gain,
                "accepted": accepted_candidate,
                "screened_out": False,
            }
        )

        if accepted_candidate:
            accepted.append(PatchFlip(layer=layer, proj=proj, row=row))
            current_fitness = fitness
            _save_patch_checkpoint(
                output_path,
                f"{task_name}_dynamic_greedy_patch",
                f"Checkpoint for dynamic practical-eval Bankai greedy patch for {task_name}.",
                model_ref,
                accepted,
                {
                    "task": task_name,
                    "search_algorithm": f"dynamic_real_greedy_screened_{fitness_mode}",
                    "search_layers": active_layers,
                    "layer_profile": layer_profile,
                    "impact_weighted": impact_weighted,
                    "final_fitness": current_fitness,
                    "step": step,
                },
            )
            if verbose:
                print(
                    f"[greedy {step+1}/{max_iters}] ACCEPT L{layer}.{proj}[{row}] "
                    f"screen={screen_gain:+.6f} fitness={fitness:+.6f} "
                    f"validation_gain={validation_gain:+.6f} flips={len(accepted)}/{max_flips}",
                    flush=True,
                )
        else:
            flip_row(handle.model, layer, proj, row)
            mx.eval(handle.model.parameters())
            if verbose and (step + 1) % 25 == 0:
                print(
                    f"[greedy {step+1}/{max_iters}] reject L{layer}.{proj}[{row}] "
                    f"screen={screen_gain:+.6f} fitness={fitness:+.6f} "
                    f"accepted={len(accepted)}",
                    flush=True,
                )

    patch = BankaiPatch(
        name=f"{task_name}_dynamic_greedy_patch",
        description=f"Dynamic practical-eval Bankai greedy patch for {task_name}.",
        base_model=model_ref,
        flips=accepted,
        metadata={
            "benchmark": task_name,
            "task": task_name,
            "search_algorithm": f"dynamic_real_greedy_screened_{fitness_mode}",
            "search_layers": active_layers,
            "layer_profile": layer_profile,
            "impact_weighted": impact_weighted,
            "search_projs": config["search"]["search_projs"],
            "control_penalty": config["search"]["control_penalty"],
            "final_fitness": current_fitness,
            "candidate_rows": candidate_rows,
            "max_flips": max_flips,
            "max_iters": max_iters,
            "fitness_mode": fitness_mode,
            "target_probe_count": len(target_probes),
            "control_probe_count": len(control_probes),
            "validation_probe_count": len(validation_probes),
            "screen_probes": screen_names,
            "screened_out": screened_out,
            "model_summary": summary,
        },
    )
    save_patch(output_path, patch)
    save_run_manifest(
        output_path.with_name("dynamic_search.json"),
        {
            "task": task_name,
            "patch_path": str(output_path),
            "best_score": current_fitness,
            "trajectory": trajectory,
            "mode": "dynamic_greedy",
            "fitness_mode": fitness_mode,
            "max_iters": max_iters,
            "screened_out": screened_out,
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
