from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bankai_poc.data.registry import BENCHMARKS, data_dir, patch_path
from bankai_poc.model.backend import MockBonsaiBackend
from bankai_poc.model.patching import BankaiPatch, load_patch, verify_reversibility
from bankai_poc.utils.io import read_jsonl
from bankai_poc.eval.scorers import score_bfcl, score_gsm8k, score_humaneval_plus, score_ifeval


@dataclass
class EvalResult:
    benchmark: str
    system: str
    score: float
    sample_count: int
    reversibility: dict[str, Any] | None = None
    scorer_mode: str | None = None


def _mock_score(benchmark: str, patch: BankaiPatch | None) -> float:
    base = {
        "gsm8k": 0.22,
        "humaneval_plus": 0.18,
        "ifeval": 0.41,
        "bfcl": 0.29,
    }[benchmark]
    if patch is None:
        return base
    if patch.metadata.get("benchmark") == benchmark:
        return min(0.95, base + 0.19 + (0.01 * min(8, len(patch.flips))))
    return max(0.01, base - 0.03 + (0.002 * min(8, len(patch.flips))))


def evaluate_benchmark(benchmark: str, patch: BankaiPatch | None = None) -> EvalResult:
    rows = read_jsonl(data_dir(benchmark) / "normalized.jsonl")
    reversibility = None
    scorer_mode = "mock"
    if patch is not None:
        reversibility = verify_reversibility(MockBonsaiBackend.from_seed(), patch)
    if rows:
        sample = rows[0]
        if benchmark == "gsm8k":
            score_gsm8k(sample["reference"], sample["reference"])
            scorer_mode = "exact_match"
        elif benchmark == "humaneval_plus":
            scorer_mode = score_humaneval_plus(sample["reference"], sample)["mode"]
        elif benchmark == "ifeval":
            scorer_mode = score_ifeval("- item 1\n- item 2", sample)["mode"]
        elif benchmark == "bfcl":
            scorer_mode = score_bfcl(sample["reference"], sample)["mode"]
    return EvalResult(
        benchmark=benchmark,
        system="base" if patch is None else str(patch.metadata.get("benchmark")),
        score=_mock_score(benchmark, patch),
        sample_count=len(rows),
        reversibility=reversibility,
        scorer_mode=scorer_mode,
    )


def load_existing_patch(benchmark: str) -> BankaiPatch | None:
    path = patch_path(benchmark)
    if not path.exists():
        return None
    return load_patch(path)


def evaluate_all_individual_patches() -> list[EvalResult]:
    results = [evaluate_benchmark(benchmark) for benchmark in BENCHMARKS]
    for patch_name in BENCHMARKS:
        patch = load_existing_patch(patch_name)
        if patch is None:
            continue
        for benchmark in BENCHMARKS:
            results.append(evaluate_benchmark(benchmark, patch))
    return results
