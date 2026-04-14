from __future__ import annotations

import argparse
from pathlib import Path

from bankai_poc.data.download import download_benchmark
from bankai_poc.data.normalize import normalize_benchmark
from bankai_poc.data.registry import BENCHMARKS, root_dir
from bankai_poc.eval.real_gsm8k import run_real_gsm8k_compare
from bankai_poc.eval.matrix import build_cross_benchmark_matrix
from bankai_poc.model.runtime import inspect_runtime
from bankai_poc.model.real_mlx import generate_text, load_real_model, model_patchable_summary, apply_real_patch
from bankai_poc.model.patching import load_patch
from bankai_poc.plotting import generate_figures
from bankai_poc.probes.generators import generate_probes
from bankai_poc.routing.router import run_routed_evaluation
from bankai_poc.search.real_runner import (
    run_real_anneal_shortlist_search,
    run_real_search,
    run_real_shortlist_search,
    run_real_two_pass_search,
)
from bankai_poc.search.runner import run_search
from bankai_poc.utils.io import dump_json


def _benchmark_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("benchmark", choices=BENCHMARKS)


def _parse_layers(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="bankai-poc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    _benchmark_arg(download_parser)

    normalize_parser = subparsers.add_parser("normalize")
    _benchmark_arg(normalize_parser)

    probes_parser = subparsers.add_parser("probes")
    _benchmark_arg(probes_parser)

    search_parser = subparsers.add_parser("search")
    _benchmark_arg(search_parser)

    subparsers.add_parser("doctor")
    subparsers.add_parser("prep-all")

    inspect_parser = subparsers.add_parser("inspect-model")
    inspect_parser.add_argument("--model", required=True)

    real_search_parser = subparsers.add_parser("real-search")
    _benchmark_arg(real_search_parser)
    real_search_parser.add_argument("--model", required=True)
    real_search_parser.add_argument("--iters", type=int, default=None)
    real_search_parser.add_argument("--target-probes", type=int, default=8)
    real_search_parser.add_argument("--control-probes", type=int, default=4)
    real_search_parser.add_argument("--layers", default=None, help="Comma-separated layer list override, e.g. 1,4,8,12,16")
    real_search_parser.add_argument("--layer-profile", choices=["stable", "balanced", "aggressive"], default=None)
    real_search_parser.add_argument("--impact-weighted", action="store_true", help="Downweight high-impact layers when sampling candidates.")

    shortlist_parser = subparsers.add_parser("real-shortlist-search")
    _benchmark_arg(shortlist_parser)
    shortlist_parser.add_argument("--model", required=True)
    shortlist_parser.add_argument("--rounds", type=int, default=4)
    shortlist_parser.add_argument("--pool", type=int, default=8)
    shortlist_parser.add_argument("--topk", type=int, default=2)
    shortlist_parser.add_argument("--target-probes", type=int, default=6)
    shortlist_parser.add_argument("--control-probes", type=int, default=3)
    shortlist_parser.add_argument("--layers", default=None, help="Comma-separated layer list override, e.g. 1,4,8,12,16")
    shortlist_parser.add_argument("--layer-profile", choices=["stable", "balanced", "aggressive"], default=None)
    shortlist_parser.add_argument("--impact-weighted", action="store_true", help="Downweight high-impact layers when sampling candidates.")

    two_pass_parser = subparsers.add_parser("real-two-pass-search")
    _benchmark_arg(two_pass_parser)
    two_pass_parser.add_argument("--model", required=True)
    two_pass_parser.add_argument("--rounds", type=int, default=4)
    two_pass_parser.add_argument("--pool", type=int, default=32)
    two_pass_parser.add_argument("--mid-topk", type=int, default=8)
    two_pass_parser.add_argument("--topk", type=int, default=2)
    two_pass_parser.add_argument("--target-probes", type=int, default=6)
    two_pass_parser.add_argument("--control-probes", type=int, default=3)
    two_pass_parser.add_argument("--pass2-target-probes", type=int, default=4)
    two_pass_parser.add_argument("--pass2-control-probes", type=int, default=2)
    two_pass_parser.add_argument("--consistency-penalty", type=float, default=0.5)
    two_pass_parser.add_argument("--layers", default=None, help="Comma-separated layer list override, e.g. 1,4,8,12,16")
    two_pass_parser.add_argument("--layer-profile", choices=["stable", "balanced", "aggressive"], default=None)
    two_pass_parser.add_argument("--impact-weighted", action="store_true", help="Downweight high-impact layers when sampling candidates.")

    anneal_parser = subparsers.add_parser("real-anneal-shortlist-search")
    _benchmark_arg(anneal_parser)
    anneal_parser.add_argument("--model", required=True)
    anneal_parser.add_argument("--steps", type=int, default=24)
    anneal_parser.add_argument("--pool", type=int, default=16)
    anneal_parser.add_argument("--topk", type=int, default=4)
    anneal_parser.add_argument("--target-probes", type=int, default=6)
    anneal_parser.add_argument("--control-probes", type=int, default=3)
    anneal_parser.add_argument("--start-temp", type=float, default=0.02)
    anneal_parser.add_argument("--end-temp", type=float, default=0.001)
    anneal_parser.add_argument("--remove-prob", type=float, default=0.15)
    anneal_parser.add_argument("--swap-prob", type=float, default=0.45)
    anneal_parser.add_argument("--layers", default=None, help="Comma-separated layer list override, e.g. 1,4,8,12,16")
    anneal_parser.add_argument("--layer-profile", choices=["stable", "balanced", "aggressive"], default=None)
    anneal_parser.add_argument("--impact-weighted", action="store_true", help="Downweight high-impact layers when sampling candidates.")
    anneal_parser.add_argument("--output", default=None, help="Optional patch JSON output path.")

    real_apply_parser = subparsers.add_parser("real-apply")
    real_apply_parser.add_argument("--model", required=True)
    real_apply_parser.add_argument("--patch", required=True)
    real_apply_parser.add_argument("--prompt", required=True)
    real_apply_parser.add_argument("--max-tokens", type=int, default=80)

    real_compare_parser = subparsers.add_parser("real-gsm8k-compare")
    real_compare_parser.add_argument("--model", required=True)
    real_compare_parser.add_argument("--patch", default="gsm8k_real_patch.json")
    real_compare_parser.add_argument("--limit", type=int, default=50)
    real_compare_parser.add_argument("--max-tokens", type=int, default=160)

    subparsers.add_parser("matrix")
    subparsers.add_parser("route")
    subparsers.add_parser("plots")

    args = parser.parse_args()
    results_dir = root_dir() / "results"

    if args.command == "download":
        print(download_benchmark(args.benchmark))
        return
    if args.command == "normalize":
        normalize_benchmark(args.benchmark)
        return
    if args.command == "probes":
        generate_probes(args.benchmark)
        return
    if args.command == "search":
        result = run_search(args.benchmark)
        print(result.patch.name)
        return
    if args.command == "doctor":
        status = inspect_runtime()
        output_path = root_dir() / "results" / "doctor.json"
        dump_json(output_path, status.to_dict())
        print(output_path)
        return
    if args.command == "prep-all":
        for benchmark in BENCHMARKS:
            try:
                download_benchmark(benchmark)
                normalize_benchmark(benchmark)
                generate_probes(benchmark)
                print(f"prepared {benchmark}")
            except Exception as exc:
                print(f"failed {benchmark}: {type(exc).__name__}: {exc}")
        return
    if args.command == "inspect-model":
        handle = load_real_model(args.model)
        summary = model_patchable_summary(handle.model)
        output_path = root_dir() / "results" / "model_inspect.json"
        dump_json(output_path, {"model": args.model, **summary})
        print(output_path)
        return
    if args.command == "real-search":
        from bankai_poc.search.real_runner import _run_real_search

        result = _run_real_search(
            args.benchmark,
            args.model,
            max_iters=args.iters,
            max_target_probes=args.target_probes,
            max_control_probes=args.control_probes,
            search_layers=_parse_layers(args.layers),
            layer_profile=args.layer_profile,
            impact_weighted=args.impact_weighted,
        )
        print(result.patch.name)
        return
    if args.command == "real-shortlist-search":
        result = run_real_shortlist_search(
            args.benchmark,
            args.model,
            rounds=args.rounds,
            shortlist_pool=args.pool,
            shortlist_topk=args.topk,
            max_target_probes=args.target_probes,
            max_control_probes=args.control_probes,
            search_layers=_parse_layers(args.layers),
            layer_profile=args.layer_profile,
            impact_weighted=args.impact_weighted,
        )
        print(result.patch.name)
        return
    if args.command == "real-two-pass-search":
        result = run_real_two_pass_search(
            args.benchmark,
            args.model,
            rounds=args.rounds,
            shortlist_pool=args.pool,
            mid_topk=args.mid_topk,
            shortlist_topk=args.topk,
            max_target_probes=args.target_probes,
            max_control_probes=args.control_probes,
            pass2_target_probes=args.pass2_target_probes,
            pass2_control_probes=args.pass2_control_probes,
            search_layers=_parse_layers(args.layers),
            layer_profile=args.layer_profile,
            impact_weighted=args.impact_weighted,
            consistency_penalty=args.consistency_penalty,
        )
        print(result.patch.name)
        return
    if args.command == "real-anneal-shortlist-search":
        result = run_real_anneal_shortlist_search(
            args.benchmark,
            args.model,
            output_path=Path(args.output) if args.output else None,
            steps=args.steps,
            shortlist_pool=args.pool,
            shortlist_topk=args.topk,
            max_target_probes=args.target_probes,
            max_control_probes=args.control_probes,
            search_layers=_parse_layers(args.layers),
            layer_profile=args.layer_profile,
            impact_weighted=args.impact_weighted,
            start_temp=args.start_temp,
            end_temp=args.end_temp,
            remove_prob=args.remove_prob,
            swap_prob=args.swap_prob,
        )
        print(result.patch.name)
        return
    if args.command == "real-apply":
        handle = load_real_model(args.model)
        patch_ref = Path(args.patch)
        patch_path = patch_ref if patch_ref.is_absolute() or patch_ref.exists() else (root_dir() / "patches" / patch_ref)
        patch = load_patch(patch_path)
        before = generate_text(handle, args.prompt, max_tokens=args.max_tokens)
        apply_real_patch(handle.model, patch)
        after = generate_text(handle, args.prompt, max_tokens=args.max_tokens)
        print("[before]")
        print(before)
        print("[after]")
        print(after)
        return
    if args.command == "real-gsm8k-compare":
        summary_path, details_path = run_real_gsm8k_compare(
            model_ref=args.model,
            patch_name=args.patch,
            limit=args.limit,
            max_tokens=args.max_tokens,
        )
        print(summary_path)
        print(details_path)
        return
    if args.command == "matrix":
        csv_path, json_path = build_cross_benchmark_matrix(results_dir)
        print(csv_path)
        print(json_path)
        return
    if args.command == "route":
        print(run_routed_evaluation(results_dir))
        return
    if args.command == "plots":
        for path in generate_figures(results_dir):
            print(path)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
