# Bankai Patch-Routing POC

This workspace scaffolds a benchmark-patched 1-bit Bonsai proof of concept inspired by [nikshepsvn/bankai](https://github.com/nikshepsvn/bankai):

- one shared base backend
- one reversible row-XOR patch per benchmark
- patch search + artifact logging
- per-benchmark and routed evaluation
- storage, latency, and reversibility reporting

The default backend is a deterministic mock backend so the full pipeline is runnable without Bonsai weights. It mirrors Bankai's `layer/proj/row` patch structure and row-level XOR application, but does not depend on `mlx`. This repo also now includes a real MLX/Bonsai path for live model inspection, live row-XOR mutation, real greedy search, real shortlist search, and base-vs-patched GSM8K spot evaluation.

## Current Status

What is real today:

- the PrismML MLX fork can load `prism-ml/Bonsai-8B-mlx-1bit`
- `inspect-model` confirms the expected packed `uint32` MLP row layout
- real row-XOR patch application and exact revert work on live Bonsai
- `real-search` and `real-shortlist-search` can produce non-empty patches on the live model

What is not proven yet:

- real benchmark gains across GSM8K, HumanEval+, IFEval, or BFCL
- that probe-level search improvements consistently translate into generation-level improvements
- that the current search objective is strong enough for broad arithmetic or reasoning gains

## Layout

```text
bankai_poc/
  configs/
  data/
  patches/
  probes/
  results/
  src/bankai_poc/
  tests/
```

## Install

```bash
cd /Users/andrewdimarogonas/Desktop/Huxli-parent/Synapse/bankai_poc
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Real MLX Setup

For true 1-bit Bonsai loading, stock `mlx` is not enough. The Bonsai model card requires the PrismML MLX fork with 1-bit kernel support.

Environment checks:

```bash
bankai-poc doctor
xcrun --find metal
xcrun --find metallib
```

Install path:

```bash
source .venv/bin/activate
pip install mlx-lm
pip install --force-reinstall 'mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism'
```

If the fork build fails with a missing Metal toolchain error, install it first:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Typical flow

```bash
bankai-poc download gsm8k
bankai-poc normalize gsm8k
bankai-poc probes gsm8k
bankai-poc search gsm8k
bankai-poc doctor
bankai-poc prep-all
bankai-poc inspect-model --model prism-ml/Bonsai-8B-mlx-1bit
bankai-poc real-search gsm8k --model prism-ml/Bonsai-8B-mlx-1bit
bankai-poc real-shortlist-search gsm8k --model prism-ml/Bonsai-8B-mlx-1bit --rounds 3 --pool 6 --topk 2 --target-probes 6 --control-probes 3
bankai-poc real-apply --model prism-ml/Bonsai-8B-mlx-1bit --patch gsm8k_real_patch.json --prompt "2 + 2 ="
bankai-poc real-gsm8k-compare --model prism-ml/Bonsai-8B-mlx-1bit --patch gsm8k_real_patch.json --limit 50 --max-tokens 80
bankai-poc matrix
bankai-poc route
bankai-poc plots
```

## Recommended Real Search Flow

For Apple Silicon machines, prefer shortlist search over naive greedy search. It screens a small candidate pool cheaply, then fully evaluates only the top candidates each round.

Smoke run:

```bash
bankai-poc real-shortlist-search gsm8k --model prism-ml/Bonsai-8B-mlx-1bit --rounds 3 --pool 6 --topk 2 --target-probes 6 --control-probes 3
```

Larger run:

```bash
bankai-poc real-shortlist-search gsm8k --model prism-ml/Bonsai-8B-mlx-1bit --rounds 8 --pool 12 --topk 3 --target-probes 8 --control-probes 4
```

Then validate the saved patch on held-out examples:

```bash
bankai-poc real-gsm8k-compare --model prism-ml/Bonsai-8B-mlx-1bit --patch gsm8k_real_patch.json --limit 50 --max-tokens 80
```

## Current Real GSM8K Example

The current shortlist-searched GSM8K patch is stored at [`patches/gsm8k_real_patch.json`](/Users/andrewdimarogonas/Desktop/Huxli-parent/Synapse/bankai_poc/patches/gsm8k_real_patch.json). It contains 3 accepted flips:

- `layer 3 / gate_proj / row 29`
- `layer 4 / gate_proj / row 43`
- `layer 3 / up_proj / row 30`

It is still tiny:

- `n_flips`: `3`
- `bits_flipped`: `12288`
- `size_bytes`: `36`

The corresponding search trace is in [`results/gsm8k_real_search.json`](/Users/andrewdimarogonas/Desktop/Huxli-parent/Synapse/bankai_poc/results/gsm8k_real_search.json), with final fitness `0.024739583333333336`.

Important caveat: this patch improved the probe objective used during search, but simple arithmetic prompt spot-checks did not yet show visible generation changes. Treat generation-level evaluation as the source of truth.

## Notes

- Benchmark source IDs are defined in `configs/*.yaml`.
- Downloaded raw artifacts live under `data/<benchmark>/raw`.
- Normalized examples are written to `data/<benchmark>/normalized.jsonl`.
- Generation eval items are written to `data/<benchmark>/eval_items.jsonl`.
- Probe sets are written to `probes/<benchmark>/probes.jsonl`.
- Patches are written to `patches/<benchmark>_patch.json`.
- Matrix and routing reports are written to `results/`.
- Patch JSON uses Bankai-style `bankai_row_xor_v1` metadata with `layer`, `proj`, and `row` flip entries.
- `doctor` writes a runtime capability report so you can see whether real `mlx`/Bankai execution is available before manual setup.
- `inspect-model` checks whether a loaded MLX model exposes the uint32 packed row structure needed for true Bankai row XOR patches.
- `real-search`, `real-shortlist-search`, `real-apply`, and `real-gsm8k-compare` are wired for Bonsai/MLX, but will trigger the first model download the first time you point them at a remote repo.
- `real-gsm8k-compare` writes summary and per-example outputs to `results/gsm8k_real_compare_summary.json` and `results/gsm8k_real_compare_details.json`.
- The mock matrix, routing, and plotting outputs are still useful for pipeline validation, but they are not real benchmark results.
