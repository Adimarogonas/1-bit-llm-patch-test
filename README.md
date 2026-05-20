# Bankai Eval Studio

This workspace builds on Nikshep Saravanan's [Bankai](https://github.com/nikshepsvn/bankai) work on ultra-sparse XOR patching for 1-bit LLMs. The core Bankai idea is the adaptation substrate here; the focus of this repo is the evaluation pipeline around it:

- practical CSV eval ingestion
- programmatic and judge-based scoring
- adaptive base-vs-patched evaluation
- dynamic probes built from actual base-model mistakes
- fixed/regressed/still-wrong comparison views
- reproducible artifacts under `results/eval_runs`

The default backend is a deterministic mock backend so the pipeline is runnable without Bonsai weights. The real path uses `prism-ml/Bonsai-8B-mlx-1bit` through the PrismML MLX fork and applies Bankai-style `layer/proj/row` XOR patches before inference.

## Current Status

What is real today:

- the PrismML MLX fork can load `prism-ml/Bonsai-8B-mlx-1bit`
- `inspect-model` confirms the expected packed `uint32` MLP row layout
- real row-XOR patch application and exact revert work on live Bonsai
- CSV evals can run through base, patched, reference, and terminal-agent models
- adaptive evals build probes from base failures and controls from base passes
- scoring supports exact/normalized labels, regex, `numeric_final`, JSON field/path checks, structured JSON matching, and command-based judges
- stored run artifacts support side-by-side fixed/regressed/still-wrong review

What is not proven yet:

- real benchmark gains across GSM8K, HumanEval+, IFEval, or BFCL
- that probe-level improvements consistently translate into generation-level improvements
- that the current dynamic-probe objectives generalize beyond small held-out CSVs

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

## Eval Studio

This repo includes an Electron UI that treats the Python package as a sidecar. The default research direction is now practical CSV evals: sentiment analysis, classification, routing, summarization, extraction, and similar product-facing tasks. The app can normalize CSV datasets, run the three-way comparison, stream CLI logs, configure an optional command-based LLM judge, and inspect saved summaries/details under `results/eval_runs`.

```bash
cd /Users/andrewdimarogonas/Desktop/Personal\ Projects/bankai_poc/electron
yarn install
yarn start
```

The app launches the sidecar with:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-dataset ...
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-run ...
```

The default comparison slots are:

- patched Bankai MLX: `prism-ml/Bonsai-8B-mlx-1bit` plus a Bankai patch JSON
- reference Bonsai MLX: configurable; default is `prism-ml/Bonsai-8B-unpacked` until you point it at the 8-bit MLX repo you want to use
- terminal agent: configurable command; the rendered prompt is sent on stdin unless the command contains `{prompt}`

CSV evals need at minimum:

- `prompt`
- `expected` or `grader`

Use `expected` for programmatic label/answer matching. Use `grader` for natural-language judge instructions, then configure a judge command. If both `expected` and `grader` are present, `expected` wins by default. Optional useful columns are `id`, `system`, `assessment`, `segment`, `category`, and `difficulty`.

Programmatic assessment modes:

- `exact_normalized`: default for `expected`; lowercase, trim, and collapse whitespace
- `classification` or `routing`: aliases for `exact_normalized`
- `exact`: trimmed raw equality
- `contains`: normalized prediction contains normalized expected
- `regex`: expected is a case-insensitive regex; generated answers are normalized for common markdown/newline/trailing-period formatting
- `numeric_final`: extract the final numeric answer, strip `$`, commas, markdown, and terminal punctuation, then compare numerically; use `numeric_final:0.01` for an explicit tolerance
- `json_field:<field>`: parse prediction as JSON and compare a field or dotted path
- `json_match`: compare expected and predicted JSON objects, with optional `strict_fields` and `soft_fields`
- `llm_judge`: explicitly use the configured judge command

CLI-only equivalent:

```bash
bankai-poc eval-dataset \
  --csv-source data/practical_evals/example_sentiment.csv \
  --task-name sentiment_analysis \
  --limit 20 \
  --output data/practical_evals/sentiment_analysis.jsonl

bankai-poc eval-run \
  --dataset data/practical_evals/sentiment_analysis.jsonl \
  --output-dir results/eval_runs/run_latest \
  --bankai-model prism-ml/Bonsai-8B-mlx-1bit \
  --bankai-patch patches/gsm8k_real_patch.json \
  --reference-model prism-ml/Bonsai-8B-unpacked \
  --agent-command "codex exec --model gpt-5.4-mini"
```

LLM-judged rows pass this JSON payload to the judge command on stdin:

```json
{"prompt": "...", "expected": "", "prediction": "...", "grader": "Pass if ...", "metadata": {}}
```

Preferred judge output:

```json
{"passed": true, "score": 1.0, "reason": "Short explanation."}
```

Reusable command notes live in `.agents/commands/`.

Adaptive eval mode:

```bash
bankai-poc eval-adapt \
  --dataset data/practical_evals/example_routing.csv \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --output-dir results/eval_runs \
  --search-mode greedy \
  --max-iters 600 \
  --fitness-mode mean \
  --accept-per-round 2
```

This runs the base model first, builds dynamic probes from failed rows, uses passing rows as controls, searches a Bankai XOR patch, and re-runs the same CSV with the patch applied. The Electron app exposes this as the main workflow with native file pickers and stored run inspection.

## Probe Improvements

The important implementation work is in the probe builder, not only in the row search loop.

- For structured JSON/tool-call outputs, probes target the first divergent boundary: tool name, argument key, then argument value.
- For non-JSON label tasks, probes use the expected label and the model's actual wrong output.
- For regex rows, probes use representative answer text instead of dumping raw regex syntax into `correct_completion`, `wrong_completion`, or probe metadata.
- For `numeric_final` rows, probes use canonical numeric completions such as `32.2` or `180.5`.
- Multi-token continuations are scored with teacher-forced mean log probability, so `send_email` vs `escalate_ticket` or an email address is not collapsed to one brittle token.
- Controls are built from rows the base model already passes, so a patch is penalized when it damages known-good behavior.

Patch search is still present (`shortlist`, `greedy`, and related real-search commands), but generation-level eval is the promotion gate. Probe fitness alone is not treated as success.

## Current Results Snapshot

The strongest current result is the adaptive tool-call pipeline on `tool_call_selection.csv`:

| Dataset | Base | Patched | Notes |
|---|---:|---:|---|
| `tool_call_selection.csv` | 62/70, 88.6% | 66/70, 94.3% | 16 row flips, 0 regressions |
| `multi_step_reasoning.csv` | 27/30, 90.0% | 28/30, 93.3% | rescored with `numeric_final`; one format-insensitive numeric fix |

The arithmetic result is intentionally modest: it mainly validates that semantic numeric scoring prevents formatting artifacts from being mistaken for model failures. Rows with real numeric mistakes, such as `1805` vs `180.5`, still fail.

## Attribution

This project heavily builds on:

```bibtex
@software{Saravanan_Bankai_Ultra-Sparse_Adaptation_2026,
author = {Saravanan, Nikshep},
license = {Apache-2.0},
month = apr,
title = {{Bankai: Ultra-Sparse Adaptation of 1-Bit LLMs via XOR Patches}},
url = {https://github.com/nikshepsvn/bankai},
year = {2026}
}
```

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
