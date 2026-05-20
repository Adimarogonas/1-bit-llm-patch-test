# Adaptive Eval and Probe Construction for Bankai-Style 1-Bit LLM Patches

## A Bankai/Bonsai Proof-of-Concept Report Focused on Evaluation

Date: May 20, 2026

## Abstract

This project investigates how to evaluate and guide request-time specialization for a true 1-bit language model using tiny reversible XOR patches. Instead of loading multiple expert models or attaching larger adapters, the system keeps one shared Bonsai 8B base model and applies a small Bankai-style row-XOR patch before inference. The patch can then be reverted exactly or swapped for another patch on the next request.

The patching substrate builds heavily on Nikshep Saravanan's Bankai work on ultra-sparse adaptation of 1-bit LLMs via XOR patches. The current proof of concept validates compatible patch mechanics on `prism-ml/Bonsai-8B-mlx-1bit`, but the main technical progress in this repo is the evaluation and probe-construction pipeline around those patches. Early GSM8K probes could move logit objectives without reliably moving generation accuracy. The newer adaptive pipeline builds probes from the base model's actual mistakes, targets the first wrong/correct decision boundary, supports multi-token continuations, and adds controls from examples the base model already gets right.

On a 70-row `tool_call_selection.csv` evaluation, a 16-flip patch improved accuracy from 62/70 to 66/70, or 88.6% to 94.3%, with zero observed regressions. The result is still narrow and dataset-specific, but it is the first clear sign in this project that boundary-aware dynamic probes can convert XOR row flips into generation-level task improvement.

## 1. Motivation

Large model specialization is usually achieved by storing additional model weights, adapters, LoRA deltas, or separate expert models. That works, but it increases storage, deployment complexity, and sometimes inference overhead.

Bankai's hypothesis is different:

> If the base model is truly 1-bit, then useful behavioral changes may be expressible as small sets of reversible bit flips.

For a 1-bit model, an XOR patch can flip selected packed weight rows. Applying the same XOR again reverts the model exactly. This creates a simple request-time specialization mechanism:

1. Load one shared base model.
2. Select a patch for the task.
3. XOR selected rows in-place.
4. Run inference.
5. XOR the same rows again to restore the base model.

This is not a true mixture-of-experts architecture. It is better described as patch-routed specialization or MoE-like specialization behavior with near-zero parameter overhead.

The contribution of this repo is not a new patch format. It is the surrounding machinery needed to tell whether a patch helped for the right reason:

- CSV eval ingestion with reproducible summary and detail artifacts.
- Semantic scorers for labels, regexes, final numeric answers, JSON fields, and structured JSON calls.
- Adaptive probes built from base-model failures instead of static benchmark labels.
- Control probes from base-passing examples to catch regressions.
- Side-by-side review of fixed, regressed, still-wrong, and still-right cases.

## 2. System Overview

The proof of concept currently contains:

- A reproducible Python project with data, probes, patches, configs, results, and source modules.
- Dataset acquisition and normalization for GSM8K, HumanEval+, IFEval, and BFCL.
- Probe generation for benchmark-specific supervision.
- Real MLX/Bonsai model loading through the PrismML MLX fork.
- Live row-XOR patch application and reversion on packed 1-bit MLP weights.
- Search runners for greedy, shortlist, two-pass, and annealing variants. These are treated as interchangeable patch-search backends.
- Generation-level evaluation with base-vs-patched comparison.
- Adaptive CSV evaluation that runs the base model, extracts failures, builds dynamic probes, searches a patch, reruns patched evaluation, and stores both example-level details and aggregate summaries.
- A results UI that reports fixed cases, regressions, still-wrong cases, changed outputs, and side-by-side base/patched generations.
- Patch checkpointing after every accepted flip so interrupted or failed searches can be recovered.
- Team runbook commands for distributing heavier patch searches across M3/M4 Apple Silicon machines.

The initial benchmark focus was GSM8K because math-answer correctness is easy to evaluate and explain. The more recent practical focus is practical CSV evals, especially tool-call selection and multi-step arithmetic. These expose the two failure modes that motivated the current pipeline: structured outputs often fail at an early schema boundary, while math outputs often look wrong under strict regex matching even when the numeric answer is correct.

## 3. Model and Patch Format

The real model used so far is:

```text
prism-ml/Bonsai-8B-mlx-1bit
```

Inspection showed Bankai-compatible packed MLP rows:

```text
weight_dtype: mlx.core.uint32
weight_shape: [12288, 128]
scales_shape: [12288, 32]
```

Each row flip changes:

```text
128 uint32 values = 4096 packed bits
```

Patch JSON stores each flip as:

```json
{
  "layer": 20,
  "proj": "gate_proj",
  "row": 25
}
```

The estimated compact storage cost is:

```text
12 bytes per row flip, excluding JSON metadata
```

A 3-flip patch is therefore approximately:

```text
36 bytes metadata-excluded
12288 flipped bits
```

## 4. Probe Construction

Bankai patch search performs better with probe-style objectives than raw benchmark examples, but the central lesson is that the probe has to match the real failure. A probe that asks "prefer this answer token over that answer token" can improve while the generated answer remains wrong, because the model may be making its first mistake much earlier.

The project now uses three probe families.

### 4.1 Static Final-Answer Probes

For GSM8K, the project moved from simple answer-token probes to stronger final-answer probes:

- Normalize GSM8K examples into prompt, rationale, full answer, and parsed answer value.
- Build prompts that include teacher-forced reasoning and end at `Final answer: `.
- Use the correct numeric answer as the positive token.
- Generate plausible wrong answers such as `+1`, `-1`, double, half, and nearby arithmetic distractors.
- Partition probes into `search`, `validation`, and `control`.

Current GSM8K probe partition sizes:

```text
search: 4186
validation: 600
control: 475
```

The key improvement was separating target, validation, and control probes so search fitness does not only optimize the same examples used for sanity checking.

These probes were useful for validating the pipeline, but they did not reliably improve generation-level GSM8K accuracy.

### 4.2 Dynamic Boundary Probes

The newer adaptive pipeline builds probes from the base model's own outputs:

1. Run the base model on the evaluation set.
2. Identify rows that failed and rows that passed.
3. Parse the expected answer and the base prediction.
4. Build search probes from failures.
5. Build validation and control probes from held-out failures and successful base cases.
6. Search for a patch.
7. Rerun evaluation with the patch and compare base vs patched generations.

For JSON-like tool calls, the probe builder targets the earliest decision boundary that explains the mistake:

| Failure type | Probe prompt ends at | Correct continuation | Wrong continuation |
|---|---|---|---|
| Wrong tool | `{"name": "` | correct tool name | base model's wrong tool name |
| Wrong argument key | `{"name": "<tool>", "arguments": {"` | correct key, such as `to` | wrong key, such as `team` |
| Wrong argument value | field prefix after the correct key | expected value | base model's wrong value |

This matters for tool-use tasks. If the base model emits:

```json
{"name": "escalate_ticket", "arguments": {"team": "billing-team@vendor.com"}}
```

but the expected behavior is:

```json
{"name": "send_email", "arguments": {"to": "billing-team@vendor.com"}}
```

then optimizing only for the email value is too late. The important mistakes are first the tool boundary, `send_email` vs `escalate_ticket`, and then the argument-key boundary, `to` vs `team`.

### 4.3 Semantic Regex and Numeric Probes

The adaptive pipeline originally treated regex expectations as literal labels. That was a mistake. A row whose expected value was:

```text
final answer:\s*90(?:\.0+)?(?![\d.])
```

would produce a probe whose positive completion was the raw regex text, not a natural answer such as:

```text
final answer: 90
```

The current probe builder sanitizes regex expectations before probe construction. Search, validation, and control probes now use representative answer text, and their metadata no longer dumps raw regex syntax. This matters because the model should be optimized toward the semantic answer, not toward a grading implementation detail.

For math rows, the preferred assessment mode is now `numeric_final`. It extracts the final answer number, strips currency signs, commas, markdown emphasis, and terminal punctuation, then compares numerically with a configurable tolerance. Dynamic probes for `numeric_final` rows use canonical numeric completions such as `32.2`, `180.5`, or `6.48`.

This distinction avoids two bad outcomes:

- Correct answers such as `Final answer: 90.` or `**450**` are no longer marked wrong because of formatting.
- Truly wrong answers such as `1805` instead of `180.5` still fail.

### 4.4 Multi-Token Probe Scoring

The scorer now supports multi-token continuations. For dynamic probes, it tokenizes `prompt + correct_completion` and `prompt + wrong_completion`, finds the shared token prefix, and scores the divergent suffixes with teacher-forced mean log probability. This avoids reducing a decision such as `send_email` vs `escalate_ticket` or a multi-token email address to one brittle token.

The same mechanism remains compatible with non-JSON tasks. If the output is not JSON-like, the dynamic builder falls back to a label-style prompt with the expected answer as the correct continuation and the model's actual wrong answer as the negative continuation.

### 4.5 Regression Controls

Dynamic probes also include controls from examples the base model already passed. For tool-call selection, controls preserve:

- Tool names, such as `search_orders` vs `cancel_subscription`.
- Tool aliases that are easy to confuse, such as `send_email` vs `notify`.
- Argument keys and values already produced correctly.

This is why the results UI now separates fixed cases from regressions. A patch that fixes the target class but turns correct `search_orders` calls into repeated `cancel_subscription` calls is not acceptable, even if its raw probe fitness improves.

## 5. Patch Search as an Adaptation Backend

Patch search is necessary, but it is no longer the main research object of this repo. The search runners exist to answer a simpler question: given a probe set that reflects actual model mistakes, can small Bankai-style row flips move generation-level behavior without damaging controls?

Several search strategies are implemented, and the UI/CLI can choose between them. The important invariant is that every candidate patch is judged against probes derived from the eval pipeline and then promoted only by generation-level base-vs-patched comparison.

Implemented search backends include greedy hill climbing, shortlist screening, two-pass shortlist screening, and annealing-style patch-state moves. The exact search strategy is less important than the contract it obeys:

1. Score candidate flips against search probes built from base failures.
2. Penalize damage to control probes built from base-passing examples.
3. Checkpoint accepted flips so interrupted runs are recoverable.
4. Rerun generation and inspect fixed/regressed/still-wrong examples before treating a patch as useful.

This framing deliberately makes search a backend. The research loop starts with eval rows, builds better probes, searches a patch only as a consequence, and then returns to generation-level eval.

## 6. Layer-Impact Findings

A layer-level probe sweep measured average absolute logit-gap changes across 8 probes:

| Layer range | Avg abs. delta gap | Interpretation |
|---|---:|---|
| 0-4 | 3.2-7.2 | High impact, syntax/embedding-sensitive |
| 5-16 | 0.7-3.0 | Moderate, decreasing impact |
| 17-21 | 0.7-1.6 | Lowest impact, most redundant |
| 22-33 | 1.6-3.4 | Moderate, increasing toward output |
| 34 | 9.0 | Highest impact |
| 35 | 3.2 | High, less than 34 |

Based on this, the search code now supports layer profiles:

```text
stable:     [0, 1, 2, 3, 4, 34, 35]
balanced:   [0, 1, 2, 3, 4, 22, 24, 28, 32, 34, 35]
aggressive: [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 34, 35]
```

The current default is the high-impact set:

```text
[0, 1, 2, 3, 4, 34, 35]
```

The app also exposes a custom layer selector so a run can focus on a hand-picked set without changing code. This became important because the best layer set appears task-dependent: GSM8K safety runs tolerated lower-impact layers, while tool-call selection responded strongly to early layers and final layers.

## 7. Experiments and Results So Far

### 7.1 Corrected GSM8K Generation Harness

Early GSM8K scores looked artificially poor. After switching to the Qwen3 chat template and allowing `max_tokens=400`, the base model performed much better:

```text
Base Bonsai 8B: 42/50 = 84%
```

This matters because earlier low GSM8K scores were likely harness artifacts rather than model capability limits.

### 7.2 Earlier GSM8K Patch Experiments

Earlier GSM8K patch experiments were useful primarily because they showed what not to trust. A 50-example comparison with the corrected harness showed:

| System | Correct | Accuracy | Delta vs base | Changed generations |
|---|---:|---:|---:|---:|
| Base | 42/50 | 84% | 0% | n/a |
| Curated shortlist patch | 42/50 | 84% | 0% | 26 |
| All-layer/wide patch | 38/50 | 76% | -8% | 30 |

Interpretation:

- The curated patch changed outputs but did not change accuracy.
- The all-layer/wide patch caused a generation-level regression.
- This supports the need for safer layer selection and generation-level validation.

Several probe-positive GSM8K patches then failed to improve generation-level accuracy. A representative annealed patch changed generations but left accuracy unchanged:

```text
Base:    42/50 = 84%
Patched: 42/50 = 84%
Delta:   0%
Changed generations: 3
Correctness changes: 0
```

Interpretation:

- Probe gains alone were not enough.
- Generation-level eval had to remain the promotion gate.
- Search budget was less important than whether the probe represented the actual failure.

### 7.3 Tool-Call Dynamic Probe Search

The most important current result comes from the adaptive tool-call pipeline on `tool_call_selection.csv`.

| Metric | Base | Patched |
|---|---:|---:|
| Correct | 62/70 | 66/70 |
| Accuracy | 88.6% | 94.3% |
| Delta | n/a | +4 rows, +5.7 points |
| Regressions | n/a | 0 |
| Changed outputs | n/a | 26 |
| Patch size | n/a | 16 row flips |
| Probe score | n/a | 1.2529 |

The fixed examples share a pattern. The base model often chose `escalate_ticket` and placed an email address under `arguments.team`; the patched model chose `send_email` and placed the same contact under `arguments.to`.

Representative fixed cases:

| Row | Base behavior | Patched behavior |
|---:|---|---|
| 35 | `escalate_ticket`, `team: billing-team@vendor.com` | `send_email`, `to: billing-team@vendor.com` |
| 52 | `escalate_ticket`, `team: support-eng@vendor.io` | `send_email`, `to: support-eng@vendor.io` |
| 56 | `escalate_ticket`, `team: product-team@startup.io` | `send_email`, `to: product-team@startup.io` |
| 62 | `escalate_ticket`, `team: engineering-leads@startup.io` | `send_email`, `to: engineering-leads@startup.io` |

This is exactly the kind of error the boundary probes were designed to target. The patch is not merely increasing the probability of a final email string; it is moving earlier structural choices: tool name first, then argument key.

The same run reported zero regressions. That is important because earlier experimental patches could fix one class of examples while breaking already-correct tool calls, such as replacing `search_orders` with repeated `cancel_subscription` calls or confusing `send_email` with `notify`.

### 7.4 Multi-Step Reasoning Eval Pipeline Check

The `multi_step_reasoning.csv` eval exposed a grading issue rather than a patch-search breakthrough. Under strict regex scoring, correct numeric answers with trailing periods, markdown emphasis, or answer-on-next-line formatting were counted as failures. After adding `numeric_final`, the same stored generations can be scored semantically.

For the `adaptive_20260520_120534` run, rescoring with `numeric_final` gives:

| Metric | Base | Patched |
|---|---:|---:|
| Correct | 27/30 | 28/30 |
| Accuracy | 90.0% | 93.3% |
| Fixed | n/a | 1 |
| Regressions | n/a | 0 |

The fixed row is a formatting-insensitive numeric success: the patched model produced `\boxed{32.2}`, which matches the expected `32.20`. The still-wrong rows remain real numeric mistakes: `1805` instead of `180.5`, and `4.32` instead of `6.48`.

This result is important for methodology. It shows why eval scoring must distinguish presentation errors from semantic errors before those examples are converted into probes. Otherwise the search loop optimizes against artifacts of the grader.

## 8. Key Technical Lessons

### Probe Formation Is Now the Core Method

The main progress came from changing what is optimized, not from making the patch operation more complex. Dynamic probes built from actual base failures are much more useful than generic probes because they target the decision the model really got wrong.

For structured outputs, the correct probe is usually not at the end of the answer. It is at the first divergent structural choice: function name, argument key, or argument value.

For math-like outputs, the correct probe should be the canonical numeric answer, not the raw regex or the surrounding markdown. The scorer and probe builder now share this principle through `numeric_final`.

### Multi-Token Scoring Is Necessary

Tool names, field names, emails, dates, and many labels are not reliably represented by one token. Scoring the divergent multi-token suffix makes the objective closer to the generated behavior while still staying much cheaper than full generation inside the search loop.

### Probe Fitness Does Not Guarantee Generation Gains

Several GSM8K patches improved the probe objective without improving generation accuracy. This confirms that generation-level benchmark evaluation must remain the source of truth. The newer tool-call result is encouraging precisely because it improved both the probe objective and the final evaluation.

### Scoring Is Part of the Research System

An eval pipeline can create false failures if it is too strict about formatting. Those false failures then become bad search probes. Adding semantic assessment modes such as `json_match` and `numeric_final` is therefore not cosmetic; it changes what the adaptation loop learns from.

### Layer Choice Matters

High-impact layers can produce larger probe movement. In the current code, the default search focuses on layers `0-4`, `34`, and `35`, with `gate_proj`, `up_proj`, and `down_proj` enabled. Lower-impact middle layers may still be useful for safer or broader patches, but the tool-call work has benefited from targeting rows with higher measured movement.

### The Patch Payload Stays Tiny

The Bankai substrate remains attractive because the payload being selected by the eval pipeline is extremely small:

| Patch | Flips | Metadata-excluded size |
|---|---:|---:|
| Stable weighted anneal | 1 | 12 bytes |
| Improved-probe shortlist | 2 | 24 bytes |
| Small anneal | 3 | 36 bytes |
| Curated shortlist | 5 | 60 bytes |
| Tool-call dynamic probe patch | 16 | 192 bytes |

Even if JSON metadata is much larger than the compact representation, the underlying patch payload is negligible relative to an 8B model.

## 9. Current Limitations

This work now shows a narrow generation-level improvement on one practical tool-call dataset, but it does not yet prove broad benchmark improvement. Current evidence supports feasibility of reversible patching, dynamic probe construction, and targeted task gains under controlled evaluation.

Known limitations:

- GSM8K patches have not yet translated probe gains into generation accuracy gains.
- Tool-call improvement is currently demonstrated on one 70-row dataset and needs replication on larger held-out sets.
- Dynamic JSON probes depend on being able to parse expected and generated structured outputs.
- Boundary probes are task-specific; every schema needs careful failure analysis.
- Numeric-final scoring handles scalar numeric answers but not symbolic equivalence, units, intervals, or multi-answer math.
- Runs on the 2020 M1 MacBook are slow, limiting search depth.
- Search trajectories are sensitive to probe selection, layer selection, and budget.
- Two-pass shortlist search is currently too slow on the M1 with improved probes.

## 10. Near-Term Plan

The immediate next step is to deepen the dynamic-probe path rather than only increasing GSM8K search budget.

Recommended next work:

- Build larger held-out tool-call datasets so the 16-flip improvement can be checked for generalization.
- Add more structured-output probe builders for non-tool JSON schemas.
- Add more semantic assessment modes before treating eval failures as adaptation targets.
- Continue targeting first divergent boundaries: tool name, argument key, argument value, and first semantically wrong free-text span.
- Compare row-level search against smaller group-level flips once the probe objective is stable.
- Run ablations for layer sets `[0-4, 34, 35]`, `[17-21]`, and mixed early/final profiles.
- Keep generation-level comparison as the promotion gate, with fixed/regression/still-wrong reporting in the UI.

Promotion criteria for a patch:

- Improves final task accuracy on held-out evaluation rows.
- Produces zero or very few regressions on base-passing rows.
- Fixes mistakes for the expected reason, visible in side-by-side generations.
- Has positive probe fitness on target probes and non-negative control movement.
- Remains small enough to preserve the Bankai deployment story.

## 11. Longer-Term Research Direction

If benchmark-specific patches begin to show reliable gains, the next phase should move from benchmark patches to capability-family patches:

- math
- code
- instruction following
- tool use

The router should then evolve from benchmark-name routing to request classification. The realistic deployment story is not “GSM8K patch for GSM8K”; it is “math patch for math-like requests.”

## 12. Conclusion

The current proof of concept validates the mechanical foundation of patch-routed specialization for a true 1-bit Bonsai model:

- Real packed row-XOR patches can be applied and reverted.
- Patch artifacts are extremely small.
- Probe-driven search can find non-empty patches and, on the current tool-call evaluation, produce a measured generation-level improvement.
- Dynamic boundary probes are the key practical advance: build them from actual base failures and optimize the first wrong/correct decision.
- Semantic scoring is part of probe quality: raw regexes and formatting-sensitive math failures should not become adaptation targets.
- Multi-token continuation scoring is necessary for structured outputs.
- Generation-level evaluation is essential because probe gains alone are not enough.

The main open question is no longer just whether XOR patches can move behavior. They can. The sharper question is whether dynamically generated, boundary-aware probes can make those changes reliable across larger held-out datasets and broader task families. The current tool-call result is the strongest evidence so far that this direction is viable.

## References

```bibtex
@misc{saravanan2026bankai,
  title   = {Bankai: Ultra-Sparse Adaptation of 1-Bit LLMs via XOR Patches},
  author  = {Saravanan, Nikshep},
  year    = {2026},
  url     = {https://github.com/nikshepsvn/bankai}
}
```
