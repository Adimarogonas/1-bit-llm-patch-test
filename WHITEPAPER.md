# Patch-Routed Specialization for True 1-Bit LLMs

## A Bankai/Bonsai Proof-of-Concept Report

Date: May 20, 2026

## Abstract

This project investigates whether a single true 1-bit language model can be specialized at request time using tiny reversible XOR patches. Instead of loading multiple expert models or attaching larger adapters, the system keeps one shared Bonsai 8B base model and applies a small Bankai-style row-XOR patch before inference. The patch can then be reverted exactly or swapped for another patch on the next request.

The current proof of concept validates the core patch mechanics on `prism-ml/Bonsai-8B-mlx-1bit`, but the main technical progress is now probe construction rather than patch mechanics alone. Early GSM8K probes could move logit objectives without reliably moving generation accuracy. The newer adaptive pipeline builds probes from the base model's actual mistakes, targets the first wrong/correct decision boundary, supports multi-token continuations, and adds controls from examples the base model already gets right.

On a 70-row `tool_call_selection.csv` evaluation, a 16-flip patch improved accuracy from 62/70 to 66/70, or 88.6% to 94.3%, with zero observed regressions. The result is still narrow and dataset-specific, but it is the first clear sign in this project that boundary-aware dynamic probes can convert XOR row flips into generation-level task improvement.

## 1. Motivation

Large model specialization is usually achieved by storing additional model weights, adapters, LoRA deltas, or separate expert models. That works, but it increases storage, deployment complexity, and sometimes inference overhead.

The Bankai hypothesis is different:

> If the base model is truly 1-bit, then useful behavioral changes may be expressible as small sets of reversible bit flips.

For a 1-bit model, an XOR patch can flip selected packed weight rows. Applying the same XOR again reverts the model exactly. This creates a simple request-time specialization mechanism:

1. Load one shared base model.
2. Select a patch for the task.
3. XOR selected rows in-place.
4. Run inference.
5. XOR the same rows again to restore the base model.

This is not a true mixture-of-experts architecture. It is better described as patch-routed specialization or MoE-like specialization behavior with near-zero parameter overhead.

## 2. System Overview

The proof of concept currently contains:

- A reproducible Python project with data, probes, patches, configs, results, and source modules.
- Dataset acquisition and normalization for GSM8K, HumanEval+, IFEval, and BFCL.
- Probe generation for benchmark-specific supervision.
- Real MLX/Bonsai model loading through the PrismML MLX fork.
- Live row-XOR patch application and reversion on packed 1-bit MLP weights.
- Search runners for greedy search, shortlist search, two-pass shortlist search, and simulated annealing shortlist search.
- GSM8K generation-level evaluation with base-vs-patched comparison.
- Adaptive evaluation that runs the base model, extracts failures, builds dynamic probes, searches a patch, and reruns patched evaluation.
- A results UI that reports fixed cases, regressions, still-wrong cases, changed outputs, and side-by-side base/patched generations.
- Patch checkpointing after every accepted flip so interrupted or failed searches can be recovered.
- Team runbook commands for distributing heavier patch searches across M3/M4 Apple Silicon machines.

The initial benchmark focus was GSM8K because math-answer correctness is easy to evaluate and explain. The more recent practical focus is tool-call selection because it exposes the exact failure mode that simple probes miss: the model often uses a plausible but wrong function name or argument key.

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

The project now uses two probe families.

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

### 4.3 Multi-Token Probe Scoring

The scorer now supports multi-token continuations. For dynamic probes, it tokenizes `prompt + correct_completion` and `prompt + wrong_completion`, finds the shared token prefix, and scores the divergent suffixes with teacher-forced mean log probability. This avoids reducing a decision such as `send_email` vs `escalate_ticket` or a multi-token email address to one brittle token.

The same mechanism remains compatible with non-JSON tasks. If the output is not JSON-like, the dynamic builder falls back to a label-style prompt with the expected answer as the correct continuation and the model's actual wrong answer as the negative continuation.

### 4.4 Regression Controls

Dynamic probes also include controls from examples the base model already passed. For tool-call selection, controls preserve:

- Tool names, such as `search_orders` vs `cancel_subscription`.
- Tool aliases that are easy to confuse, such as `send_email` vs `notify`.
- Argument keys and values already produced correctly.

This is why the results UI now separates fixed cases from regressions. A patch that fixes the target class but turns correct `search_orders` calls into repeated `cancel_subscription` calls is not acceptable, even if its raw probe fitness improves.

## 5. Search Algorithms Tested

### 5.1 Greedy Real Search

The simplest search samples a candidate row, flips it, evaluates probe fitness, and keeps it only if it improves the current patch. This is straightforward but expensive and myopic.

Plain-English behavior:

```text
Try one flip. If it helps immediately, keep it. Otherwise undo it.
```

This can get stuck because row flips interact. A flip that looks bad alone may be useful with another flip, and a flip that looks good alone may damage generation when stacked with other flips.

### 5.2 Shortlist Search

Shortlist search samples a pool of candidate rows, cheaply screens them on a small probe subset, then fully evaluates only the best-looking finalists.

Plain-English behavior:

```text
Look at a batch of possible flips cheaply.
Spend full evaluation only on the best few.
Keep the best improving candidate.
```

This was much faster than naive greedy search and produced real non-empty patches.

### 5.3 Two-Pass Shortlist Search

Two-pass search adds a second screening stage:

1. Cheap first pass over a wider pool.
2. More careful second pass over the best mid-candidates.
3. Full evaluation over the final top candidates.

This is conceptually sound, but on the 2020 M1 MacBook with 16GB RAM it was too slow for practical iteration using the improved probes.

### 5.4 Simulated Annealing Shortlist Search

The current most promising search variant is simulated annealing over shortlisted candidates. It proposes patch-state moves:

- Add a flip.
- Remove a flip.
- Swap one flip for another.

It accepts improvements, and sometimes accepts worse states early in the run. It always saves the best patch state seen.

Plain-English behavior:

```text
Try messy detours early.
Allow undoing and swapping flips.
Gradually become stricter.
Save the best patch found along the way.
```

This matters because Bankai row flips are blunt. A row flip changes 4096 packed bits, so the useful unit may be a small combination of flips rather than a single locally optimal flip.

### 5.5 Search Candidate Updates

The current real search now samples from the MLP projections that have actually shown movement in practice:

```text
gate_proj
up_proj
down_proj
```

Candidate row selection is scale-guided. Instead of taking the first rows in a tensor, the candidate builder ranks rows by scale magnitude and samples from the highest-scale rows in the selected layers. This makes the search budget land on rows that are more likely to move logits.

Every accepted flip is checkpointed to a recoverable patch file. This matters operationally because long MLX searches can fail after finding useful flips; the patch should not be lost because final JSON serialization or a later evaluation step failed.

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

### 7.2 Earlier Patch Comparison

A 50-example GSM8K comparison with the corrected harness showed:

| System | Correct | Accuracy | Delta vs base | Changed generations |
|---|---:|---:|---:|---:|
| Base | 42/50 | 84% | 0% | n/a |
| Curated shortlist patch | 42/50 | 84% | 0% | 26 |
| All-layer/wide patch | 38/50 | 76% | -8% | 30 |

Interpretation:

- The curated patch changed outputs but did not change accuracy.
- The all-layer/wide patch caused a generation-level regression.
- This supports the need for safer layer selection and generation-level validation.

### 7.3 Improved-Probe Shortlist Search

A one-pass improved-probe shortlist run found:

```text
Patch: gsm8k_real_patch_curated_data_v2_pool32_topk2_t6_c3_r3.json
Search: shortlist
Rounds: 3
Pool: 32
Top-k: 2
Target probes: 6
Control probes: 3
Fitness: 0.009114583333333334
Flips: 2
Rows:
  L12.up_proj[32]
  L20.up_proj[5]
Patch size: 24 bytes metadata-excluded
```

### 7.4 Simulated Annealing Search

A very small annealing run found a higher probe fitness with less search budget:

```text
Patch: gsm8k_real_patch_anneal_s4_pool4_topk1.json
Search: simulated annealing shortlist
Steps: 4
Pool: 4
Top-k: 1
Target probes: 3
Control probes: 1
Layers: [1, 4, 8]
Fitness: 0.015625
Flips: 3
Rows:
  L8.gate_proj[35]
  L8.gate_proj[29]
  L8.gate_proj[19]
Patch size: 36 bytes metadata-excluded
```

A 50-example GSM8K generation eval for this annealed patch showed:

```text
Base:    42/50 = 84%
Patched: 42/50 = 84%
Delta:   0%
Changed generations: 3
Correctness changes: 0
```

Interpretation:

- The patch changed some generations.
- It did not improve accuracy.
- It also did not reproduce the -8% regression seen in the all-layer/wide patch.

### 7.5 Stable Layer-Weighted Annealing

After adding layer profiles and impact weighting, a small stable-profile run found:

```text
Patch: gsm8k_real_patch_anneal_stable_weighted_s4_pool4_topk1.json
Search: simulated annealing shortlist
Steps: 4
Pool: 4
Top-k: 1
Target probes: 3
Control probes: 1
Layer profile: stable
Impact weighted: true
Fitness: 0.013020833333333334
Flips: 1
Row:
  L20.gate_proj[25]
Patch size: 12 bytes metadata-excluded
```

A 20-example GSM8K generation eval showed:

```text
Base:    15/20 = 75%
Patched: 15/20 = 75%
Delta:   0%
Changed generations: 0
Correctness changes: 0
```

Interpretation:

- The stable layer-20 one-flip patch appears highly non-disruptive.
- At this patch strength, it is behaviorally inert on the first 20 generation examples.
- This is useful as a safety signal but not yet a performance gain.

### 7.6 Tool-Call Dynamic Probe Search

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

## 8. Key Technical Lessons

### Probe Formation Is Now the Core Method

The main progress came from changing what is optimized, not from making the patch operation more complex. Dynamic probes built from actual base failures are much more useful than generic probes because they target the decision the model really got wrong.

For structured outputs, the correct probe is usually not at the end of the answer. It is at the first divergent structural choice: function name, argument key, or argument value.

### Multi-Token Scoring Is Necessary

Tool names, field names, emails, dates, and many labels are not reliably represented by one token. Scoring the divergent multi-token suffix makes the objective closer to the generated behavior while still staying much cheaper than full generation inside the search loop.

### Probe Fitness Does Not Guarantee Generation Gains

Several GSM8K patches improved the probe objective without improving generation accuracy. This confirms that generation-level benchmark evaluation must remain the source of truth. The newer tool-call result is encouraging precisely because it improved both the probe objective and the final evaluation.

### Layer Choice Matters

High-impact layers can produce larger probe movement. In the current code, the default search focuses on layers `0-4`, `34`, and `35`, with `gate_proj`, `up_proj`, and `down_proj` enabled. Lower-impact middle layers may still be useful for safer or broader patches, but the tool-call work has benefited from targeting rows with higher measured movement.

### Annealing Looks Promising

The small annealing run found 3 flips with higher probe fitness faster than a comparable shortlist strategy. More importantly, it avoided the observed generation-level regression from the broader all-layer patch.

The likely reason is that annealing searches patch states rather than greedily accumulating locally good flips. It can add, remove, and swap rows, which is better aligned with interacting row flips.

### Current Patches Are Extremely Small

The observed real patches are tiny:

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
- Runs on the 2020 M1 MacBook are slow, limiting search depth.
- Search trajectories are sensitive to probe selection, layer selection, and budget.
- Two-pass shortlist search is currently too slow on the M1 with improved probes.

## 10. Near-Term Plan

The immediate next step is to deepen the dynamic-probe path rather than only increasing GSM8K search budget.

Recommended next work:

- Build larger held-out tool-call datasets so the 16-flip improvement can be checked for generalization.
- Add more structured-output probe builders for non-tool JSON schemas.
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
- Multi-token continuation scoring is necessary for structured outputs.
- Simulated annealing over shortlist candidates can find compact, non-destructive patches faster than purely greedy accumulation.
- Generation-level evaluation is essential because probe gains alone are not enough.

The main open question is no longer just whether XOR patches can move behavior. They can. The sharper question is whether dynamically generated, boundary-aware probes can make those changes reliable across larger held-out datasets and broader task families. The current tool-call result is the strongest evidence so far that this direction is viable.
