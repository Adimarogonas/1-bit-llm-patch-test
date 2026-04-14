# Patch-Routed Specialization for True 1-Bit LLMs

## A Bankai/Bonsai Proof-of-Concept Report

Date: April 14, 2026

## Abstract

This project investigates whether a single true 1-bit language model can be specialized at request time using tiny reversible XOR patches. Instead of loading multiple expert models or attaching larger adapters, the system keeps one shared Bonsai 8B base model and applies a small Bankai-style row-XOR patch before inference. The patch can then be reverted exactly or swapped for another patch on the next request.

The current proof of concept validates the core patch mechanics on `prism-ml/Bonsai-8B-mlx-1bit`, implements benchmark-specific probe generation, and explores several patch-search strategies for GSM8K. The early result is not yet benchmark improvement, but it shows that real reversible 1-bit row patches can be found, applied, reverted, and evaluated with negligible storage overhead. Simulated annealing over shortlist candidates currently looks more promising than purely greedy search because it can find compact, non-destructive patches faster.

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
- Team runbook commands for distributing heavier patch searches across M3/M4 Apple Silicon machines.

The initial benchmark focus is GSM8K because math-answer correctness is easy to evaluate and explain.

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

Bankai patch search performs better with probe-style objectives than raw benchmark examples. For GSM8K, the project moved from simple answer-token probes to stronger final-answer probes:

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
stable:     [8, 12, 16, 17, 18, 19, 20, 21, 22, 24, 28]
balanced:   [5, 8, 12, 16, 17, 18, 19, 20, 21, 22, 24, 28, 32, 35]
aggressive: [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 34, 35]
```

The GSM8K default was changed away from the old high-impact default `[1, 2, 3, 4, 34]` and toward a lower-regression set:

```text
[8, 12, 16, 17, 18, 19, 20, 21, 22, 24, 28, 32]
```

Search can also use impact-weighted sampling, which downweights high-impact layers when drawing candidate rows.

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

## 8. Key Technical Lessons

### Probe Fitness Does Not Guarantee Generation Gains

Several patches improved the probe objective without improving GSM8K generation accuracy. This confirms that generation-level benchmark evaluation must remain the source of truth.

### Layer Choice Matters

High-impact layers can produce larger probe movement but appear more likely to disrupt generation. Lower-impact middle layers may be safer but may need larger or better-directed searches to produce visible behavior changes.

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

Even if JSON metadata is much larger than the compact representation, the underlying patch payload is negligible relative to an 8B model.

## 9. Current Limitations

This work does not yet prove benchmark improvement. Current evidence supports feasibility of reversible patching and search infrastructure, not final task gains.

Known limitations:

- GSM8K is the only benchmark with meaningful generation-level evaluation so far.
- Probe improvements have not yet translated into generation accuracy gains.
- Runs on the 2020 M1 MacBook are slow, limiting search depth.
- The best current stable patch is non-disruptive but also behaviorally inert on 20 examples.
- Search trajectories are sensitive to probe selection, layer selection, and budget.
- Two-pass shortlist search is currently too slow on the M1 with improved probes.

## 10. Near-Term Plan

The immediate next step is distributed search on newer M3/M4 MacBooks using the team runbook.

Recommended heavy run:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search gsm8k \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 32 --pool 16 --topk 3 \
  --target-probes 8 --control-probes 4 \
  --layer-profile stable --impact-weighted \
  --start-temp 0.03 --end-temp 0.0008 \
  --output patches/gsm8k_real_patch_anneal_stable_s32_pool16_topk3.json
```

Recommended very heavy follow-up:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search gsm8k \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 72 --pool 32 --topk 4 \
  --target-probes 12 --control-probes 6 \
  --layer-profile balanced --impact-weighted \
  --start-temp 0.04 --end-temp 0.0005 \
  --output patches/gsm8k_real_patch_anneal_balanced_s72_pool32_topk4.json
```

Every candidate patch should then be tested with:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-gsm8k-compare \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --patch <patch-name>.json \
  --limit 50 \
  --max-tokens 400
```

Promotion criteria for a patch:

- Improves or preserves 50-example GSM8K accuracy.
- Produces visible generation changes.
- Does not increase truncation or formatting failures.
- Has positive probe fitness on target probes.
- Does not show excessive validation/control degradation.

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
- Probe-driven search can find non-empty patches.
- Simulated annealing over shortlist candidates can find compact, non-destructive patches faster than purely greedy accumulation.
- Generation-level evaluation is essential because probe gains alone are not enough.

The main open question remains whether deeper searches on faster machines can convert probe-level improvements into reliable benchmark-level gains. The infrastructure is now in place to test that question across GSM8K, HumanEval+, IFEval, and BFCL.

