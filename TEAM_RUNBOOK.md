# Team Runbook: Real Bankai Patch Search

Use this when distributing real patch search across multiple Apple Silicon laptops.

## Goal

Each teammate runs a real search for one benchmark on their own machine and returns:

- the patch JSON
- the real search trajectory JSON
- the runtime doctor JSON
- the model inspect JSON

Optional:

- a small prompt before/after sanity check
- a held-out comparison summary

## Benchmark Assignment

Assign one machine per benchmark:

- Machine A: `gsm8k`
- Machine B: `humaneval_plus`
- Machine C: `ifeval`
- Machine D: `bfcl`

If you have extra machines:

- Machine E: longer `gsm8k`
- Machine F: longer `ifeval`

## Repo Setup

```bash
cd /path/to/Synapse/bankai_poc
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install mlx-lm
pip install --force-reinstall 'mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism'
```

If the Prism MLX build fails because of missing Metal tools:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Then retry the Prism MLX install.

## Environment Validation

Run these first:

```bash
source .venv/bin/activate
bankai-poc doctor
bankai-poc inspect-model --model prism-ml/Bonsai-8B-mlx-1bit
bankai-poc prep-all
```

Expected:

- `results/doctor.json` should show:
  - `mlx_available: true`
  - `mlx_lm_available: true`
  - `prism_mlx_ready: true`
- `results/model_inspect.json` should show:
  - `patchable: true`

## Smoke Test

Before a longer search, run a short annealed shortlist search to confirm the machine works:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search gsm8k \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 1 --pool 2 --topk 1 \
  --target-probes 2 --control-probes 1 \
  --layer-profile stable --impact-weighted \
  --output patches/gsm8k_real_patch_anneal_smoke.json
```

If this produces a `*_real_patch.json` and `*_real_search.json`, the machine is ready.

## Recommended Search Presets

### Interactive

Use this to confirm the setup is healthy:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search <benchmark> \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 4 --pool 4 --topk 1 \
  --target-probes 3 --control-probes 1 \
  --layer-profile stable --impact-weighted \
  --output patches/<benchmark>_real_patch_anneal_interactive.json
```

### Medium

Use this for the first serious artifact on M1/M2 machines:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search <benchmark> \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 12 --pool 8 --topk 2 \
  --target-probes 6 --control-probes 3 \
  --layer-profile stable --impact-weighted \
  --output patches/<benchmark>_real_patch_anneal_stable_s12_pool8_topk2.json
```

### Long

Use this on M3/M4 machines after the medium run looks healthy:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search <benchmark> \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 32 --pool 16 --topk 3 \
  --target-probes 8 --control-probes 4 \
  --layer-profile stable --impact-weighted \
  --start-temp 0.03 --end-temp 0.0008 \
  --output patches/<benchmark>_real_patch_anneal_stable_s32_pool16_topk3.json
```

### Very Long

Only run this if the machine is clearly fast enough and the search is producing accepted flips:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search <benchmark> \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 72 --pool 32 --topk 4 \
  --target-probes 12 --control-probes 6 \
  --layer-profile balanced --impact-weighted \
  --start-temp 0.04 --end-temp 0.0005 \
  --output patches/<benchmark>_real_patch_anneal_balanced_s72_pool32_topk4.json
```

Do not start with the very long command. First confirm:

- accepted flips appear at smaller budgets
- per-step runtime is acceptable
- fitness improves after accepted flips

After each run, copy the search trajectory before launching another search because `results/<benchmark>_real_search.json` is overwritten:

```bash
cp results/<benchmark>_real_search.json results/<benchmark>_real_search_anneal_stable_s32_pool16_topk3.json
```

## Benchmark-Specific Commands

### GSM8K

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search gsm8k \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 32 --pool 16 --topk 3 \
  --target-probes 8 --control-probes 4 \
  --layer-profile stable --impact-weighted \
  --start-temp 0.03 --end-temp 0.0008 \
  --output patches/gsm8k_real_patch_anneal_stable_s32_pool16_topk3.json
cp results/gsm8k_real_search.json results/gsm8k_real_search_anneal_stable_s32_pool16_topk3.json
```

### HumanEval+

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search humaneval_plus \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 32 --pool 16 --topk 3 \
  --target-probes 8 --control-probes 4 \
  --layer-profile stable --impact-weighted \
  --start-temp 0.03 --end-temp 0.0008 \
  --output patches/humaneval_plus_real_patch_anneal_stable_s32_pool16_topk3.json
cp results/humaneval_plus_real_search.json results/humaneval_plus_real_search_anneal_stable_s32_pool16_topk3.json
```

### IFEval

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search ifeval \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 32 --pool 16 --topk 3 \
  --target-probes 8 --control-probes 4 \
  --layer-profile stable --impact-weighted \
  --start-temp 0.03 --end-temp 0.0008 \
  --output patches/ifeval_real_patch_anneal_stable_s32_pool16_topk3.json
cp results/ifeval_real_search.json results/ifeval_real_search_anneal_stable_s32_pool16_topk3.json
```

### BFCL

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search bfcl \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 32 --pool 16 --topk 3 \
  --target-probes 8 --control-probes 4 \
  --layer-profile stable --impact-weighted \
  --start-temp 0.03 --end-temp 0.0008 \
  --output patches/bfcl_real_patch_anneal_stable_s32_pool16_topk3.json
cp results/bfcl_real_search.json results/bfcl_real_search_anneal_stable_s32_pool16_topk3.json
```

## Heavier M3/M4 Follow-Up Commands

Use these only on fast M3/M4 machines after the benchmark-specific command above completes.

### GSM8K Balanced Very Long

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search gsm8k \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 72 --pool 32 --topk 4 \
  --target-probes 12 --control-probes 6 \
  --layer-profile balanced --impact-weighted \
  --start-temp 0.04 --end-temp 0.0005 \
  --output patches/gsm8k_real_patch_anneal_balanced_s72_pool32_topk4.json
cp results/gsm8k_real_search.json results/gsm8k_real_search_anneal_balanced_s72_pool32_topk4.json
```

### HumanEval+ Balanced Very Long

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search humaneval_plus \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 72 --pool 32 --topk 4 \
  --target-probes 12 --control-probes 6 \
  --layer-profile balanced --impact-weighted \
  --start-temp 0.04 --end-temp 0.0005 \
  --output patches/humaneval_plus_real_patch_anneal_balanced_s72_pool32_topk4.json
cp results/humaneval_plus_real_search.json results/humaneval_plus_real_search_anneal_balanced_s72_pool32_topk4.json
```

### IFEval Balanced Very Long

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search ifeval \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 72 --pool 32 --topk 4 \
  --target-probes 12 --control-probes 6 \
  --layer-profile balanced --impact-weighted \
  --start-temp 0.04 --end-temp 0.0005 \
  --output patches/ifeval_real_patch_anneal_balanced_s72_pool32_topk4.json
cp results/ifeval_real_search.json results/ifeval_real_search_anneal_balanced_s72_pool32_topk4.json
```

### BFCL Balanced Very Long

```bash
PYTHONUNBUFFERED=1 bankai-poc real-anneal-shortlist-search bfcl \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --steps 72 --pool 32 --topk 4 \
  --target-probes 12 --control-probes 6 \
  --layer-profile balanced --impact-weighted \
  --start-temp 0.04 --end-temp 0.0005 \
  --output patches/bfcl_real_patch_anneal_balanced_s72_pool32_topk4.json
cp results/bfcl_real_search.json results/bfcl_real_search_anneal_balanced_s72_pool32_topk4.json
```

## Sanity Check After Search

Replace the prompt and patch name as needed:

```bash
bankai-poc real-apply --model prism-ml/Bonsai-8B-mlx-1bit --patch gsm8k_real_patch.json --prompt "2 + 2 =" --max-tokens 8
```

## Optional GSM8K 50-Example Comparison

For the GSM8K machine:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-gsm8k-compare \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --patch gsm8k_real_patch_anneal_stable_s32_pool16_topk3.json \
  --limit 50 \
  --max-tokens 400
```

This writes:

- `results/gsm8k_real_compare_summary.json`
- `results/gsm8k_real_compare_details.json`

## Files To Send Back

Each teammate should return these files:

- `results/doctor.json`
- `results/model_inspect.json`
- `results/<benchmark>_real_search_anneal_stable_s32_pool16_topk3.json`
- `patches/<benchmark>_real_patch_anneal_stable_s32_pool16_topk3.json`

If they ran the balanced very long follow-up:

- `results/<benchmark>_real_search_anneal_balanced_s72_pool32_topk4.json`
- `patches/<benchmark>_real_patch_anneal_balanced_s72_pool32_topk4.json`

If they ran GSM8K comparison:

- `results/gsm8k_real_compare_summary.json`
- `results/gsm8k_real_compare_details.json`

## What To Check In Their Results

For `results/<benchmark>_real_search_*.json`:

- did any candidates get accepted?
- what was the best fitness?
- how many screen rejects vs full rejects?
- where were accepted flips located?

For `patches/<benchmark>_real_patch_*.json`:

- `n_flips`
- `bits_flipped`
- `size_bytes`

## Naming Convention

Ask teammates not to rename artifacts after a run. Prefer explicit output names:

- `gsm8k_real_patch_anneal_stable_s32_pool16_topk3.json`
- `humaneval_plus_real_patch_anneal_stable_s32_pool16_topk3.json`
- `ifeval_real_patch_anneal_stable_s32_pool16_topk3.json`
- `bfcl_real_patch_anneal_stable_s32_pool16_topk3.json`

If they want to preserve multiple runs, they should copy the files after each run:

```bash
cp results/gsm8k_real_search.json results/gsm8k_real_search_anneal_stable_s32_pool16_topk3.json
```

## Suggested Team Message

Use this when sending work out:

```text
Please run the Bankai POC search in bankai_poc/ for your assigned benchmark.

1. Set up the venv and install Prism MLX as described in TEAM_RUNBOOK.md
2. Run:
   bankai-poc doctor
   bankai-poc inspect-model --model prism-ml/Bonsai-8B-mlx-1bit
   bankai-poc prep-all
3. Run the benchmark-specific annealed shortlist command from the runbook
4. Send back:
   results/doctor.json
   results/model_inspect.json
   results/<benchmark>_real_search_anneal_stable_s32_pool16_topk3.json
   patches/<benchmark>_real_patch_anneal_stable_s32_pool16_topk3.json

If the machine is fast and the first run accepts flips, try the balanced very long M3/M4 follow-up command.
If Prism MLX fails to build, install the Metal Toolchain and retry.
```
