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

Before a longer search, run a short real search to confirm the machine works:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search gsm8k --model prism-ml/Bonsai-8B-mlx-1bit --iters 6 --target-probes 6 --control-probes 3
```

If this produces a `*_real_patch.json` and `*_real_search.json`, the machine is ready.

## Recommended Search Presets

### Interactive

Use this to confirm the setup is healthy:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search <benchmark> --model prism-ml/Bonsai-8B-mlx-1bit --iters 6 --target-probes 6 --control-probes 3
```

### Medium

Use this for the first serious artifact:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search <benchmark> --model prism-ml/Bonsai-8B-mlx-1bit --iters 24 --target-probes 8 --control-probes 4
```

### Long

Use this only after the medium run looks healthy:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search <benchmark> --model prism-ml/Bonsai-8B-mlx-1bit --iters 100 --target-probes 8 --control-probes 4
```

### Very Long

Only run this if the machine is clearly fast enough and the search is producing accepted flips:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search <benchmark> --model prism-ml/Bonsai-8B-mlx-1bit --iters 300 --target-probes 8 --control-probes 4
```

Do not start at `300` iterations. First confirm:

- accepted flips appear at smaller budgets
- per-iteration runtime is acceptable
- fitness improves after accepted flips

## Benchmark-Specific Commands

### GSM8K

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search gsm8k --model prism-ml/Bonsai-8B-mlx-1bit --iters 24 --target-probes 8 --control-probes 4
```

### HumanEval+

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search humaneval_plus --model prism-ml/Bonsai-8B-mlx-1bit --iters 24 --target-probes 8 --control-probes 4
```

### IFEval

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search ifeval --model prism-ml/Bonsai-8B-mlx-1bit --iters 24 --target-probes 8 --control-probes 4
```

### BFCL

```bash
PYTHONUNBUFFERED=1 bankai-poc real-search bfcl --model prism-ml/Bonsai-8B-mlx-1bit --iters 24 --target-probes 8 --control-probes 4
```

## Sanity Check After Search

Replace the prompt and patch name as needed:

```bash
bankai-poc real-apply --model prism-ml/Bonsai-8B-mlx-1bit --patch gsm8k_real_patch.json --prompt "2 + 2 =" --max-tokens 8
```

## Optional GSM8K 50-Example Comparison

For the GSM8K machine:

```bash
PYTHONUNBUFFERED=1 bankai-poc real-gsm8k-compare --model prism-ml/Bonsai-8B-mlx-1bit --patch gsm8k_real_patch.json --limit 50 --max-tokens 80
```

This writes:

- `results/gsm8k_real_compare_summary.json`
- `results/gsm8k_real_compare_details.json`

## Files To Send Back

Each teammate should return these files:

- `results/doctor.json`
- `results/model_inspect.json`
- `results/<benchmark>_real_search.json`
- `patches/<benchmark>_real_patch.json`

If they ran GSM8K comparison:

- `results/gsm8k_real_compare_summary.json`
- `results/gsm8k_real_compare_details.json`

## What To Check In Their Results

For `results/<benchmark>_real_search.json`:

- did any candidates get accepted?
- what was the best fitness?
- how many screen rejects vs full rejects?
- where were accepted flips located?

For `patches/<benchmark>_real_patch.json`:

- `n_flips`
- `bits_flipped`
- `size_bytes`

## Naming Convention

Ask teammates not to rename artifacts. Keep the default names:

- `gsm8k_real_patch.json`
- `humaneval_plus_real_patch.json`
- `ifeval_real_patch.json`
- `bfcl_real_patch.json`

If they want to preserve multiple runs, they should copy the files after each run:

```bash
cp patches/gsm8k_real_patch.json patches/gsm8k_real_patch_run1.json
cp results/gsm8k_real_search.json results/gsm8k_real_search_run1.json
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
3. Run the benchmark-specific real search command from the runbook
4. Send back:
   results/doctor.json
   results/model_inspect.json
   results/<benchmark>_real_search.json
   patches/<benchmark>_real_patch.json

If the machine is fast and the first run accepts flips, try a longer run.
If Prism MLX fails to build, install the Metal Toolchain and retry.
```

