# adaptive-eval

Run the practical eval adaptation loop:

1. Evaluate the base MLX model on a CSV dataset.
2. Build dynamic token-pair probes from base-model failures.
3. Use passing rows as control probes.
4. Search for a Bankai row-XOR patch.
5. Re-run the same eval with the patched model.

Example:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-adapt \
  --dataset data/practical_evals/example_routing.csv \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --output-dir results/eval_runs \
  --rounds 4 \
  --pool 8 \
  --topk 2 \
  --accept-per-round 1
```

Beefier local run:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-adapt \
  --dataset data/practical_evals/example_routing.csv \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --output-dir results/eval_runs \
  --rounds 10 \
  --pool 24 \
  --topk 4 \
  --accept-per-round 2 \
  --search-mode greedy \
  --max-iters 600 \
  --fitness-mode mean \
  --target-probes 32 \
  --control-probes 16 \
  --candidate-rows 96 \
  --max-flips 32 \
  --layer-profile balanced \
  --impact-weighted
```

Maximum exploratory run:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-adapt \
  --dataset data/practical_evals/example_routing.csv \
  --model prism-ml/Bonsai-8B-mlx-1bit \
  --output-dir results/eval_runs \
  --rounds 18 \
  --pool 48 \
  --topk 8 \
  --accept-per-round 3 \
  --search-mode greedy \
  --max-iters 1200 \
  --fitness-mode min \
  --target-probes 64 \
  --control-probes 32 \
  --candidate-rows 0 \
  --max-flips 48 \
  --layer-profile aggressive \
  --impact-weighted
```

Search modes:

- `--search-mode shortlist` keeps the previous batched shortlist search and can accept `--accept-per-round` candidates each round.
- `--search-mode greedy` uses screened greedy hill climbing: sample one row, test it on the two worst target probes, fully score only if the screen improves, and keep it only if fitness improves.
- `--fitness-mode mean` optimizes average target-probe improvement.
- `--fitness-mode min` optimizes the worst target-probe improvement and is stricter.
- `--candidate-rows 0` searches all rows in each selected layer/projection. Use this only for heavier local runs.

Artifacts are saved in `results/eval_runs/adaptive_<timestamp>/`:

- `base/summary.json`
- `base/details.json`
- `dynamic_probes.jsonl`
- `dynamic_patch.json`
- `dynamic_search.json`
- `patched/summary.json`
- `patched/details.json`
- `adaptive_manifest.json`
