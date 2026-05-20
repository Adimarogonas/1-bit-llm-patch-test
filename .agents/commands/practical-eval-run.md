# practical-eval-run

Run the three-way practical eval pipeline against a normalized JSONL dataset or a CSV with `prompt` plus `expected`/`grader`.

Scoring is a per-row decision: rows with `expected` are graded programmatically (see [practical-eval-dataset.md](practical-eval-dataset.md) for assessment modes), even when `--judge-command` is set. The judge runs only for rows without `expected` or with `assessment=llm_judge` — see [llm-judge.md](llm-judge.md).

Default comparison slots:

- `bankai-patched`: patched Bankai MLX model.
- `bonsai-reference-mlx`: configurable reference MLX model.
- terminal agent: any command that accepts the rendered prompt on stdin, or uses `{prompt}` in the command string.

Example:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-run \
  --dataset data/practical_evals/sentiment_analysis.jsonl \
  --output-dir results/eval_runs/run_sentiment \
  --bankai-model prism-ml/Bonsai-8B-mlx-1bit \
  --bankai-patch patches/gsm8k_real_patch.json \
  --reference-model prism-ml/Bonsai-8B-unpacked \
  --agent-command "codex exec --model gpt-5.4-mini"
```

To run only a terminal command for smoke testing:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-run \
  --dataset data/practical_evals/example_sentiment.csv \
  --output-dir results/eval_runs/run_terminal_smoke \
  --no-bankai \
  --no-reference \
  --agent-name echo-agent \
  --agent-command "printf positive"
```
