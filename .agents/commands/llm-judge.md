# llm-judge

Configure an optional command-based LLM judge for rows with a `grader` column.

Rows with `expected` are scored programmatically by default, even when a judge command is configured. Use `assessment=llm_judge` only when the row should be judged by the LLM instead of matched against `expected`.

The judge command receives a JSON payload on stdin unless the command string contains `{payload}`. Payload fields:

- `prompt`
- `expected`
- `prediction`
- `grader`
- `metadata`

Preferred judge output is JSON:

```json
{"passed": true, "score": 1.0, "reason": "The output satisfies the grader."}
```

Plain text is also accepted. Outputs starting with `pass`, `true`, `yes`, or `1` pass; outputs starting with `fail`, `false`, `no`, or `0` fail.

Example:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-run \
  --dataset data/practical_evals/example_llm_grader.csv \
  --output-dir results/eval_runs/run_llm_judged \
  --judge-command "codex exec --model gpt-5.4-mini" \
  --agent-command "codex exec --model gpt-5.4-mini"
```
