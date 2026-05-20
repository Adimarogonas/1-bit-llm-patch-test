# practical-eval-dataset

Create or normalize a practical CSV eval dataset for Bankai Eval Studio.

Minimum CSV columns:

- `prompt`: the full task prompt sent to the model.
- `expected` or `grader`: use `expected` for exact normalized label/answer matching; use `grader` for LLM judge instructions.

Recommended optional columns:

- `id`: stable case identifier.
- `system`: system prompt for this row.
- `assessment`: programmatic scorer. Defaults to `exact_normalized` when `expected` is present.
- task-specific metadata columns such as `segment`, `category`, `route_set`, or `difficulty`.

Scoring routing: rows with `expected` are scored programmatically by default, **even when a judge command is configured**. The LLM judge only runs for rows where `expected` is blank, or where the row sets `assessment=llm_judge`.

Programmatic assessment modes:

- `exact_normalized`: lowercase, trim, and collapse whitespace before comparing `prediction` to `expected`.
- `classification`: alias for `exact_normalized`.
- `routing`: alias for `exact_normalized`.
- `exact`: raw string equality after trimming.
- `contains`: normalized prediction must contain normalized expected.
- `regex`: `expected` is treated as a case-insensitive regex.
- `json_field:<field>`: parse prediction as JSON and compare the named field to `expected`; dot paths are supported.
- `llm_judge`: run the configured judge command instead of programmatic scoring.

Example:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-dataset \
  --csv-source data/practical_evals/example_sentiment.csv \
  --task-name sentiment_analysis \
  --limit 50 \
  --output data/practical_evals/sentiment_analysis.jsonl
```

Column overrides:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-dataset \
  --csv-source path/to/eval.csv \
  --task-name routing \
  --prompt-column prompt \
  --expected-column expected \
  --grader-column grader \
  --id-column id \
  --system-column system \
  --assessment-column assessment
```
