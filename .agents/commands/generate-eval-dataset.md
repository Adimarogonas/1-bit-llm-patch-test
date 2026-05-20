# generate-eval-dataset

Generate a synthetic practical eval CSV for Bankai Eval Studio from a natural-language spec.

## Inputs

The user supplies (ask if missing):

- `count`: number of rows to generate (required).
- `purpose`: what the dataset evaluates — task description, target capability, input domain, and the system prompt the model should run under.
- `expected`: how the `expected` column should be populated — exact-label set (e.g. `positive|negative|neutral`), regex-normalizable answer, free text, or `none` if every row is graded by an LLM judge instead.
- `expected_json`: for JSON or tool-call evals, the full canonical expected JSON object. Required for structured-output adaptation datasets.
- `assessment`: how `expected` is compared against the model output (see [Assessment modes](#assessment-modes)). Optional — defaults to `exact_normalized` when `expected` is set.
- `grader`: LLM-judge instructions to write into the `grader` column — leave blank when `expected` + `assessment` are sufficient. Required when `assessment=llm_judge` or when a row has no `expected`.
- `output`: destination CSV path. Default `data/practical_evals/<task_slug>.csv`.

## Output schema

Always emit a CSV with this header order:

```
id,prompt,expected,expected_json,assessment,strict_fields,soft_fields,grader,system,task_type
```

Rules:

- `id`: 1-indexed integer, unique per row.
- `prompt`: the full task prompt — self-contained, no external context required, varied across rows so the dataset exercises edge cases (not just trivial happy-path examples).
- `expected`: the deterministic answer the assessor compares against. Leave blank only for rows that must be LLM-judged.
- `expected_json`: the full expected JSON object for structured-output rows. Use compact valid JSON. Leave blank for non-JSON tasks.
- `assessment`: programmatic mode (see below). Leave blank to take the default — `exact_normalized` when `expected` is set, `llm_judge` when only `grader` is set.
- `strict_fields`: comma-separated JSON paths to evaluate exactly when `expected_json` is set, such as `name,arguments.to`. Leave blank to score all expected JSON leaf fields.
- `soft_fields`: comma-separated JSON paths present in `expected_json` but intentionally excluded from exact scoring, such as `arguments.subject,arguments.body`.
- `grader`: LLM-judge rubric. Populate only for rows that route to the judge — same form as [llm-judge.md](llm-judge.md) expects.
- `system`: the system prompt the model under test should run with. Keep it consistent within a dataset unless the spec requires variation.
- `task_type`: short slug matching the dataset's purpose (e.g. `sentiment`, `routing`, `summarization`).

You may add task-specific metadata columns after `task_type` (e.g. `segment`, `route_set`, `difficulty`) when they help slice results — match the conventions used in [data/practical_evals/](../../data/practical_evals/).

## Assessment modes

Programmatic scoring is the default whenever `expected` is set — the judge command is **not** invoked for those rows, even if one is configured. The judge only runs for rows where `expected` is blank, or where `assessment` explicitly equals `llm_judge`.

| `assessment` value | Behavior |
| --- | --- |
| _(blank)_ | Defaults to `exact_normalized` if `expected` is set, else `llm_judge` if `grader` is set, else `ungraded`. |
| `exact` / `exact_raw` | Byte-for-byte equality after stripping leading/trailing whitespace. |
| `exact_normalized` / `normalized` / `label` / `classification` / `routing` | Lowercased, whitespace-collapsed equality. Use for sentiment, classification, routing. |
| `contains` | Pass when the normalized `expected` value appears anywhere in the normalized prediction. |
| `regex` | Treat `expected` as a regex; pass when it matches the prediction (`re.IGNORECASE | re.MULTILINE`). |
| `json_field:path.to.field` | Parse prediction as JSON, walk the dotted path, compare the value to `expected` with normalized equality. Path may also be supplied via metadata column `json_field`. |
| `json_match` / `tool_call_json` | Parse prediction and `expected_json` as JSON, then compare `strict_fields`. If `strict_fields` is blank, compare all expected JSON leaf fields. Exclude paths in `soft_fields`. |
| `llm_judge` | Force LLM-judge scoring even when `expected` is set — use sparingly, for rows where the deterministic answer is only a hint. |

Choose `assessment` per row based on what you wrote in `expected`. Mixed datasets are fine: most rows can use `exact_normalized` while a handful set `assessment=llm_judge` for open-ended cases.

## Structured JSON and Tool-Call Evals

For tool-call or JSON instruction-following datasets, do **not** use only a scalar field target like:

```csv
expected,assessment
billing-team@vendor.com,json_field:arguments.to
```

That is acceptable for simple pass/fail scoring, but it is under-specified for adaptive probe generation. The patch search needs the full expected structure so it can target the first wrong decision boundary.

Instead, provide the full expected call:

```csv
expected,expected_json,assessment,strict_fields,soft_fields
billing-team@vendor.com,"{""name"":""send_email"",""arguments"":{""to"":""billing-team@vendor.com"",""subject"":""Invoice discrepancy"",""body"":""Please investigate the invoice discrepancy.""}}",json_match,"name,arguments.to","arguments.subject,arguments.body"
```

For the standard tool-call schema:

```text
search_orders(customer_email, status)
refund_order(order_id, amount, reason)
update_shipping_address(order_id, street, city, zip)
cancel_subscription(subscription_id, effective_date)
send_email(to, subject, body)
escalate_ticket(ticket_id, priority, team)
```

Recommended strict fields:

| Tool | `strict_fields` | `soft_fields` |
| --- | --- | --- |
| `search_orders` | `name,arguments.customer_email,arguments.status` | _(blank)_ |
| `refund_order` | `name,arguments.order_id,arguments.amount` | `arguments.reason` |
| `update_shipping_address` | `name,arguments.order_id,arguments.street,arguments.city,arguments.zip` | _(blank)_ |
| `cancel_subscription` | `name,arguments.subscription_id,arguments.effective_date` | _(blank)_ |
| `send_email` | `name,arguments.to` | `arguments.subject,arguments.body` |
| `escalate_ticket` | `name,arguments.ticket_id,arguments.priority,arguments.team` | _(blank)_ |

Keep the scalar `expected` column as the headline value for UI readability, but make `expected_json` the source of truth for structured scoring and dynamic probes. This lets the adaptive probe builder compare expected-vs-actual JSON and generate probes for:

- tool/function name boundaries
- argument key boundaries
- argument value boundaries

without hardcoding a specific dataset's tool names or field mappings.

## Generation guidance

- Cover the answer space evenly. For classification, balance labels. For routing, hit every queue. For graded free-text, vary length, tone, and difficulty.
- Include adversarial / ambiguous rows (roughly 10–20% of the set) so the eval can distinguish weak models from strong ones. These are good candidates for `assessment=llm_judge` with a written `grader` rubric.
- Keep prompts realistic — phrase them like user requests, not textbook exercises.
- Escape commas and quotes correctly. Wrap any field containing `,`, `"`, or newlines in double quotes and double any embedded `"`.
- Do not duplicate prompts.

## After writing the CSV

1. Print the output path and row count.
2. Show the first 3 rows for a sanity check.
3. Suggest the next command to normalize it into JSONL:

```bash
PYTHONPATH=src .venv/bin/python -m bankai_poc.cli eval-dataset \
  --csv-source <output> \
  --task-name <task_slug> \
  --output data/practical_evals/<task_slug>.jsonl
```

## Example invocation

> Generate 25 rows. Purpose: classify customer support emails by urgency as `low`, `medium`, or `high`. System prompt: "You triage support emails. Return only one label." Expected: exact label match. No grader. Output: `data/practical_evals/urgency_triage.csv`.

Produces a CSV with 25 balanced, varied rows where `expected` is one of `low|medium|high`, `expected_json` is blank, `assessment` is `exact_normalized` (or blank), and `grader` is empty.

> Generate 12 rows. Purpose: summarize customer issues in one sentence. System prompt: "You summarize customer support issues accurately and concisely." No fixed expected. Grader: pass if one sentence, mentions the core issue, no invented details.

Produces 12 rows with blank `expected`, `assessment=llm_judge` (or blank), and a per-row `grader` rubric.

> Generate 70 rows. Purpose: select the correct business tool call from the provided tool schema. System prompt: "You are a function-calling model..." Expected: full JSON tool call. Assessment: `json_match`.

Produces rows with compact `expected_json`, `assessment=json_match`, strict structural fields for tool names and critical arguments, and soft fields for open-ended email text.

## Related

- [practical-eval-dataset.md](practical-eval-dataset.md) — normalize an existing CSV into the JSONL format the runner consumes.
- [practical-eval-run.md](practical-eval-run.md) — run the three-way pipeline against the dataset.
- [llm-judge.md](llm-judge.md) — judge-command contract for rows that route to the judge.
