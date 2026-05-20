from __future__ import annotations

import json
from pathlib import Path

from bankai_poc.eval.dynamic_probes import build_dynamic_probes_from_details
from bankai_poc.eval.pipeline import (
    CsvEvalDatasetConfig,
    JudgeConfig,
    ModelConfig,
    PipelineConfig,
    build_csv_eval_dataset,
    run_programmatic_assessment,
    run_pipeline,
)
from bankai_poc.search.probe_eval import compute_fitness_min
from bankai_poc.utils.io import load_json, write_jsonl


def test_terminal_pipeline_writes_summary_and_details(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(
        dataset_path,
        [
            {
                "benchmark": "gsm8k",
                "id": "case-1",
                "prompt": "What is 2+2?",
                "reference": "4",
                "metadata": {},
            }
        ],
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=dataset_path,
            models=[ModelConfig(name="echo-agent", kind="terminal", command="printf 'Final answer: 4'")],
            output_dir=tmp_path / "run",
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["model"] == "echo-agent"
    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["score"]["passed"] is True


def test_csv_practical_eval_scores_expected_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "sentiment.csv"
    csv_path.write_text(
        "id,prompt,expected,grader,system\n"
        "case-1,Return only the label positive.,positive,,Return one sentiment label.\n",
        encoding="utf-8",
    )
    dataset_path = build_csv_eval_dataset(
        CsvEvalDatasetConfig(
            source_path=csv_path,
            task_name="sentiment",
            output_path=tmp_path / "sentiment.jsonl",
            shuffle=False,
        )
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=dataset_path,
            models=[ModelConfig(name="echo-agent", kind="terminal", command="printf positive")],
            output_dir=tmp_path / "run_csv",
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["benchmark"] == "all"
    assert summary["summary"][0]["scored_examples"] == 1
    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["task"] == "sentiment"


def test_llm_judge_scores_grader_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "judge.csv"
    csv_path.write_text(
        "id,prompt,expected,grader\n"
        "case-1,Summarize this issue.,,Pass if concise.\n",
        encoding="utf-8",
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[ModelConfig(name="answer-agent", kind="terminal", command="printf concise")],
            output_dir=tmp_path / "run_judge",
            judge=JudgeConfig(command="printf '{\"passed\": true, \"score\": 1, \"reason\": \"ok\"}'"),
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["score"]["mode"] == "llm_judge"


def test_expected_column_uses_programmatic_scoring_even_with_judge(tmp_path: Path) -> None:
    csv_path = tmp_path / "routing.csv"
    csv_path.write_text(
        "id,prompt,expected,grader,assessment\n"
        "case-1,Route this to billing.,billing,Use the expected route.,routing\n",
        encoding="utf-8",
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[ModelConfig(name="route-agent", kind="terminal", command="printf billing")],
            output_dir=tmp_path / "run_programmatic",
            judge=JudgeConfig(command="printf '{\"passed\": false, \"score\": 0, \"reason\": \"should not run\"}'"),
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["score"]["mode"] == "routing"


def test_json_field_programmatic_assessment(tmp_path: Path) -> None:
    csv_path = tmp_path / "json_field.csv"
    csv_path.write_text(
        "id,prompt,expected,assessment\n"
        "case-1,Return JSON with route field.,billing,json_field:route\n",
        encoding="utf-8",
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[ModelConfig(name="json-agent", kind="terminal", command="printf '{\"route\":\"billing\"}'")],
            output_dir=tmp_path / "run_json_field",
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["score"]["mode"] == "json_field"


def test_json_field_scoring_handles_trailing_text_and_truncation(tmp_path: Path) -> None:
    csv_path = tmp_path / "json_field.csv"
    csv_path.write_text(
        "id,prompt,expected,assessment\n"
        "case-1,Return JSON with email field.,enterprise@bigco.com,json_field:arguments.to\n",
        encoding="utf-8",
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[
                ModelConfig(
                    name="json-agent",
                    kind="terminal",
                    command=(
                        "printf '{\"name\":\"send_email\",\"arguments\":{\"to\":\"enterprise@bigco.com\","
                        "\"body\":\"long unfinished'"
                    ),
                )
            ],
            output_dir=tmp_path / "run_json_prefix",
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["score"]["actual"] == "enterprise@bigco.com"


def test_expected_json_scores_full_tool_call(tmp_path: Path) -> None:
    csv_path = tmp_path / "tool_call_json.csv"
    csv_path.write_text(
        'id,prompt,expected_json,assessment\n'
        'case-1,Email billing.,'
        '"{""name"":""send_email"",""arguments"":{""to"":""billing-team@vendor.com""}}",json_match\n',
        encoding="utf-8",
    )

    summary_path, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[
                ModelConfig(
                    name="json-agent",
                    kind="terminal",
                    command='printf \'{"name":"send_email","arguments":{"to":"billing-team@vendor.com"}}\'',
                )
            ],
            output_dir=tmp_path / "run_expected_json",
        )
    )

    summary = load_json(summary_path)
    details = load_json(details_path)

    assert summary["summary"][0]["accuracy"] == 1.0
    assert details["rows"][0]["expected_json"] == '{"name":"send_email","arguments":{"to":"billing-team@vendor.com"}}'
    assert details["rows"][0]["score"]["mode"] == "json_match"
    assert details["rows"][0]["score"]["passed"] is True


def test_expected_json_detects_wrong_tool_and_argument_key(tmp_path: Path) -> None:
    csv_path = tmp_path / "tool_call_json.csv"
    csv_path.write_text(
        'id,prompt,expected_json,assessment\n'
        'case-1,Email billing.,'
        '"{""name"":""send_email"",""arguments"":{""to"":""billing-team@vendor.com""}}",json_match\n',
        encoding="utf-8",
    )

    _, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[
                ModelConfig(
                    name="json-agent",
                    kind="terminal",
                    command='printf \'{"name":"escalate_ticket","arguments":{"team":"billing-team@vendor.com"}}\'',
                )
            ],
            output_dir=tmp_path / "run_expected_json_fail",
        )
    )

    details = load_json(details_path)
    score = details["rows"][0]["score"]

    assert score["passed"] is False
    mismatch_paths = {mismatch["path"] for mismatch in score["mismatches"]}
    assert "name" in mismatch_paths
    assert "arguments.to" in mismatch_paths
    assert "arguments" in mismatch_paths


def test_expected_json_scores_truncated_output_when_strict_fields_match(tmp_path: Path) -> None:
    csv_path = tmp_path / "tool_call_json.csv"
    csv_path.write_text(
        'id,prompt,expected_json,assessment,strict_fields,soft_fields\n'
        'case-1,Email Lisa.,'
        '"{""name"":""send_email"",""arguments"":{""to"":""lisa.park@retail.io"",""subject"":""Thanks"",""body"":""Thanks.""}}",'
        'json_match,"name,arguments.to","arguments.subject,arguments.body"\n',
        encoding="utf-8",
    )

    _, details_path = run_pipeline(
        PipelineConfig(
            dataset_path=csv_path,
            models=[
                ModelConfig(
                    name="json-agent",
                    kind="terminal",
                    command=(
                        "printf '{\"name\":\"send_email\",\"arguments\":{\"to\":\"lisa.park@retail.io\","
                        "\"subject\":\"Thanks\",\"body\":\"Long body\"}'"
                    ),
                )
            ],
            output_dir=tmp_path / "run_expected_json_truncated",
        )
    )

    details = load_json(details_path)
    score = details["rows"][0]["score"]

    assert score["passed"] is True
    assert score["reason"] == "ok_truncated_json_fields"
    assert score["actual_fields"]["arguments.to"] == "lisa.park@retail.io"


def test_regex_assessment_accepts_common_answer_formatting() -> None:
    row = {"id": "case-1"}

    assert run_programmatic_assessment(
        row,
        "### **Final Answer:**\n**90**",
        r"final answer:\s*90(?:\.0+)?(?![\d.])",
        "regex",
    )["passed"] is True
    assert run_programmatic_assessment(
        row,
        "work\nFinal answer: 90.",
        r"final answer:\s*90(?:\.0+)?(?![\d.])",
        "regex",
    )["passed"] is True
    assert run_programmatic_assessment(
        row,
        "Final answer: 91.",
        r"final answer:\s*90(?:\.0+)?(?![\d.])",
        "regex",
    )["passed"] is False


def test_numeric_final_assessment_compares_final_answer_number() -> None:
    row = {"id": "case-1"}

    score = run_programmatic_assessment(
        row,
        "Work shown here.\n### **Final Answer:**\n**$1,234.50.**",
        "$1,234.5",
        "numeric_final",
    )
    assert score["passed"] is True
    assert score["actual"] == 1234.5
    assert score["expected_numeric"] == 1234.5

    assert run_programmatic_assessment(
        row,
        "Final answer: 6.481",
        "6.480",
        "numeric_final:0.01",
    )["passed"] is True
    assert run_programmatic_assessment(
        row,
        "Final answer: 91.",
        "90",
        "numeric_final",
    )["passed"] is False


def test_dynamic_probes_use_failures_as_targets_and_passes_as_controls(tmp_path: Path) -> None:
    dataset_path = tmp_path / "routing.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment\n"
        "fail-1,Route to billing.,billing,routing\n"
        "pass-1,Route to sales.,sales,routing\n",
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "routing",
      "benchmark": "routing",
      "id": "fail-1",
      "prompt": "Route to billing.",
      "expected": "billing",
      "prediction": "sales",
      "score": {"passed": false}
    },
    {
      "model": "base-model",
      "task": "routing",
      "benchmark": "routing",
      "id": "pass-1",
      "prompt": "Route to sales.",
      "expected": "sales",
      "prediction": "sales",
      "score": {"passed": true}
    }
  ]
}
""",
        encoding="utf-8",
    )
    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(probes) == 2
    assert probes[0]["metadata"]["partition"] == "search"
    assert probes[0]["correct_completion"] == "billing"
    assert probes[0]["wrong_completion"] == "sales"
    assert probes[1]["metadata"]["partition"] == "control"


def test_dynamic_regex_probe_uses_representative_answer_not_pattern(tmp_path: Path) -> None:
    dataset_path = tmp_path / "math.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment\n"
        "case-1,What is 45 plus 45?,final answer:\\s*90(?:\\.0+)?(?![\\d.]),regex\n"
        "case-2,What is 20 plus 20?,final answer:\\s*40(?:\\.0+)?(?![\\d.]),regex\n",
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "math",
      "benchmark": "math",
      "id": "case-1",
      "prompt": "What is 45 plus 45?",
      "expected": "final answer:\\\\s*90(?:\\\\.0+)?(?![\\\\d.])",
      "prediction": "Final answer: 91.",
      "score": {"assessment": "regex", "passed": false}
    },
    {
      "model": "base-model",
      "task": "math",
      "benchmark": "math",
      "id": "case-2",
      "prompt": "What is 20 plus 20?",
      "expected": "final answer:\\\\s*40(?:\\\\.0+)?(?![\\\\d.])",
      "prediction": "Final answer: 40",
      "score": {"assessment": "regex", "expected": "final answer:\\\\s*40(?:\\\\.0+)?(?![\\\\d.])", "passed": true}
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]
    probe = next(probe for probe in probes if probe["metadata"]["partition"] == "search")
    control_probe = next(probe for probe in probes if probe["metadata"]["partition"] == "control")
    raw_probe_text = probes_path.read_text(encoding="utf-8")

    assert probe["metadata"]["target_kind"] == "regex_label"
    assert probe["correct_completion"] == "final answer: 90"
    assert probe["wrong_completion"] == "91"
    assert control_probe["correct_completion"] == "final answer: 40"
    assert control_probe["metadata"]["base_score"]["expected"] == "final answer: 40"
    assert "\\\\s*" not in raw_probe_text
    assert "(?!" not in raw_probe_text


def test_dynamic_numeric_final_probe_uses_canonical_numbers(tmp_path: Path) -> None:
    dataset_path = tmp_path / "math.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment\n"
        'case-1,What is the total?,"$1,234.50",numeric_final\n'
        "case-2,What is half of ten?,5.0,numeric_final\n",
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "math",
      "benchmark": "math",
      "id": "case-1",
      "prompt": "What is the total?",
      "expected": "$1,234.50",
      "prediction": "Final answer: $1235.",
      "score": {"assessment": "numeric_final", "passed": false}
    },
    {
      "model": "base-model",
      "task": "math",
      "benchmark": "math",
      "id": "case-2",
      "prompt": "What is half of ten?",
      "expected": "5.0",
      "prediction": "Final answer: 5",
      "score": {"assessment": "numeric_final", "passed": true}
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "numeric_probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]
    search_probe = next(probe for probe in probes if probe["metadata"]["partition"] == "search")
    control_probe = next(probe for probe in probes if probe["metadata"]["partition"] == "control")

    assert search_probe["metadata"]["target_kind"] == "numeric_final"
    assert search_probe["correct_completion"] == "1234.5"
    assert search_probe["wrong_completion"] == "1235"
    assert control_probe["correct_completion"] == "5"


def test_dynamic_json_field_probe_targets_wrong_tool_before_argument(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tool_call.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment,system\n"
        'case-1,Email ops-team@vendor.co about the delayed shipment.,ops-team@vendor.co,json_field:arguments.to,Return JSON.\n',
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "tool_call",
      "benchmark": "tool_call",
      "id": "case-1",
      "prompt": "Email ops-team@vendor.co about the delayed shipment.",
      "expected": "ops-team@vendor.co",
      "prediction": "{\\"name\\": \\"escalate_ticket\\", \\"arguments\\": {\\"team\\": \\"ops-team@vendor.co\\"}}",
      "score": {
        "assessment": "json_field:arguments.to",
        "passed": false
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]
    probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_tool_name")

    assert probe["prompt"].endswith('{"name": "')
    assert probe["correct_completion"] == "send_email"
    assert probe["wrong_completion"] == "escalate_ticket"
    assert probe["metadata"]["target_kind"] == "json_tool_name"
    key_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_argument_key.to")
    assert key_probe["prompt"].endswith('{"name": "send_email", "arguments": {"')
    assert key_probe["correct_completion"] == "to"
    assert key_probe["wrong_completion"] == "team"


def test_dynamic_json_field_probe_targets_argument_after_correct_tool(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tool_call.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment,system\n"
        'case-1,Email ops-team@vendor.co about the delayed shipment.,ops-team@vendor.co,json_field:arguments.to,Return JSON.\n',
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "tool_call",
      "benchmark": "tool_call",
      "id": "case-1",
      "prompt": "Email ops-team@vendor.co about the delayed shipment.",
      "expected": "ops-team@vendor.co",
      "prediction": "{\\"name\\": \\"send_email\\", \\"arguments\\": {\\"to\\": \\"urgent\\"}}",
      "score": {
        "assessment": "json_field:arguments.to",
        "actual": "urgent",
        "passed": false
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probe = json.loads(probes_path.read_text(encoding="utf-8").splitlines()[0])

    assert probe["prompt"].endswith('{"name": "send_email", "arguments": {"to": "')
    assert probe["correct_completion"] == "ops-team@vendor.co"
    assert probe["wrong_completion"] == "urgent"
    assert probe["metadata"]["target_kind"] == "json_argument.to"


def test_dynamic_json_control_preserves_tool_name_boundary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tool_call.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment,system\n"
        'case-1,Find cancelled orders for billing-test@shop.com.,cancelled,json_field:arguments.status,Return JSON.\n',
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "tool_call",
      "benchmark": "tool_call",
      "id": "case-1",
      "prompt": "Find cancelled orders for billing-test@shop.com.",
      "expected": "cancelled",
      "prediction": "{\\"name\\": \\"search_orders\\", \\"arguments\\": {\\"customer_email\\": \\"billing-test@shop.com\\", \\"status\\": \\"cancelled\\"}}",
      "score": {
        "assessment": "json_field:arguments.status",
        "actual": "cancelled",
        "passed": true
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]

    tool_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_tool_name_control")
    value_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_argument_control.status")

    assert tool_probe["prompt"].endswith('{"name": "')
    assert tool_probe["correct_completion"] == "search_orders"
    assert tool_probe["wrong_completion"] == "cancel_subscription"
    assert value_probe["prompt"].endswith('{"name": "search_orders", "arguments": {"status": "')
    assert value_probe["correct_completion"] == "cancelled"


def test_dynamic_json_control_penalizes_send_email_to_notify(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tool_call.csv"
    dataset_path.write_text(
        "id,prompt,expected,assessment,system\n"
        'case-1,Email customer-success@retailco.io.,send_email,json_field:name,Return JSON.\n',
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "tool_call",
      "benchmark": "tool_call",
      "id": "case-1",
      "prompt": "Email customer-success@retailco.io.",
      "expected": "send_email",
      "prediction": "{\\"name\\": \\"send_email\\", \\"arguments\\": {\\"to\\": \\"customer-success@retailco.io\\"}}",
      "score": {
        "assessment": "json_field:name",
        "actual": "send_email",
        "passed": true
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]
    tool_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_tool_name_control")

    assert tool_probe["correct_completion"] == "send_email"
    assert tool_probe["wrong_completion"] == "notify"


def test_dynamic_expected_json_probe_uses_structured_diff_without_known_tool_names(tmp_path: Path) -> None:
    dataset_path = tmp_path / "custom_tool_call.csv"
    dataset_path.write_text(
        'id,prompt,expected_json,assessment,system\n'
        'case-1,Page ops at vendor.,'
        '"{""name"":""page_user"",""arguments"":{""recipient"":""ops-team@vendor.co""}}",json_match,Return JSON.\n',
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "custom_tool_call",
      "benchmark": "custom_tool_call",
      "id": "case-1",
      "prompt": "Page ops at vendor.",
      "expected": "",
      "expected_json": "{\\"name\\":\\"page_user\\",\\"arguments\\":{\\"recipient\\":\\"ops-team@vendor.co\\"}}",
      "prediction": "{\\"name\\": \\"assign_user\\", \\"arguments\\": {\\"team\\": \\"ops-team@vendor.co\\"}}",
      "score": {
        "assessment": "json_match",
        "passed": false
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]

    tool_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_path.name")
    key_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_path.arguments.key")

    assert tool_probe["correct_completion"] == "page_user"
    assert tool_probe["wrong_completion"] == "assign_user"
    assert key_probe["prompt"].endswith('{"name": "page_user", "arguments": {')
    assert key_probe["correct_completion"] == "recipient"
    assert key_probe["wrong_completion"] == "team"


def test_dynamic_expected_json_controls_use_path_specific_rivals(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tool_call.csv"
    dataset_path.write_text(
        'id,prompt,expected_json,assessment,strict_fields,soft_fields,system\n'
        'case-1,Escalate urgent ticket.,'
        '"{""name"":""escalate_ticket"",""arguments"":{""ticket_id"":""TKT-1"",""priority"":""urgent"",""team"":""security""}}",'
        'json_match,"name,arguments.ticket_id,arguments.priority,arguments.team",,Return JSON.\n'
        'case-2,Escalate high ticket.,'
        '"{""name"":""escalate_ticket"",""arguments"":{""ticket_id"":""TKT-2"",""priority"":""high"",""team"":""network""}}",'
        'json_match,"name,arguments.ticket_id,arguments.priority,arguments.team",,Return JSON.\n'
        'case-3,Email ops.,'
        '"{""name"":""send_email"",""arguments"":{""to"":""ops@example.com"",""subject"":""Ops"",""body"":""Ping ops.""}}",'
        'json_match,"name,arguments.to","arguments.subject,arguments.body",Return JSON.\n',
        encoding="utf-8",
    )
    details_path = tmp_path / "details.json"
    details_path.write_text(
        """
{
  "rows": [
    {
      "model": "base-model",
      "task": "tool_call",
      "benchmark": "tool_call",
      "id": "case-1",
      "prompt": "Escalate urgent ticket.",
      "expected_json": "{\\"name\\":\\"escalate_ticket\\",\\"arguments\\":{\\"ticket_id\\":\\"TKT-1\\",\\"priority\\":\\"urgent\\",\\"team\\":\\"security\\"}}",
      "prediction": "{\\"name\\":\\"escalate_ticket\\",\\"arguments\\":{\\"ticket_id\\":\\"TKT-1\\",\\"priority\\":\\"urgent\\",\\"team\\":\\"security\\"}}",
      "score": {"assessment": "json_match", "passed": true}
    }
  ]
}
""",
        encoding="utf-8",
    )

    probes_path = build_dynamic_probes_from_details(details_path, dataset_path, tmp_path / "probes.jsonl")
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines()]

    name_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_path.name_control")
    priority_probe = next(probe for probe in probes if probe["metadata"]["target_kind"] == "json_path.arguments.priority_control")

    assert name_probe["correct_completion"] == "escalate_ticket"
    assert name_probe["wrong_completion"] == "send_email"
    assert priority_probe["correct_completion"] == "urgent"
    assert priority_probe["wrong_completion"] == "high"


def test_min_fitness_uses_worst_target_probe() -> None:
    fitness = compute_fitness_min(
        target_gaps={"easy": 2.0, "hard": 0.3},
        control_gaps={"control": 1.0},
        target_baseline={"easy": 1.0, "hard": 0.2},
        control_baseline={"control": 1.0},
        control_penalty=2.0,
    )

    assert round(fitness, 6) == 0.1
