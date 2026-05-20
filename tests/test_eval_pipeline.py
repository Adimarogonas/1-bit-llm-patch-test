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


def test_min_fitness_uses_worst_target_probe() -> None:
    fitness = compute_fitness_min(
        target_gaps={"easy": 2.0, "hard": 0.3},
        control_gaps={"control": 1.0},
        target_baseline={"easy": 1.0, "hard": 0.2},
        control_baseline={"control": 1.0},
        control_penalty=2.0,
    )

    assert round(fitness, 6) == 0.1
