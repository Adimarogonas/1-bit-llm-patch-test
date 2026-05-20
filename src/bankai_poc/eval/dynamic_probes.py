from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from bankai_poc.eval.pipeline import (
    extract_final_number,
    extract_json_field_value,
    normalize_label,
    normalize_regex_prediction,
    parse_json_object_prefix,
    read_eval_rows,
    regex_representative_completion,
)
from bankai_poc.utils.io import load_json, write_jsonl

TOOL_NAMES = [
    "search_orders",
    "refund_order",
    "update_shipping_address",
    "cancel_subscription",
    "send_email",
    "escalate_ticket",
    "notify",
]

NUMERIC_ASSESSMENTS = {"numeric_final", "final_numeric", "numeric"}


def build_dynamic_probes_from_details(
    details_path: Path,
    dataset_path: Path,
    output_path: Path,
    model_name: str | None = None,
    max_target: int = 32,
    max_control: int = 16,
) -> Path:
    details = load_json(details_path)["rows"]
    source_rows = {str(row["id"]): row for row in read_eval_rows(dataset_path)}
    label_set = _label_set(source_rows.values())
    json_context = _json_probe_context(source_rows.values())
    model_rows = [row for row in details if model_name is None or row["model"] == model_name]
    targets = [row for row in model_rows if row.get("score", {}).get("passed") is False and _has_expected_target(row)]
    controls = [row for row in model_rows if row.get("score", {}).get("passed") is True and _has_expected_target(row)]
    selected_controls = _select_diverse_rows(controls, source_rows, max_control)
    selected_control_ids = {str(row.get("id")) for row in selected_controls}
    validation_candidates = [row for row in controls if str(row.get("id")) not in selected_control_ids]
    selected_validation = _select_diverse_rows(validation_candidates, source_rows, max(1, max_control // 2))

    probes: list[dict[str, Any]] = []
    for row in targets[:max_target]:
        source = source_rows.get(str(row["id"]), {})
        probes.extend(_probes_for_row(row, source, label_set, json_context, "search"))
    for row in selected_controls:
        source = source_rows.get(str(row["id"]), {})
        probes.extend(_probes_for_row(row, source, label_set, json_context, "control"))
    for row in selected_validation:
        source = source_rows.get(str(row["id"]), {})
        probes.extend(_probes_for_row(row, source, label_set, json_context, "validation"))

    if not [probe for probe in probes if probe["metadata"]["partition"] == "search"]:
        # If the base model has no failures, use the hardest available controls as a
        # conservative target set. This creates a no-op-ish patch search instead of
        # failing the adaptation pipeline.
        for row in controls[: min(max_target, len(controls))]:
            source = source_rows.get(str(row["id"]), {})
            probes.extend(_probes_for_row(row, source, label_set, json_context, "search"))

    write_jsonl(output_path, probes)
    return output_path


def _select_diverse_rows(rows: list[dict[str, Any]], source_rows: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source = source_rows.get(str(row.get("id")), {})
        bucket = _row_diversity_key(row, source)
        buckets.setdefault(bucket, []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            bucket_rows = buckets[key]
            if not bucket_rows:
                continue
            selected.append(bucket_rows.pop(0))
            if len(selected) >= limit:
                break
    return selected


def _row_diversity_key(row: dict[str, Any], source: dict[str, Any]) -> str:
    payload = _expected_payload(row) or _expected_payload(source)
    if payload:
        name = _json_path_get(payload, "name")
        if name:
            return f"name:{name}"
    assessment = (row.get("score") or {}).get("assessment") or source.get("assessment") or ""
    return f"assessment:{assessment or row.get('task') or row.get('benchmark') or 'unknown'}"


def _label_set(rows: Any) -> list[str]:
    labels: list[str] = []
    for row in rows:
        expected = str(row.get("expected") or row.get("reference") or "").strip()
        assessment = str(row.get("assessment") or "").strip().lower()
        if expected and assessment == "regex":
            expected = regex_representative_completion(expected)
        elif expected and _assessment_mode(assessment) in NUMERIC_ASSESSMENTS:
            expected = _numeric_completion(expected) or expected
        if expected and expected not in labels:
            labels.append(expected)
        expected_payload = _expected_payload(row)
        if expected_payload:
            for _, value in _json_leaf_items(expected_payload):
                text = str(value).strip()
                if text and text not in labels:
                    labels.append(text)
    return labels


def _has_expected_target(row: dict[str, Any]) -> bool:
    return bool(row.get("expected") or row.get("reference") or row.get("expected_json") or row.get("metadata", {}).get("expected_json"))


def _json_probe_context(rows: Any) -> dict[str, Any]:
    values_by_path: dict[str, list[str]] = {}
    argument_keys: list[str] = []
    for row in rows:
        payload = _expected_payload(row)
        if not payload:
            continue
        for path, value in _json_leaf_items(payload):
            text = str(value).strip()
            if text and text not in values_by_path.setdefault(path, []):
                values_by_path[path].append(text)
        arguments = payload.get("arguments")
        if isinstance(arguments, dict):
            for key in arguments:
                if key not in argument_keys:
                    argument_keys.append(str(key))
    return {"values_by_path": values_by_path, "argument_keys": argument_keys}


def _probes_for_row(
    row: dict[str, Any],
    source: dict[str, Any],
    label_set: list[str],
    json_context: dict[str, Any],
    partition: str,
) -> list[dict[str, Any]]:
    expected = str(row.get("expected") or row.get("reference") or "").strip()
    prediction = str(row.get("prediction") or "")
    assessment = (row.get("score") or {}).get("assessment") or (source or {}).get("assessment") or ""
    if assessment == "regex":
        expected = regex_representative_completion(expected)
    elif _assessment_mode(assessment) in NUMERIC_ASSESSMENTS:
        expected = _numeric_completion(expected) or expected
    source = source or row
    expected_payload = _expected_payload(row) or _expected_payload(source)
    if expected_payload:
        targets = _json_structured_targets(source, row, prediction, expected_payload, label_set, json_context, partition)
        if targets:
            return [
                _make_probe(row, prompt, correct, wrong, partition, target_kind, prediction, label_set)
                for prompt, correct, wrong, target_kind in targets
            ]

    if partition in {"control", "validation"} and assessment.startswith("json_field:"):
        targets = _json_control_targets(source, row, prediction, expected, assessment.split(":", 1)[1])
        if targets:
            return [
                _make_probe(row, prompt, correct, wrong, partition, target_kind, prediction, label_set)
                for prompt, correct, wrong, target_kind in targets
            ]
    if partition == "search" and assessment.startswith("json_field:"):
        targets = _json_search_targets(source, row, prediction, expected, assessment.split(":", 1)[1])
        if targets:
            return [
                _make_probe(row, prompt, correct, wrong, partition, target_kind, prediction, label_set)
                for prompt, correct, wrong, target_kind in targets
            ]

    prompt, correct, wrong, target_kind = _probe_target(source, row, prediction, label_set, expected, assessment)
    return [_make_probe(row, prompt, correct, wrong, partition, target_kind, prediction, label_set)]


def _make_probe(
    row: dict[str, Any],
    prompt: str,
    correct: str,
    wrong: str,
    partition: str,
    target_kind: str,
    prediction: str,
    label_set: list[str],
) -> dict[str, Any]:
    return {
        "benchmark": row.get("task") or row.get("benchmark") or "practical_eval",
        "probe_type": "dynamic_practical_pair",
        "name": f"{row.get('id')}::{partition}::{target_kind}",
        "prompt": prompt,
        "correct_token": correct,
        "wrong_token": wrong,
        "correct_completion": correct,
        "wrong_completion": wrong,
        "metadata": {
            "partition": partition,
            "source_row_id": row.get("id"),
            "base_prediction": prediction,
            "base_score": _probe_base_score(row, target_kind, correct),
            "actual_wrong_value": wrong,
            "target_kind": target_kind,
            "label_set": label_set,
        },
    }


def _probe_base_score(row: dict[str, Any], target_kind: str, correct: str) -> dict[str, Any]:
    score = dict(row.get("score", {}))
    if target_kind == "regex_label" and "expected" in score:
        score["expected"] = correct
    return score


def _json_control_targets(
    source: dict[str, Any],
    row: dict[str, Any],
    prediction: str,
    expected: str,
    field: str,
) -> list[tuple[str, str, str, str]]:
    actual_name = extract_json_field_value(prediction, "name")
    if actual_name is None:
        return []

    correct_tool = str(actual_name)
    targets = [
        (
            _probe_prompt(source, '{"name": "'),
            correct_tool,
            _rival_tool_name(correct_tool, source, expected, field),
            "json_tool_name_control",
        )
    ]

    if field == "name":
        return targets

    if field.startswith("arguments."):
        argument_name = field.split(".", 1)[1]
        actual_value = extract_json_field_value(prediction, field)
        correct_value = str(actual_value) if actual_value is not None else expected
        wrong_value = _control_wrong_value(correct_value, expected, field)
        targets.append(
            (
                _probe_prompt(source, f'{{"name": "{correct_tool}", "arguments": {{"{argument_name}": "'),
                correct_value,
                wrong_value,
                f"json_argument_control.{argument_name}",
            )
        )

    return targets


def _json_search_targets(
    source: dict[str, Any],
    row: dict[str, Any],
    prediction: str,
    expected: str,
    field: str,
) -> list[tuple[str, str, str, str]]:
    if field == "name":
        prompt, correct, wrong, target_kind = _json_probe_target(source, row, prediction, expected, field) or (
            _probe_prompt(source, '{"name": "'),
            expected,
            _fallback_wrong_label([expected, *TOOL_NAMES], expected),
            "json_tool_name",
        )
        return [(prompt, correct, wrong, target_kind)]

    if not field.startswith("arguments."):
        prompt, correct, wrong, target_kind = _json_probe_target(source, row, prediction, expected, field) or (
            _probe_prompt(source, ""),
            expected,
            _fallback_wrong_label([expected, *TOOL_NAMES], expected),
            f"json_field.{field}",
        )
        return [(prompt, correct, wrong, target_kind)]

    argument_name = field.split(".", 1)[1]
    actual_name = extract_json_field_value(prediction, "name")
    correct_tool = _tool_name_for_field(field, parse_json_object_prefix(prediction), expected)
    targets: list[tuple[str, str, str, str]] = []

    if actual_name is not None and correct_tool and str(actual_name) != correct_tool:
        targets.append((_probe_prompt(source, '{"name": "'), correct_tool, str(actual_name), "json_tool_name"))

    wrong_argument_name = _wrong_argument_name(prediction, expected, argument_name)
    if wrong_argument_name and wrong_argument_name != argument_name:
        targets.append(
            (
                _probe_prompt(source, f'{{"name": "{correct_tool}", "arguments": {{"'),
                argument_name,
                wrong_argument_name,
                f"json_argument_key.{argument_name}",
            )
        )

    actual_value = (row.get("score") or {}).get("actual")
    if actual_value is None:
        actual_value = extract_json_field_value(prediction, field)
    wrong_value = str(actual_value) if actual_value is not None and str(actual_value) != expected else ""
    if not wrong_value:
        wrong_value = _value_from_wrong_argument(prediction, wrong_argument_name, expected) or _fallback_wrong_label(
            [expected, "urgent", "low", "search_orders"],
            expected,
        )
    targets.append(
        (
            _probe_prompt(source, f'{{"name": "{correct_tool}", "arguments": {{"{argument_name}": "'),
            expected,
            wrong_value,
            f"json_argument.{argument_name}",
        )
    )
    return targets


def _json_structured_targets(
    source: dict[str, Any],
    row: dict[str, Any],
    prediction: str,
    expected_payload: dict[str, Any],
    label_set: list[str],
    json_context: dict[str, Any],
    partition: str,
) -> list[tuple[str, str, str, str]]:
    actual_payload = parse_json_object_prefix(prediction) or {}
    targets: list[tuple[str, str, str, str]] = []
    soft_fields = _metadata_field_set(row, source, "soft_fields")
    strict_fields = _metadata_field_set(row, source, "strict_fields")
    values_by_path: dict[str, list[str]] = json_context.get("values_by_path", {})
    argument_keys: list[str] = json_context.get("argument_keys", [])

    expected_name = _json_path_get(expected_payload, "name")
    actual_name = _json_path_get(actual_payload, "name")
    if isinstance(expected_name, str):
        if partition in {"control", "validation"}:
            wrong_name = _first_other_value(values_by_path.get("name", []), expected_name)
            if wrong_name:
                targets.append((_probe_prompt(source, '{"name": "'), expected_name, wrong_name, "json_path.name_control"))
        elif actual_name != expected_name:
            wrong_name = str(actual_name) if actual_name is not None else _first_other_value(values_by_path.get("name", []), expected_name)
            if wrong_name:
                targets.append((_probe_prompt(source, '{"name": "'), expected_name, wrong_name, "json_path.name"))

    expected_arguments = expected_payload.get("arguments")
    actual_arguments = actual_payload.get("arguments") if isinstance(actual_payload, dict) else None
    if isinstance(expected_arguments, dict):
        actual_arguments = actual_arguments if isinstance(actual_arguments, dict) else {}
        expected_keys = [
            str(key)
            for key in expected_arguments
            if f"arguments.{key}" not in soft_fields and (not strict_fields or f"arguments.{key}" in strict_fields)
        ]
        if partition in {"control", "validation"}:
            wrong_key = _wrong_json_argument_key(expected_arguments, actual_arguments) or _first_other_value(argument_keys, expected_keys[0] if expected_keys else "")
        else:
            wrong_key = _wrong_json_argument_key(expected_arguments, actual_arguments)
        if expected_keys and wrong_key:
            targets.append(
                (
                    _probe_prompt(source, f'{{"name": "{expected_name or actual_name or ""}", "arguments": {{'),
                    expected_keys[0] if partition in {"control", "validation"} else _first_missing_or_changed_key(expected_arguments, actual_arguments),
                    wrong_key,
                    "json_path.arguments.key_control" if partition in {"control", "validation"} else "json_path.arguments.key",
                )
            )

    for path, expected_value in _json_leaf_items(expected_payload):
        if path == "name":
            continue
        if path in soft_fields or (strict_fields and path not in strict_fields):
            continue
        actual_value = _json_path_get(actual_payload, path)
        if partition in {"control", "validation"}:
            wrong_value = _first_other_value(values_by_path.get(path, []), str(expected_value))
        elif _json_values_equal(expected_value, actual_value):
            continue
        else:
            wrong_value = str(actual_value) if actual_value is not None else _first_other_value(values_by_path.get(path, []), str(expected_value))
        if not wrong_value:
            continue
        prefix = _json_prompt_prefix_for_path(expected_payload, path)
        if prefix:
            targets.append(
                (
                    _probe_prompt(source, prefix),
                    str(expected_value),
                    wrong_value,
                    f"json_path.{path}_control" if partition in {"control", "validation"} else f"json_path.{path}",
                )
            )

    return targets


def _metadata_field_set(row: dict[str, Any], source: dict[str, Any], key: str) -> set[str]:
    value = row.get(key) or row.get("metadata", {}).get(key) or source.get(key) or source.get("metadata", {}).get(key) or ""
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _expected_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    expected_json = str(row.get("expected_json") or row.get("metadata", {}).get("expected_json") or "").strip()
    return parse_json_object_prefix(expected_json) if expected_json else None


def _json_leaf_items(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(payload, dict):
        items: list[tuple[str, Any]] = []
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_json_leaf_items(value, child_prefix))
        return items
    return [(prefix, payload)]


def _json_path_get(payload: Any, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _json_values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) or isinstance(actual, str):
        return normalize_label(str(expected)) == normalize_label(str(actual))
    return expected == actual


def _first_different_json_value(label_set: list[str], expected: str) -> str:
    for label in label_set:
        if label != expected:
            return label
    return _fallback_wrong_label(label_set, expected)


def _first_other_value(values: list[str], expected: str) -> str:
    for value in values:
        if value != expected:
            return value
    return ""


def _first_missing_or_changed_key(expected_arguments: dict[str, Any], actual_arguments: dict[str, Any]) -> str:
    for key, value in expected_arguments.items():
        if key not in actual_arguments or not _json_values_equal(value, actual_arguments.get(key)):
            return str(key)
    return str(next(iter(expected_arguments), ""))


def _wrong_json_argument_key(expected_arguments: dict[str, Any], actual_arguments: dict[str, Any]) -> str:
    expected_keys = set(expected_arguments)
    for actual_key, actual_value in actual_arguments.items():
        if actual_key in expected_keys:
            continue
        for expected_value in expected_arguments.values():
            if _json_values_equal(expected_value, actual_value):
                return str(actual_key)
    for actual_key in actual_arguments:
        if actual_key not in expected_keys:
            return str(actual_key)
    return ""


def _json_prompt_prefix_for_path(expected_payload: dict[str, Any], path: str) -> str:
    if path.startswith("arguments."):
        expected_name = _json_path_get(expected_payload, "name")
        argument_name = path.split(".", 1)[1]
        if "." in argument_name:
            return ""
        return f'{{"name": "{expected_name or ""}", "arguments": {{"{argument_name}": "'
    if "." not in path:
        return f'{{"{path}": "'
    return ""


def _probe_target(
    source: dict[str, Any],
    row: dict[str, Any],
    prediction: str,
    label_set: list[str],
    expected: str,
    assessment: str,
) -> tuple[str, str, str, str]:
    if assessment.startswith("json_field:"):
        field = assessment.split(":", 1)[1]
        target = _json_probe_target(source, row, prediction, expected, field)
        if target is not None:
            return target

    if assessment == "regex":
        wrong = _regex_wrong_value(prediction, expected) or _fallback_wrong_label(label_set, expected)
        return _probe_prompt(source, ""), expected, wrong, "regex_label"
    if _assessment_mode(assessment) in NUMERIC_ASSESSMENTS:
        wrong = _numeric_wrong_value(prediction, expected) or _fallback_wrong_label(label_set, expected)
        return _probe_prompt(source, ""), expected, wrong, "numeric_final"

    wrong = _wrong_value_for_row(row, prediction, label_set, expected, assessment) or _fallback_wrong_label(label_set, expected)
    return _probe_prompt(source, ""), expected, wrong, "label"


def _json_probe_target(
    source: dict[str, Any],
    row: dict[str, Any],
    prediction: str,
    expected: str,
    field: str,
) -> tuple[str, str, str, str] | None:
    actual_name = extract_json_field_value(prediction, "name")
    if field == "name":
        wrong_name = str(actual_name) if actual_name is not None and str(actual_name) != expected else ""
        wrong = wrong_name or _fallback_wrong_label([expected, "search_orders", "send_email", "escalate_ticket"], expected)
        return _probe_prompt(source, '{"name": "'), expected, wrong, "json_tool_name"

    if field.startswith("arguments."):
        correct_tool = _tool_name_for_field(field, parse_json_object_prefix(prediction), expected)
        if actual_name is not None and correct_tool and str(actual_name) != correct_tool:
            return _probe_prompt(source, '{"name": "'), correct_tool, str(actual_name), "json_tool_name"

        argument_name = field.split(".", 1)[1]
        actual_value = (row.get("score") or {}).get("actual")
        if actual_value is None:
            actual_value = extract_json_field_value(prediction, field)
        wrong = str(actual_value) if actual_value is not None and str(actual_value) != expected else ""
        if not wrong:
            wrong = _fallback_wrong_label([expected, "urgent", "low", "search_orders"], expected)
        prefix = f'{{"name": "{correct_tool}", "arguments": {{"{argument_name}": "'
        return _probe_prompt(source, prefix), expected, wrong, f"json_argument.{argument_name}"

    actual_value = extract_json_field_value(prediction, field)
    wrong = str(actual_value) if actual_value is not None and str(actual_value) != expected else ""
    if not wrong:
        return None
    return _probe_prompt(source, f'{{"{field}": "'), expected, wrong, f"json_field.{field}"


def _probe_prompt(row: dict[str, Any], assistant_prefix: str = "") -> str:
    system = row.get("system") or "Complete the user's practical evaluation task. Return only the requested answer."
    prompt = row["prompt"]
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


def _tool_name_for_field(field: str, payload: dict[str, Any] | None, expected: str) -> str:
    if field == "name":
        return ""
    if field.startswith("arguments."):
        argument_name = field.split(".", 1)[1]
        if argument_name == "status":
            return "search_orders"
        if argument_name == "to":
            return "send_email"
        if argument_name == "priority":
            return "escalate_ticket"
        if argument_name in {"order_id", "amount", "reason"}:
            return "refund_order"
        if argument_name in {"street", "city", "zip"}:
            return "update_shipping_address"
        if argument_name in {"subscription_id", "effective_date"}:
            return "cancel_subscription"
    if payload and isinstance(payload.get("name"), str):
        return payload["name"]
    return "send_email" if "@" in expected else ""


def _rival_tool_name(correct_tool: str, source: dict[str, Any], expected: str, field: str) -> str:
    prompt = f"{source.get('prompt', '')} {expected} {field}".lower()
    if correct_tool == "search_orders" and "cancel" in prompt:
        return "cancel_subscription"
    if correct_tool == "send_email":
        return "notify"
    if correct_tool == "escalate_ticket":
        return "send_email"
    if correct_tool == "cancel_subscription":
        return "search_orders"
    for tool_name in TOOL_NAMES:
        if tool_name != correct_tool:
            return tool_name
    return f"not_{correct_tool}"


def _wrong_argument_name(prediction: str, expected: str, correct_argument_name: str) -> str | None:
    payload = parse_json_object_prefix(prediction)
    arguments = payload.get("arguments") if isinstance(payload, dict) else None
    if not isinstance(arguments, dict):
        return None
    for key, value in arguments.items():
        if key != correct_argument_name and str(value) == expected:
            return str(key)
    if correct_argument_name == "to" and "team" in arguments:
        return "team"
    return None


def _value_from_wrong_argument(prediction: str, wrong_argument_name: str | None, expected: str) -> str | None:
    if not wrong_argument_name:
        return None
    value = extract_json_field_value(prediction, f"arguments.{wrong_argument_name}")
    if value is None:
        return None
    text = str(value)
    return text if text != expected else wrong_argument_name


def _control_wrong_value(correct_value: str, expected: str, field: str) -> str:
    if field == "arguments.status":
        for status in ["cancelled", "pending", "delivered", "shipped"]:
            if status != correct_value:
                return status
    if field == "arguments.priority":
        for priority in ["urgent", "high", "medium", "low"]:
            if priority != correct_value:
                return priority
    if "email" in field or "to" in field:
        return "urgent" if correct_value != "urgent" else "send_email"
    if expected and expected != correct_value:
        return expected
    return f"not_{correct_value}_{hashlib.sha256(correct_value.encode('utf-8')).hexdigest()[:6]}"


def _wrong_value_for_row(
    row: dict[str, Any],
    prediction: str,
    label_set: list[str],
    expected: str,
    assessment: str,
) -> str | None:
    if assessment == "regex":
        return _regex_wrong_value(prediction, expected)
    if _assessment_mode(assessment) in NUMERIC_ASSESSMENTS:
        return _numeric_wrong_value(prediction, expected)
    if assessment.startswith("json_field:"):
        field = assessment.split(":", 1)[1]
        value = (row.get("score") or {}).get("actual")
        if value is None:
            value = extract_json_field_value(prediction, field)
        if value is not None and str(value) != expected:
            return str(value)
    return _predicted_label(prediction, label_set, expected)


def _assessment_mode(assessment: str) -> str:
    return (assessment or "").partition(":")[0].strip().lower()


def _numeric_completion(text: str) -> str | None:
    number = extract_final_number(text)
    return f"{number:g}" if number is not None else None


def _numeric_wrong_value(prediction: str, expected: str) -> str | None:
    actual = _numeric_completion(prediction)
    expected_number = _numeric_completion(expected) or expected
    if actual and normalize_label(actual) != normalize_label(expected_number):
        return actual
    return None


def _regex_wrong_value(prediction: str, expected: str) -> str | None:
    text = normalize_regex_prediction(prediction)
    answer = _extract_answer_label_value(text)
    expected_answer = _extract_answer_label_value(expected) or expected
    if (
        answer
        and normalize_label(answer) != normalize_label(expected)
        and normalize_label(answer) != normalize_label(expected_answer)
    ):
        return answer
    return None


def _extract_answer_label_value(prediction: str) -> str | None:
    lines = [line.strip() for line in prediction.splitlines()]
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        match = re.search(r"\b(?:final\s+)?answer\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_answer_value(match.group(1))
        if value:
            return value
        for next_line in lines[index + 1 :]:
            value = _clean_answer_value(next_line)
            if value:
                return value
    return None


def _clean_answer_value(value: str) -> str:
    text = re.sub(r"[*_`]+", "", value or "").strip()
    text = re.sub(r"^[#:\-\s]+", "", text).strip()
    text = re.sub(r"(?<=\d)\.(?=\s*$)", "", text).strip()
    return text


def _predicted_label(prediction: str, label_set: list[str], expected: str) -> str | None:
    normalized_prediction = normalize_label(prediction)
    for label in label_set:
        if label == expected:
            continue
        normalized_label = normalize_label(label)
        if normalized_prediction == normalized_label or normalized_label in normalized_prediction.split():
            return label
    for label in label_set:
        if label != expected and normalize_label(label) in normalized_prediction:
            return label
    return None


def _fallback_wrong_label(label_set: list[str], expected: str) -> str:
    for label in label_set:
        if label != expected:
            return label
    digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:6]
    return f"not_{expected}_{digest}"
