from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bankai_poc.utils.io import load_json, load_yaml, write_jsonl

from .registry import config_path, data_dir


def _normalize_gsm8k(row: dict[str, Any]) -> dict[str, Any]:
    full_answer = row["answer"]
    rationale = full_answer
    answer = full_answer
    if "####" in full_answer:
        rationale, answer = full_answer.split("####", 1)
    answer = answer.strip()
    rationale = rationale.strip()
    answer_value = None
    cleaned = answer.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match:
        answer_value = match.group(0)
    return {
        "benchmark": "gsm8k",
        "id": row.get("question"),
        "prompt": row["question"],
        "reference": answer,
        "metadata": {
            "full_answer": full_answer,
            "rationale": rationale,
            "answer_value": answer_value,
        },
    }


def _normalize_humaneval_plus(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row.get("prompt") or row.get("question") or row.get("instruction", "")
    reference = row.get("canonical_solution") or row.get("solution") or ""
    return {
        "benchmark": "humaneval_plus",
        "id": row.get("task_id") or row.get("name"),
        "prompt": prompt,
        "reference": reference,
        "metadata": {"entry_point": row.get("entry_point"), "test": row.get("test")},
    }


def _normalize_ifeval(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "ifeval",
        "id": row.get("key"),
        "prompt": row["prompt"],
        "reference": "",
        "metadata": {
            "instruction_ids": row.get("instruction_id_list", []),
            "kwargs": row.get("kwargs", []),
        },
    }


def _flatten_messages(turns: list[Any]) -> str:
    messages: list[str] = []
    for turn in turns:
        if isinstance(turn, list):
            for message in turn:
                if isinstance(message, dict):
                    messages.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        elif isinstance(turn, dict):
            messages.append(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
    return "\n".join(messages)


def _extract_tool_name(ground_truth: Any) -> str | None:
    if isinstance(ground_truth, list) and ground_truth:
        first = ground_truth[0]
        if isinstance(first, dict) and first:
            return next(iter(first.keys()))
    if isinstance(ground_truth, dict) and ground_truth:
        return next(iter(ground_truth.keys()))
    if isinstance(ground_truth, str):
        return ground_truth
    return None


def _load_bfcl_raw_rows(raw_dir: Path) -> list[dict[str, Any]]:
    possible_answers: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for file_path in sorted(raw_dir.rglob("*.json")):
        rel = file_path.relative_to(raw_dir).as_posix()
        if rel.startswith("BFCL_v3_"):
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
        elif rel.startswith("possible_answer/"):
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        possible_answers[row["id"]] = row.get("ground_truth")
    for row in records:
        row["ground_truth"] = possible_answers.get(row.get("id"))
    return records


def _normalize_bfcl(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("question") or row.get("prompt") or row.get("user_query") or ""
    prompt = _flatten_messages(question) if isinstance(question, list) else str(question)
    target_tool = _extract_tool_name(row.get("ground_truth"))
    return {
        "benchmark": "bfcl",
        "id": row.get("id") or row.get("question_id"),
        "prompt": prompt,
        "reference": target_tool or "",
        "metadata": {
            "tools": row.get("function", row.get("functions", [])),
            "target_tool": target_tool,
            "ground_truth": row.get("ground_truth"),
        },
    }


NORMALIZERS = {
    "gsm8k": _normalize_gsm8k,
    "humaneval_plus": _normalize_humaneval_plus,
    "ifeval": _normalize_ifeval,
    "bfcl": _normalize_bfcl,
}


def normalize_benchmark(benchmark: str) -> None:
    config = load_yaml(config_path(benchmark))
    source = config["source"]
    raw_dir = data_dir(benchmark) / "raw"
    if source["kind"] == "huggingface_dataset":
        raw = load_json(raw_dir / "dataset.json")
    elif source["kind"] == "huggingface_json_files":
        raw = _load_bfcl_raw_rows(raw_dir)
    else:
        raise ValueError(f"Unsupported source kind: {source['kind']}")
    rows = [NORMALIZERS[benchmark](row) for row in raw]
    write_jsonl(data_dir(benchmark) / "normalized.jsonl", rows)
