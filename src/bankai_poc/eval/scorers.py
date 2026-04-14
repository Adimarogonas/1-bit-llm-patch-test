from __future__ import annotations

import json
import re
from typing import Any


_NUMBER_RE = r"-?\d+(?:\.\d+)?"


def extract_last_number(text: str) -> str | None:
    matches = re.findall(_NUMBER_RE, text.replace(",", ""))
    return matches[-1] if matches else None


def extract_gsm8k_answer(text: str) -> str | None:
    cleaned = (text or "").replace(",", "")
    patterns = [
        r"(?i)final answer\s*[:=]\s*(" + _NUMBER_RE + r")",
        r"(?i)answer\s*[:=]\s*(" + _NUMBER_RE + r")",
        r"(?i)the answer is\s*(" + _NUMBER_RE + r")",
        r"####\s*(" + _NUMBER_RE + r")",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return extract_last_number(cleaned)


def score_gsm8k(prediction: str, reference: str) -> bool:
    return extract_gsm8k_answer(prediction or "") == extract_gsm8k_answer(reference or "")


def gsm8k_output_diagnostics(prediction: str) -> dict[str, Any]:
    text = prediction or ""
    stripped = text.rstrip()
    return {
        "extracted": extract_gsm8k_answer(text),
        "has_final_answer_marker": bool(re.search(r"(?i)final answer\s*[:=]|####", text)),
        "has_answer_phrase": bool(re.search(r"(?i)the answer is|answer\s*[:=]", text)),
        "has_think_block": "<think>" in text or "</think>" in text,
        "likely_truncated": bool(stripped) and stripped[-1] not in ".!?\n>" and extract_gsm8k_answer(text) is None,
    }


def score_humaneval_plus(prediction: str, row: dict[str, Any]) -> dict[str, Any]:
    try:
        from evalplus.evaluate import evaluate  # type: ignore

        return {
            "score": None,
            "passed": None,
            "mode": "evalplus_available_but_not_wired",
            "note": "EvalPlus is installed, but this scaffold does not yet build official sample files.",
        }
    except Exception:
        entry_point = (row.get("metadata") or {}).get("entry_point") or ""
        passed = bool(entry_point and f"def {entry_point}" in prediction)
        return {
            "score": float(passed),
            "passed": passed,
            "mode": "heuristic",
            "note": "Fallback heuristic checks whether the expected entry point is defined.",
        }


def score_ifeval(prediction: str, row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata", {})
    instruction_ids = meta.get("instruction_ids", [])
    passed = []
    if "punctuation:no_comma" in instruction_ids:
        passed.append("," not in prediction)
    if "detectable_format:number_bullets" in instruction_ids:
        bullets = len(re.findall(r"(?m)^\s*[-*]\s+", prediction))
        target = _first_present(meta.get("kwargs", []), "num_bullets")
        if target is not None:
            passed.append(bullets == int(target))
    if "length_constraints:number_words" in instruction_ids:
        words = len(re.findall(r"\b\w+\b", prediction))
        target = _first_present(meta.get("kwargs", []), "num_words")
        if target is not None:
            passed.append(words >= int(target))
    final = all(passed) if passed else False
    return {"score": float(final), "passed": final, "checks": passed, "mode": "heuristic"}


def _first_present(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def score_bfcl(prediction: str, row: dict[str, Any]) -> dict[str, Any]:
    target_tool = (row.get("metadata") or {}).get("target_tool")
    parsed = parse_tool_prediction(prediction)
    tool_match = parsed.get("tool_name") == target_tool if target_tool else False
    return {
        "score": float(tool_match),
        "passed": tool_match,
        "tool_name": parsed.get("tool_name"),
        "mode": "heuristic",
    }


def parse_tool_prediction(prediction: str) -> dict[str, Any]:
    try:
        payload = json.loads(prediction)
        if isinstance(payload, dict):
            if "name" in payload:
                return {"tool_name": payload.get("name"), "arguments": payload.get("arguments")}
            if "tool_name" in payload:
                return {"tool_name": payload.get("tool_name"), "arguments": payload.get("arguments")}
    except Exception:
        pass

    match = re.search(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(", prediction)
    if match:
        return {"tool_name": match.group(1), "arguments": None}
    return {"tool_name": None, "arguments": None}
