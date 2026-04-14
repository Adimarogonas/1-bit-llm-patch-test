from __future__ import annotations

import hashlib
from typing import Any

from bankai_poc.data.registry import data_dir, probes_path
from bankai_poc.utils.io import read_jsonl, write_jsonl

from bankai_poc.eval.items import build_eval_items


def _last_token(text: str) -> str:
    tokens = text.strip().split()
    return tokens[-1] if tokens else ""


def _stable_partition(name: str) -> str:
    value = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % 10
    if value == 0:
        return "control"
    if value == 1:
        return "validation"
    return "search"


def _gsm8k_prompt(question: str, rationale: str) -> str:
    cleaned_rationale = rationale.strip()
    if cleaned_rationale:
        assistant_prefix = f"{cleaned_rationale}\n\nFinal answer: "
    else:
        assistant_prefix = "Final answer: "
    return (
        "<|im_start|>system\n"
        "You are a careful math tutor. Solve the user's grade-school math problem step by step. "
        "End with a single line in the exact format: Final answer: <number><|im_end|>\n"
        "<|im_start|>user\n"
        "Solve this grade-school math problem. Show concise reasoning, then end with the required final line.\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


def _gsm8k_wrong_answers(correct: str) -> list[str]:
    candidates: list[str] = []
    try:
        value = int(correct.replace(",", ""))
    except ValueError:
        return ["0"]

    def add(item: int) -> None:
        token = str(item)
        if token != correct and token not in candidates:
            candidates.append(token)

    add(value + 1)
    add(value - 1)
    add(value * 2)
    if value % 2 == 0:
        add(value // 2)
    if abs(value) >= 10:
        try:
            add(int(str(abs(value))[:-1]) if abs(value) >= 10 else value)
        except ValueError:
            pass
    if value != 0:
        add(0)
    if abs(value) < 100000:
        add(value * 10)
    return candidates[:4] or ["0"]


def _gsm8k_probe(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = dict(row["metadata"])
    correct = str(metadata.get("answer_value") or _last_token(str(row["reference"])))
    rationale = str(metadata.get("rationale") or "")
    prompt = _gsm8k_prompt(row["prompt"], rationale)
    partition = _stable_partition(str(row["id"]))
    probes = []
    for idx, wrong in enumerate(_gsm8k_wrong_answers(correct), start=1):
        probes.append(
            {
                "benchmark": "gsm8k",
                "probe_type": "final_answer_pair",
                "name": f"{row['id']}::wrong{idx}",
                "prompt": prompt,
                "correct_token": correct,
                "wrong_token": wrong,
                "metadata": {
                    **metadata,
                    "partition": partition,
                    "wrong_token": wrong,
                },
            }
        )
    return probes


def _humaneval_probe(row: dict[str, Any]) -> dict[str, Any]:
    first_line = row["reference"].splitlines()[0] if row["reference"] else "pass"
    return {
        "benchmark": "humaneval_plus",
        "probe_type": "completion_pair",
        "name": str(row["id"]),
        "prompt": row["prompt"],
        "correct_token": first_line,
        "wrong_token": "raise NotImplementedError",
        "metadata": row["metadata"],
    }


def _ifeval_probe(row: dict[str, Any]) -> dict[str, Any]:
    instruction_ids = row["metadata"].get("instruction_ids", [])
    positive = "|".join(str(item) for item in instruction_ids[:2]) or "follow"
    return {
        "benchmark": "ifeval",
        "probe_type": "compliance_pair",
        "name": str(row["id"]),
        "prompt": row["prompt"],
        "correct_token": positive,
        "wrong_token": "ignore",
        "metadata": row["metadata"],
    }


def _bfcl_probe(row: dict[str, Any]) -> dict[str, Any]:
    target = row["reference"]
    positive = target if isinstance(target, str) else str(target)
    return {
        "benchmark": "bfcl",
        "probe_type": "tool_choice_pair",
        "name": str(row["id"]),
        "prompt": row["prompt"],
        "correct_token": positive[:120],
        "wrong_token": "NO_TOOL",
        "metadata": row["metadata"],
    }


GENERATORS = {
    "gsm8k": _gsm8k_probe,
    "humaneval_plus": _humaneval_probe,
    "ifeval": _ifeval_probe,
    "bfcl": _bfcl_probe,
}


def generate_probes(benchmark: str) -> None:
    normalized = read_jsonl(data_dir(benchmark) / "normalized.jsonl")
    probes = []
    for row in normalized:
        generated = GENERATORS[benchmark](row)
        if isinstance(generated, list):
            probes.extend(generated)
        else:
            probes.append(generated)
    write_jsonl(probes_path(benchmark), probes)
    build_eval_items(benchmark)
