from __future__ import annotations

import dataclasses
import csv
import json
import random
import re
import shlex
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from bankai_poc.data.registry import BENCHMARKS, data_dir, root_dir
from bankai_poc.eval.scorers import (
    extract_gsm8k_answer,
    gsm8k_output_diagnostics,
    score_bfcl,
    score_gsm8k,
    score_humaneval_plus,
    score_ifeval,
)
from bankai_poc.utils.io import dump_json, read_jsonl, write_jsonl


LogFn = Callable[[str], None]


@dataclasses.dataclass
class EvalDatasetConfig:
    benchmarks: list[str]
    limit_per_benchmark: int = 20
    seed: int = 7
    shuffle: bool = True
    output_path: Path | None = None


@dataclasses.dataclass
class CsvEvalDatasetConfig:
    source_path: Path
    task_name: str = "practical_eval"
    prompt_column: str = "prompt"
    expected_column: str = "expected"
    expected_json_column: str = "expected_json"
    grader_column: str = "grader"
    id_column: str = "id"
    system_column: str = "system"
    assessment_column: str = "assessment"
    limit: int | None = None
    seed: int = 7
    shuffle: bool = True
    output_path: Path | None = None


@dataclasses.dataclass
class ModelConfig:
    name: str
    kind: str
    enabled: bool = True
    model_ref: str = ""
    patch_path: Path | None = None
    command: str = ""
    timeout_s: int = 180


@dataclasses.dataclass
class JudgeConfig:
    mode: str = "auto"
    command: str = ""
    timeout_s: int = 180


@dataclasses.dataclass
class PipelineConfig:
    dataset_path: Path
    models: list[ModelConfig]
    max_tokens: int = 160
    output_dir: Path | None = None
    judge: JudgeConfig = dataclasses.field(default_factory=JudgeConfig)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_eval_runs_dir() -> Path:
    return root_dir() / "results" / "eval_runs"


def default_eval_datasets_dir() -> Path:
    return root_dir() / "data" / "practical_evals"


def build_eval_dataset(config: EvalDatasetConfig) -> Path:
    rng = random.Random(config.seed)
    rows: list[dict[str, Any]] = []
    for benchmark in config.benchmarks:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        source_path = data_dir(benchmark) / "normalized.jsonl"
        source_rows = read_jsonl(source_path)
        if config.shuffle:
            source_rows = source_rows[:]
            rng.shuffle(source_rows)
        for row in source_rows[: config.limit_per_benchmark]:
            rows.append(
                {
                    "benchmark": benchmark,
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "reference": row.get("reference", ""),
                    "metadata": row.get("metadata", {}),
                }
            )
    output_path = config.output_path or default_eval_runs_dir() / f"dataset_{timestamp_id()}.jsonl"
    write_jsonl(output_path, rows)
    return output_path


def build_csv_eval_dataset(config: CsvEvalDatasetConfig) -> Path:
    rows = read_csv_eval_rows(
        config.source_path,
        task_name=config.task_name,
        prompt_column=config.prompt_column,
        expected_column=config.expected_column,
        expected_json_column=config.expected_json_column,
        grader_column=config.grader_column,
        id_column=config.id_column,
        system_column=config.system_column,
        assessment_column=config.assessment_column,
    )
    rng = random.Random(config.seed)
    if config.shuffle:
        rng.shuffle(rows)
    if config.limit is not None:
        rows = rows[: config.limit]
    output_path = config.output_path or default_eval_datasets_dir() / f"{config.task_name}_{timestamp_id()}.jsonl"
    write_jsonl(output_path, rows)
    return output_path


def read_eval_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv_eval_rows(path)
    return read_jsonl(path)


def read_csv_eval_rows(
    path: Path,
    task_name: str | None = None,
    prompt_column: str = "prompt",
    expected_column: str = "expected",
    expected_json_column: str = "expected_json",
    grader_column: str = "grader",
    id_column: str = "id",
    system_column: str = "system",
    assessment_column: str = "assessment",
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        grader_column = _resolve_optional_column(reader.fieldnames, grader_column, ["llm_grader", "llm grader"])
        _require_column(reader.fieldnames, prompt_column, path)
        has_expected = expected_column in reader.fieldnames
        has_expected_json = expected_json_column in reader.fieldnames
        has_grader = grader_column in reader.fieldnames
        if not has_expected and not has_expected_json and not has_grader:
            raise ValueError(
                f"{path} needs at least an '{expected_column}', '{expected_json_column}', or '{grader_column}' column"
            )
        rows = []
        for index, raw in enumerate(reader, start=1):
            prompt = (raw.get(prompt_column) or "").strip()
            if not prompt:
                continue
            expected = (raw.get(expected_column) or "").strip() if has_expected else ""
            expected_json = (raw.get(expected_json_column) or "").strip() if has_expected_json else ""
            grader = (raw.get(grader_column) or "").strip() if has_grader else ""
            assessment = (raw.get(assessment_column) or "").strip() if assessment_column in reader.fieldnames else ""
            if not assessment:
                assessment = "json_match" if expected_json else "exact_normalized" if expected else "llm_judge" if grader else "ungraded"
            rows.append(
                {
                    "task": task_name or path.stem,
                    "benchmark": task_name or path.stem,
                    "id": (raw.get(id_column) or str(index)).strip() if id_column in reader.fieldnames else str(index),
                    "prompt": prompt,
                    "expected": expected,
                    "expected_json": expected_json,
                    "reference": expected,
                    "grader": grader,
                    "assessment": assessment,
                    "system": (raw.get(system_column) or "").strip() if system_column in reader.fieldnames else "",
                    "metadata": {
                        key: value
                        for key, value in raw.items()
                        if key
                        not in {
                            prompt_column,
                            expected_column,
                            expected_json_column,
                            grader_column,
                            id_column,
                            system_column,
                            assessment_column,
                        }
                    },
                }
            )
    return rows


def _resolve_optional_column(fieldnames: list[str], requested: str, aliases: list[str]) -> str:
    if requested in fieldnames:
        return requested
    for alias in aliases:
        if alias in fieldnames:
            return alias
    return requested


def _require_column(fieldnames: list[str], column: str, path: Path) -> None:
    if column not in fieldnames:
        raise ValueError(f"{path} is missing required column '{column}'")


def score_prediction(row: dict[str, Any], prediction: str, judge: JudgeConfig | None = None) -> dict[str, Any]:
    if _is_practical_row(row):
        return score_practical_prediction(row, prediction, judge or JudgeConfig())

    benchmark = row["benchmark"]
    if benchmark == "gsm8k":
        passed = score_gsm8k(prediction, row.get("reference", ""))
        return {
            "score": float(passed),
            "passed": passed,
            "mode": "exact_match",
            "extracted": extract_gsm8k_answer(prediction),
            "diagnostics": gsm8k_output_diagnostics(prediction),
        }
    if benchmark == "humaneval_plus":
        return score_humaneval_plus(prediction, row)
    if benchmark == "ifeval":
        return score_ifeval(prediction, row)
    if benchmark == "bfcl":
        return score_bfcl(prediction, row)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _is_practical_row(row: dict[str, Any]) -> bool:
    benchmark = row.get("benchmark")
    return benchmark not in BENCHMARKS or "expected" in row or "grader" in row


def score_practical_prediction(row: dict[str, Any], prediction: str, judge: JudgeConfig) -> dict[str, Any]:
    expected = str(row.get("expected") or row.get("reference") or "").strip()
    expected_json = str(row.get("expected_json") or row.get("metadata", {}).get("expected_json") or "").strip()
    grader = str(row.get("grader") or "").strip()
    assessment = str(row.get("assessment") or "").strip() or (
        "json_match" if expected_json else "exact_normalized" if expected else "llm_judge" if grader else "ungraded"
    )

    if (expected or expected_json) and assessment != "llm_judge":
        return run_programmatic_assessment(row, prediction, expected, assessment)
    if grader and judge.command.strip():
        return run_llm_judge(row, prediction, judge)
    if grader:
        return {
            "score": None,
            "passed": None,
            "mode": "needs_llm_judge",
            "grader": grader,
            "note": "This row has grader instructions but no judge command was configured.",
        }
    return {"score": None, "passed": None, "mode": "ungraded"}


def run_programmatic_assessment(row: dict[str, Any], prediction: str, expected: str, assessment: str) -> dict[str, Any]:
    mode, _, arg = assessment.partition(":")
    mode = (mode or "exact_normalized").strip().lower()
    arg = arg.strip()

    if mode in {"exact", "exact_raw"}:
        passed = prediction.strip() == expected.strip()
    elif mode in {"exact_normalized", "normalized", "label", "classification", "routing"}:
        passed = normalize_label(prediction) == normalize_label(expected)
    elif mode == "contains":
        passed = normalize_label(expected) in normalize_label(prediction)
    elif mode == "regex":
        passed = regex_prediction_matches(expected, prediction)
    elif mode in {"numeric_final", "final_numeric", "numeric"}:
        tolerance = _parse_numeric_tolerance(arg)
        expected_number = extract_final_number(expected)
        actual = extract_final_number(prediction)
        passed = (
            expected_number is not None
            and actual is not None
            and abs(actual - expected_number) <= tolerance
        )
    elif mode == "json_field":
        field_name = arg or str(row.get("metadata", {}).get("json_field", ""))
        actual = extract_json_field_value(prediction, field_name)
        passed = normalize_label(str(actual)) == normalize_label(expected) if actual is not None else False
    elif mode in {"json", "json_match", "json_exact", "tool_call", "tool_call_json"}:
        result = score_expected_json(prediction, row)
        return {
            "score": float(result["passed"]),
            "passed": result["passed"],
            "mode": "json_match",
            "expected": expected,
            "assessment": assessment,
            **result,
        }
    else:
        passed = normalize_label(prediction) == normalize_label(expected)
        mode = "exact_normalized"

    return {
        "score": float(passed),
        "passed": passed,
        "mode": mode,
        "expected": expected,
        "assessment": assessment,
        **({"actual": actual} if mode in {"json_field", "numeric_final", "final_numeric", "numeric"} else {}),
        **({"expected_numeric": expected_number, "tolerance": tolerance} if mode in {"numeric_final", "final_numeric", "numeric"} else {}),
    }


def regex_prediction_matches(pattern: str, prediction: str) -> bool:
    return _regex_search(pattern, prediction) or _regex_search(pattern, normalize_regex_prediction(prediction))


def normalize_regex_prediction(prediction: str) -> str:
    text = prediction or ""
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"(?<=\d)\.(?=\s*(?:$|\n))", "", text)
    return text


def regex_representative_completion(pattern: str) -> str:
    text = pattern or ""
    text = re.sub(r"\(\?[:=!<][^)]*\)", "", text)
    text = re.sub(r"\(\?:[^)]*\)\?", "", text)
    text = re.sub(r"\[[^\]]*\][*+?]?", " ", text)
    text = re.sub(r"\\s[*+?]?", " ", text)
    text = re.sub(r"\\[dDwWsS][*+?]?", " ", text)
    text = re.sub(r"\\([\\.^$|?*+()[\]{}])", r"\1", text)
    text = re.sub(r"[$^]", "", text)
    text = re.sub(r"[()?+*{}|]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or pattern


def _regex_search(pattern: str, prediction: str) -> bool:
    try:
        return re.search(pattern, prediction, flags=re.IGNORECASE | re.MULTILINE) is not None
    except re.error:
        return False


def extract_final_number(text: str) -> float | None:
    normalized = _normalize_numeric_text(text)
    answer_matches = list(
        re.finditer(r"\b(?:final\s+)?answer\s*:\s*([^\n]*)", normalized, flags=re.IGNORECASE)
    )
    for match in reversed(answer_matches):
        number = _last_number(match.group(1))
        if number is not None:
            return number
        following = normalized[match.end() :]
        next_line = following.splitlines()[0] if following else ""
        number = _last_number(next_line)
        if number is not None:
            return number
    return _last_number(normalized)


def _normalize_numeric_text(text: str) -> str:
    normalized = re.sub(r"[*_`]+", "", text or "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"(?<=\d)\.(?=\s*(?:$|\n))", "", normalized)
    return normalized


def _last_number(text: str) -> float | None:
    matches = list(re.finditer(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text or ""))
    if not matches:
        return None
    value = matches[-1].group(0).rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def _parse_numeric_tolerance(raw: str) -> float:
    if not raw:
        return 1e-6
    try:
        tolerance = float(raw)
    except ValueError:
        return 1e-6
    return max(tolerance, 0.0)


def score_json_field(prediction: str, expected: str, field_name: str) -> bool:
    actual = extract_json_field_value(prediction, field_name)
    return normalize_label(str(actual)) == normalize_label(expected) if actual is not None else False


def score_expected_json(prediction: str, row: dict[str, Any]) -> dict[str, Any]:
    expected_text = str(row.get("expected_json") or row.get("metadata", {}).get("expected_json") or "").strip()
    expected_payload = parse_json_object_prefix(expected_text)
    actual_payload = parse_json_object_prefix(prediction)
    strict_fields = _split_metadata_list(row, "strict_fields")
    soft_fields = set(_split_metadata_list(row, "soft_fields"))
    if expected_payload is None:
        return {
            "passed": False,
            "reason": "expected_json is not a JSON object",
            "expected_payload": None,
            "actual_payload": actual_payload,
            "mismatches": [{"path": "$", "expected": "json object", "actual": expected_text}],
        }
    if actual_payload is None:
        result = _score_truncated_expected_json(prediction, expected_payload, strict_fields, soft_fields)
        if result is not None:
            return result
        return {
            "passed": False,
            "reason": "prediction is not a JSON object",
            "expected_payload": expected_payload,
            "actual_payload": None,
            "mismatches": [{"path": "$", "expected": expected_payload, "actual": prediction}],
        }

    expected_paths = strict_fields or [path for path, _ in _json_leaf_items(expected_payload)]
    expected_paths = [path for path in expected_paths if path not in soft_fields]
    mismatches: list[dict[str, Any]] = []
    for path in expected_paths:
        expected_value = _json_path_get(expected_payload, path)
        actual_value = _json_path_get(actual_payload, path)
        if not _json_values_equal(expected_value, actual_value):
            mismatches.append({"path": path, "expected": expected_value, "actual": actual_value})

    expected_arguments = expected_payload.get("arguments")
    actual_arguments = actual_payload.get("arguments")
    if isinstance(expected_arguments, dict):
        expected_keys = {key for key in expected_arguments if f"arguments.{key}" not in soft_fields}
        actual_keys = (
            {key for key in actual_arguments if f"arguments.{key}" not in soft_fields}
            if isinstance(actual_arguments, dict)
            else set()
        )
        if expected_keys and expected_keys != actual_keys:
            mismatches.append(
                {
                    "path": "arguments",
                    "expected_keys": sorted(expected_keys),
                    "actual_keys": sorted(actual_keys),
                }
            )

    return {
        "passed": not mismatches,
        "reason": "ok" if not mismatches else "json mismatch",
        "expected_payload": expected_payload,
        "actual_payload": actual_payload,
        "mismatches": mismatches,
    }


def _score_truncated_expected_json(
    prediction: str,
    expected_payload: dict[str, Any],
    strict_fields: list[str],
    soft_fields: set[str],
) -> dict[str, Any] | None:
    expected_paths = strict_fields or [path for path, _ in _json_leaf_items(expected_payload)]
    expected_paths = [path for path in expected_paths if path not in soft_fields]
    if not expected_paths:
        return None

    mismatches: list[dict[str, Any]] = []
    extracted: dict[str, Any] = {}
    for path in expected_paths:
        expected_value = _json_path_get(expected_payload, path)
        actual_value = extract_json_field_value(prediction, path)
        extracted[path] = actual_value
        if actual_value is None or not _json_values_equal(expected_value, actual_value):
            mismatches.append({"path": path, "expected": expected_value, "actual": actual_value})

    expected_arguments = expected_payload.get("arguments")
    if isinstance(expected_arguments, dict):
        strict_argument_keys = {
            path.split(".", 1)[1]
            for path in expected_paths
            if path.startswith("arguments.") and "." not in path.split(".", 1)[1]
        }
        missing_keys = [key for key in strict_argument_keys if extracted.get(f"arguments.{key}") is None]
        if missing_keys:
            mismatches.append({"path": "arguments", "missing_keys": sorted(missing_keys)})

    return {
        "passed": not mismatches,
        "reason": "ok_truncated_json_fields" if not mismatches else "truncated json mismatch",
        "expected_payload": expected_payload,
        "actual_payload": None,
        "actual_fields": extracted,
        "mismatches": mismatches,
    }


def _split_metadata_list(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key) or row.get("metadata", {}).get(key) or ""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


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


def extract_json_field_value(prediction: str, field_name: str) -> Any:
    if not field_name:
        return None
    payload = parse_json_object_prefix(prediction)
    if payload is None:
        return extract_json_field_by_regex(prediction, field_name)
    value: Any = payload
    for part in field_name.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return extract_json_field_by_regex(prediction, field_name)
    return value


def parse_json_object_prefix(prediction: str) -> dict[str, Any] | None:
    text = (prediction or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        payload, _ = json.JSONDecoder().raw_decode(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def extract_json_field_by_regex(prediction: str, field_name: str) -> str | None:
    text = prediction or ""
    field = field_name.split(".")[-1]
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1)
    match = re.search(rf"'{re.escape(field)}'\s*:\s*'([^']*)'", text)
    return match.group(1) if match else None


def normalize_label(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def run_llm_judge(row: dict[str, Any], prediction: str, judge: JudgeConfig) -> dict[str, Any]:
    payload = {
        "prompt": row.get("prompt", ""),
        "expected": row.get("expected") or row.get("reference", ""),
        "prediction": prediction,
        "grader": row.get("grader", ""),
        "metadata": row.get("metadata", {}),
    }
    raw_payload = json.dumps(payload, ensure_ascii=False)
    args = _command_args(judge.command, raw_payload)
    use_stdin = not any("{prompt}" in part or "{payload}" in part for part in shlex.split(judge.command))
    args = [part.replace("{payload}", raw_payload) for part in args]
    completed = subprocess.run(
        args,
        input=raw_payload if use_stdin else None,
        capture_output=True,
        text=True,
        timeout=judge.timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "score": 0.0,
            "passed": False,
            "mode": "llm_judge_error",
            "error": completed.stderr.strip(),
        }
    return parse_judge_output(completed.stdout)


def parse_judge_output(output: str) -> dict[str, Any]:
    text = output.strip()
    try:
        payload = json.loads(text)
        score = payload.get("score")
        passed = payload.get("passed")
        if passed is None and score is not None:
            passed = float(score) >= 0.5
        if score is None and passed is not None:
            score = float(bool(passed))
        return {
            "score": score,
            "passed": passed,
            "mode": "llm_judge",
            "reason": payload.get("reason") or payload.get("rationale") or "",
            "raw": payload,
        }
    except Exception:
        lowered = text.lower()
        passed = lowered.startswith("pass") or lowered in {"true", "yes", "1"}
        failed = lowered.startswith("fail") or lowered in {"false", "no", "0"}
        return {
            "score": float(passed) if passed or failed else None,
            "passed": passed if passed or failed else None,
            "mode": "llm_judge_text",
            "reason": text,
        }


def render_eval_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    if _is_practical_row(row):
        system = row.get("system") or "Complete the user's practical evaluation task. Return only the requested answer."
        return [{"role": "system", "content": system}, {"role": "user", "content": row["prompt"]}]

    benchmark = row["benchmark"]
    prompt = row["prompt"]
    if benchmark == "gsm8k":
        system = (
            "You are a careful math tutor. Solve the user's grade-school math problem step by step. "
            "End with one line exactly like: Final answer: <number>"
        )
        user = f"Solve this grade-school math problem.\n\nQuestion: {prompt}"
    elif benchmark == "bfcl":
        tools = row.get("metadata", {}).get("tools", [])
        system = (
            "You are a function-calling model. Reply with only the selected tool call as JSON, "
            'using {"name": "<tool_name>", "arguments": {...}}.'
        )
        user = f"Available tools:\n{tools}\n\nRequest:\n{prompt}"
    elif benchmark == "humaneval_plus":
        system = "Complete the Python function. Reply with code only."
        user = prompt
    else:
        system = "Follow the user's instruction exactly."
        user = prompt
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def plain_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


class BaseRunner:
    def __init__(self, config: ModelConfig, max_tokens: int, log: LogFn) -> None:
        self.config = config
        self.max_tokens = max_tokens
        self.log = log

    def generate(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MlxRunner(BaseRunner):
    def __init__(self, config: ModelConfig, max_tokens: int, log: LogFn) -> None:
        super().__init__(config, max_tokens, log)
        from bankai_poc.model.real_mlx import load_real_model

        if not config.model_ref:
            raise ValueError(f"{config.name} needs an MLX model reference")
        self.log(f"[{config.name}] loading {config.model_ref}")
        self.handle = load_real_model(config.model_ref)

    def _prompt(self, row: dict[str, Any]) -> str:
        from bankai_poc.model.real_mlx import render_chat_prompt

        messages = render_eval_prompt(row)
        return render_chat_prompt(self.handle, messages)

    def generate(self, row: dict[str, Any]) -> str:
        from bankai_poc.model.real_mlx import generate_text

        return generate_text(self.handle, self._prompt(row), max_tokens=self.max_tokens)


class PatchedMlxRunner(MlxRunner):
    def __init__(self, config: ModelConfig, max_tokens: int, log: LogFn) -> None:
        super().__init__(config, max_tokens, log)
        from bankai_poc.model.patching import load_patch

        if config.patch_path is None:
            raise ValueError(f"{config.name} needs a patch path")
        patch_path = config.patch_path
        if not patch_path.is_absolute() and not patch_path.exists():
            patch_path = root_dir() / "patches" / patch_path
        self.log(f"[{config.name}] loading patch {patch_path}")
        self.patch = load_patch(patch_path)

    def generate(self, row: dict[str, Any]) -> str:
        from bankai_poc.model.real_mlx import apply_real_patch, generate_text, revert_real_patch

        prompt = self._prompt(row)
        apply_real_patch(self.handle.model, self.patch)
        try:
            return generate_text(self.handle, prompt, max_tokens=self.max_tokens)
        finally:
            revert_real_patch(self.handle.model, self.patch)


class TerminalAgentRunner(BaseRunner):
    def generate(self, row: dict[str, Any]) -> str:
        if not self.config.command.strip():
            raise ValueError(f"{self.config.name} needs a command")
        prompt = plain_prompt(render_eval_prompt(row))
        args = _command_args(self.config.command, prompt)
        use_stdin = not any("{prompt}" in part for part in shlex.split(self.config.command))
        completed = subprocess.run(
            args,
            input=prompt if use_stdin else None,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"command exited {completed.returncode}: {stderr}")
        return completed.stdout.strip()


def _command_args(command: str, prompt: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Command is empty")
    return [part.replace("{prompt}", prompt) for part in parts]


def _runner_for(config: ModelConfig, max_tokens: int, log: LogFn) -> BaseRunner:
    if config.kind == "bankai_mlx":
        return PatchedMlxRunner(config, max_tokens, log)
    if config.kind == "mlx":
        return MlxRunner(config, max_tokens, log)
    if config.kind == "terminal":
        return TerminalAgentRunner(config, max_tokens, log)
    raise ValueError(f"Unknown model kind: {config.kind}")


def run_pipeline(config: PipelineConfig, log: LogFn | None = None) -> tuple[Path, Path]:
    log = log or (lambda message: None)
    rows = read_eval_rows(config.dataset_path)
    output_dir = config.output_dir or default_eval_runs_dir() / f"run_{timestamp_id()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = [model for model in config.models if model.enabled]
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_config in enabled_models:
        runner = _runner_for(model_config, config.max_tokens, log)
        started = time.perf_counter()
        model_details: list[dict[str, Any]] = []
        try:
            for index, row in enumerate(rows, start=1):
                task = row.get("task") or row.get("benchmark", "eval")
                log(f"[{model_config.name}] {index}/{len(rows)} {task} {row['id']}")
                row_started = time.perf_counter()
                try:
                    prediction = runner.generate(row)
                    error = None
                except Exception as exc:
                    prediction = ""
                    error = f"{type(exc).__name__}: {exc}"
                    log(f"[{model_config.name}] error: {error}")
                latency_ms = (time.perf_counter() - row_started) * 1000.0
                score = (
                    score_prediction(row, prediction, config.judge)
                    if error is None
                    else {"score": 0.0, "passed": False, "mode": "error"}
                )
                detail = {
                    "model": model_config.name,
                    "model_kind": model_config.kind,
                    "benchmark": task,
                    "task": task,
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "reference": row.get("reference", ""),
                    "expected": row.get("expected", row.get("reference", "")),
                    "expected_json": row.get("expected_json", ""),
                    "strict_fields": row.get("strict_fields", row.get("metadata", {}).get("strict_fields", "")),
                    "soft_fields": row.get("soft_fields", row.get("metadata", {}).get("soft_fields", "")),
                    "grader": row.get("grader", ""),
                    "prediction": prediction,
                    "error": error,
                    "latency_ms": latency_ms,
                    "score": score,
                }
                model_details.append(detail)
                detail_rows.append(detail)
        finally:
            runner.close()
        summary_rows.extend(_summaries_for_model(model_config, model_details, time.perf_counter() - started))

    summary_path = output_dir / "summary.json"
    details_path = output_dir / "details.json"
    dump_json(
        summary_path,
        {
            "dataset_path": str(config.dataset_path),
            "max_tokens": config.max_tokens,
            "judge": dataclasses.asdict(config.judge),
            "models": [_model_config_json(model) for model in enabled_models],
            "summary": summary_rows,
        },
    )
    dump_json(details_path, {"rows": detail_rows})
    log(f"Wrote {summary_path}")
    log(f"Wrote {details_path}")
    return summary_path, details_path


def _model_config_json(model: ModelConfig) -> dict[str, Any]:
    payload = dataclasses.asdict(model)
    if payload["patch_path"] is not None:
        payload["patch_path"] = str(payload["patch_path"])
    return payload


def _summaries_for_model(model: ModelConfig, rows: Iterable[dict[str, Any]], elapsed_s: float) -> list[dict[str, Any]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_list = list(rows)
    for row in row_list:
        by_benchmark[row["benchmark"]].append(row)

    summaries = [_summary_row(model, "all", row_list, elapsed_s)]
    for benchmark, benchmark_rows in sorted(by_benchmark.items()):
        summaries.append(_summary_row(model, benchmark, benchmark_rows, elapsed_s))
    return summaries


def _summary_row(model: ModelConfig, benchmark: str, rows: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["score"].get("passed") is True)
    scored = sum(1 for row in rows if row["score"].get("passed") is not None)
    errors = sum(1 for row in rows if row.get("error"))
    mean_latency = sum(float(row["latency_ms"]) for row in rows) / total if total else 0.0
    return {
        "model": model.name,
        "kind": model.kind,
        "benchmark": benchmark,
        "examples": total,
        "scored_examples": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else 0.0,
        "errors": errors,
        "mean_latency_ms": mean_latency,
        "elapsed_s": elapsed_s,
    }
