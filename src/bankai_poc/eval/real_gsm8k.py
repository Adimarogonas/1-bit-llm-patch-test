from __future__ import annotations

from pathlib import Path
from typing import Any

from bankai_poc.data.registry import data_dir, root_dir
from bankai_poc.eval.scorers import extract_gsm8k_answer, gsm8k_output_diagnostics, score_gsm8k
from bankai_poc.model.patching import load_patch
from bankai_poc.model.real_mlx import (
    apply_real_patch,
    generate_text,
    load_real_model,
    render_chat_prompt,
    revert_real_patch,
)
from bankai_poc.utils.io import dump_json, read_jsonl


def _gsm8k_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful math tutor. Solve the user's grade-school math problem step by step. "
                "End your response with a single line in the exact format: Final answer: <number>"
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem. Show concise reasoning, then end with the required final line.\n\n"
                f"Question: {question}"
            ),
        },
    ]


def run_real_gsm8k_compare(
    model_ref: str,
    patch_name: str,
    limit: int = 50,
    max_tokens: int = 160,
) -> tuple[Path, Path]:
    rows = read_jsonl(data_dir("gsm8k") / "normalized.jsonl")[:limit]
    handle = load_real_model(model_ref)
    patch_path = root_dir() / "patches" / patch_name
    patch = load_patch(patch_path)

    example_rows: list[dict[str, Any]] = []
    base_correct = 0
    patched_correct = 0

    for index, row in enumerate(rows, start=1):
        print(f"[gsm8k {index}/{len(rows)}] generating base", flush=True)
        prompt = render_chat_prompt(handle, _gsm8k_messages(row["prompt"]))
        base_output = generate_text(handle, prompt, max_tokens=max_tokens)
        base_pass = score_gsm8k(base_output, row["reference"])
        base_diag = gsm8k_output_diagnostics(base_output)

        print(f"[gsm8k {index}/{len(rows)}] generating patched", flush=True)
        apply_real_patch(handle.model, patch)
        patched_output = generate_text(handle, prompt, max_tokens=max_tokens)
        revert_real_patch(handle.model, patch)
        patched_pass = score_gsm8k(patched_output, row["reference"])
        patched_diag = gsm8k_output_diagnostics(patched_output)

        base_correct += int(base_pass)
        patched_correct += int(patched_pass)
        example_rows.append(
            {
                "index": index,
                "question": row["prompt"],
                "reference": row["reference"],
                "reference_extracted": extract_gsm8k_answer(row["reference"]),
                "prompt": prompt,
                "base_output": base_output,
                "base_extracted": base_diag["extracted"],
                "base_correct": base_pass,
                "base_diagnostics": base_diag,
                "patched_output": patched_output,
                "patched_extracted": patched_diag["extracted"],
                "patched_correct": patched_pass,
                "patched_diagnostics": patched_diag,
                "changed": base_output != patched_output,
            }
        )
        print(
            f"[gsm8k {index}/{len(rows)}] base={int(base_pass)} patched={int(patched_pass)} "
            f"running_base={base_correct}/{index} running_patched={patched_correct}/{index}",
            flush=True,
        )

    results_dir = root_dir() / "results"
    summary_path = results_dir / "gsm8k_real_compare_summary.json"
    details_path = results_dir / "gsm8k_real_compare_details.json"

    dump_json(
        summary_path,
        {
            "model": model_ref,
            "patch": patch_name,
            "examples": len(example_rows),
            "prompt_style": "qwen3_chatml_final_answer",
            "max_tokens": max_tokens,
            "base_correct": base_correct,
            "patched_correct": patched_correct,
            "base_accuracy": base_correct / len(example_rows) if example_rows else 0.0,
            "patched_accuracy": patched_correct / len(example_rows) if example_rows else 0.0,
            "delta_accuracy": ((patched_correct - base_correct) / len(example_rows)) if example_rows else 0.0,
            "changed_generations": sum(1 for row in example_rows if row["changed"]),
            "base_with_final_answer_marker": sum(1 for row in example_rows if row["base_diagnostics"]["has_final_answer_marker"]),
            "patched_with_final_answer_marker": sum(1 for row in example_rows if row["patched_diagnostics"]["has_final_answer_marker"]),
            "base_likely_truncated": sum(1 for row in example_rows if row["base_diagnostics"]["likely_truncated"]),
            "patched_likely_truncated": sum(1 for row in example_rows if row["patched_diagnostics"]["likely_truncated"]),
        },
    )
    dump_json(details_path, {"rows": example_rows})
    return summary_path, details_path
