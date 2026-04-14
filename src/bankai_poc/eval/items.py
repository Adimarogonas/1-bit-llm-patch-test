from __future__ import annotations

from bankai_poc.data.registry import data_dir
from bankai_poc.utils.io import read_jsonl, write_jsonl


def build_eval_items(benchmark: str) -> None:
    rows = read_jsonl(data_dir(benchmark) / "normalized.jsonl")
    items = []
    for row in rows:
        items.append(
            {
                "benchmark": benchmark,
                "id": row["id"],
                "prompt": row["prompt"],
                "reference": row["reference"],
                "metadata": row["metadata"],
            }
        )
    write_jsonl(data_dir(benchmark) / "eval_items.jsonl", items)
