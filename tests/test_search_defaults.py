from __future__ import annotations

import ast
from pathlib import Path
import re

DEFAULT_LAYERS = [0, 1, 2, 3, 4, 34, 35]
DEFAULT_PROJS = ["gate_proj", "up_proj", "down_proj"]


def test_default_search_layers_match_paper_layer_set() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "bankai_poc" / "search" / "real_runner.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    assignments = {node.target.id: node.value for node in module.body if isinstance(node, ast.AnnAssign)}

    assert ast.literal_eval(assignments["DEFAULT_SEARCH_LAYERS"]) == DEFAULT_LAYERS
    assert ast.literal_eval(assignments["DEFAULT_SEARCH_PROJS"]) == DEFAULT_PROJS
    assert '"search_layers": DEFAULT_SEARCH_LAYERS' in source
    assert '"search_projs": DEFAULT_SEARCH_PROJS' in source
    assert '"stable": DEFAULT_SEARCH_LAYERS' in source


def test_benchmark_configs_use_default_search_scope() -> None:
    root = Path(__file__).resolve().parents[1]

    for config_path in (root / "configs").glob("*.yaml"):
        config = config_path.read_text(encoding="utf-8")
        layers_match = re.search(r"^\s*search_layers:\s*(\[.*\])\s*$", config, re.MULTILINE)
        projs_match = re.search(r"^\s*search_projs:\s*(\[.*\])\s*$", config, re.MULTILINE)
        assert layers_match is not None
        assert projs_match is not None
        assert ast.literal_eval(layers_match.group(1)) == DEFAULT_LAYERS
        assert ast.literal_eval(projs_match.group(1)) == DEFAULT_PROJS


def test_candidate_rows_are_top_scale_not_first_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "bankai_poc" / "search" / "real_runner.py").read_text(encoding="utf-8")

    assert "np.argsort(row_scales)[-row_limit:][::-1]" in source
    assert "for row in range(row_limit)" not in source
