from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeStatus:
    python: str
    mlx_available: bool
    mlx_version: str | None
    bankai_available: bool
    github_bankai_compatible: bool
    mlx_lm_available: bool
    metal_compiler_available: bool
    metallib_available: bool
    prism_mlx_ready: bool
    datasets_available: bool
    numpy_available: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "mlx_available": self.mlx_available,
            "mlx_version": self.mlx_version,
            "bankai_available": self.bankai_available,
            "github_bankai_compatible": self.github_bankai_compatible,
            "mlx_lm_available": self.mlx_lm_available,
            "metal_compiler_available": self.metal_compiler_available,
            "metallib_available": self.metallib_available,
            "prism_mlx_ready": self.prism_mlx_ready,
            "datasets_available": self.datasets_available,
            "numpy_available": self.numpy_available,
            "notes": self.notes,
        }


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _find_xcrun_tool(name: str) -> str | None:
    if shutil.which("xcrun") is None:
        return None
    try:
        result = subprocess.run(
            ["xcrun", "--find", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _mlx_version() -> str | None:
    if not _has_module("mlx"):
        return None
    try:
        import importlib.metadata

        return importlib.metadata.version("mlx")
    except Exception:
        return None


def inspect_runtime() -> RuntimeStatus:
    notes: list[str] = []
    mlx_available = _has_module("mlx")
    mlx_version = _mlx_version()
    bankai_available = _has_module("bankai")
    mlx_lm_available = _has_module("mlx_lm")
    datasets_available = _has_module("datasets")
    numpy_available = _has_module("numpy")
    github_bankai_compatible = False
    metal_path = _find_xcrun_tool("metal")
    metallib_path = _find_xcrun_tool("metallib")
    metal_compiler_available = metal_path is not None
    metallib_available = metallib_path is not None
    prism_mlx_ready = mlx_available and mlx_lm_available and metal_compiler_available and metallib_available

    if bankai_available:
        try:
            import bankai  # type: ignore

            github_bankai_compatible = hasattr(bankai, "patch") or hasattr(bankai, "Patch")
        except Exception:
            github_bankai_compatible = False

    if not mlx_available:
        notes.append("`mlx` not installed; real Bonsai/Bankai patch application is unavailable.")
    if mlx_available and mlx_version is not None:
        notes.append(f"`mlx` version detected: {mlx_version}. Stock MLX will not load Bonsai 1-bit correctly; use the PrismML fork.")
    if not mlx_lm_available:
        notes.append("`mlx_lm` not installed; cannot load MLX language models.")
    if not metal_compiler_available or not metallib_available:
        notes.append("Metal compiler tools are not fully available through `xcrun`; PrismML MLX builds will fail.")
    if bankai_available and not github_bankai_compatible:
        notes.append("A `bankai` package is installed, but it is not the GitHub XOR-patching Bankai repo.")
    if not bankai_available:
        notes.append("GitHub Bankai package not installed; using local Bankai-shaped patch format only.")
    if not (mlx_available and mlx_lm_available):
        notes.append("Patch search/eval can still run in mock mode for pipeline validation.")
    if prism_mlx_ready:
        notes.append("MLX runtime prerequisites look present. Real Bonsai loading still depends on having the PrismML MLX fork installed.")

    return RuntimeStatus(
        python=sys.version.split()[0],
        mlx_available=mlx_available,
        mlx_version=mlx_version,
        bankai_available=bankai_available,
        github_bankai_compatible=github_bankai_compatible,
        mlx_lm_available=mlx_lm_available,
        metal_compiler_available=metal_compiler_available,
        metallib_available=metallib_available,
        prism_mlx_ready=prism_mlx_ready,
        datasets_available=datasets_available,
        numpy_available=numpy_available,
        notes=notes,
    )
