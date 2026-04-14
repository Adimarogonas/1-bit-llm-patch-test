from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bankai_poc.utils.io import dump_json, load_json

from .backend import MockBonsaiBackend


@dataclass
class PatchFlip:
    layer: int
    proj: str
    row: int


@dataclass
class BankaiPatch:
    name: str
    description: str
    base_model: str
    flips: list[PatchFlip] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    format: str = "bankai_row_xor_v1"

    @property
    def n_bits_flipped(self) -> int:
        return len(self.flips) * 4096

    @property
    def size_bytes(self) -> int:
        return len(self.flips) * 12

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "format": self.format,
            "name": self.name,
            "description": self.description,
            "base_model": self.base_model,
            "flips": [asdict(flip) for flip in self.flips],
            "stats": {
                "n_flips": len(self.flips),
                "bits_flipped": self.n_bits_flipped,
                "size_bytes": self.size_bytes,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BankaiPatch":
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            base_model=payload.get("base_model", "unknown"),
            flips=[PatchFlip(**flip) for flip in payload["flips"]],
            metadata=payload.get("metadata", {}),
            version=payload.get("version", 1),
            format=payload.get("format", "bankai_row_xor_v1"),
        )


def apply_patch(model: MockBonsaiBackend, patch: BankaiPatch) -> float:
    started = time.perf_counter()
    for flip in patch.flips:
        module = model.modules[(flip.layer, flip.proj)]
        module.weight[flip.row] ^= np.uint32(0xFFFFFFFF)
    return (time.perf_counter() - started) * 1000.0


def revert_patch(model: MockBonsaiBackend, patch: BankaiPatch) -> float:
    return apply_patch(model, patch)


def verify_reversibility(model: MockBonsaiBackend, patch: BankaiPatch) -> dict[str, Any]:
    before = model.checksum()
    apply_ms = apply_patch(model, patch)
    after_apply = model.checksum()
    revert_ms = revert_patch(model, patch)
    after_revert = model.checksum()
    return {
        "base_checksum": before,
        "after_apply_checksum": after_apply,
        "after_revert_checksum": after_revert,
        "reversible": before == after_revert,
        "apply_latency_ms": apply_ms,
        "revert_latency_ms": revert_ms,
        "size_bytes": patch.size_bytes,
    }


def save_patch(path: Path, patch: BankaiPatch) -> None:
    dump_json(path, patch.to_json())


def load_patch(path: Path) -> BankaiPatch:
    return BankaiPatch.from_json(load_json(path))
