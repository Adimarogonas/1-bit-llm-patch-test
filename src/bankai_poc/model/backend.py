from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class MockModule:
    weight: np.ndarray
    scales: np.ndarray


@dataclass
class MockBonsaiBackend:
    modules: Dict[Tuple[int, str], MockModule]

    @classmethod
    def from_seed(
        cls,
        seed: int = 7,
        rows: int = 128,
        cols: int = 128,
        layers: tuple[int, ...] = (1, 2, 3, 4, 34),
        projs: tuple[str, ...] = ("gate_proj", "up_proj"),
    ) -> "MockBonsaiBackend":
        rng = np.random.default_rng(seed)
        modules: Dict[Tuple[int, str], MockModule] = {}
        for layer in layers:
            for proj in projs:
                modules[(layer, proj)] = MockModule(
                    weight=rng.integers(0, np.iinfo(np.uint32).max, size=(rows, cols), dtype=np.uint32),
                    scales=rng.random(size=(rows, cols), dtype=np.float32),
                )
        return cls(modules=modules)

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.modules):
            digest.update(f"{key[0]}:{key[1]}".encode("utf-8"))
            digest.update(self.modules[key].weight.tobytes())
        return digest.hexdigest()

    def copy(self) -> "MockBonsaiBackend":
        return MockBonsaiBackend(
            modules={
                key: MockModule(weight=value.weight.copy(), scales=value.scales.copy())
                for key, value in self.modules.items()
            }
        )
