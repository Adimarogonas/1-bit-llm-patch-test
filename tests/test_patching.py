import numpy as np

from bankai_poc.model.backend import MockBonsaiBackend
from bankai_poc.model.patching import BankaiPatch, PatchFlip, load_patch, save_patch, verify_reversibility
from bankai_poc.utils.io import dump_json, load_json


def test_xor_patch_is_exactly_reversible() -> None:
    model = MockBonsaiBackend.from_seed()
    patch = BankaiPatch(
        name="test",
        description="test patch",
        base_model="mock",
        flips=[PatchFlip(layer=1, proj="gate_proj", row=0)],
    )
    verification = verify_reversibility(model, patch)
    assert verification["reversible"] is True


def test_save_patch_serializes_numpy_scalar_flip_values(tmp_path) -> None:
    patch_path = tmp_path / "patch.json"
    patch = BankaiPatch(
        name="numpy-row",
        description="patch with numpy scalar values",
        base_model="mock",
        flips=[PatchFlip(layer=np.int64(34), proj="gate_proj", row=np.int64(127))],
        metadata={"final_fitness": np.float32(0.125), "search_layers": np.array([0, 1, 34])},
    )

    save_patch(patch_path, patch)
    payload = load_json(patch_path)
    loaded = load_patch(patch_path)

    assert payload["flips"][0] == {"layer": 34, "proj": "gate_proj", "row": 127}
    assert payload["metadata"]["final_fitness"] == 0.125
    assert payload["metadata"]["search_layers"] == [0, 1, 34]
    assert loaded.flips[0] == PatchFlip(layer=34, proj="gate_proj", row=127)


def test_dump_json_serializes_numpy_values(tmp_path) -> None:
    output_path = tmp_path / "payload.json"

    dump_json(output_path, {"row": np.int64(7), "score": np.float32(1.5), "layers": np.array([0, 4, 35])})

    assert load_json(output_path) == {"row": 7, "score": 1.5, "layers": [0, 4, 35]}
