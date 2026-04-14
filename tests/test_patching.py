from bankai_poc.model.backend import MockBonsaiBackend
from bankai_poc.model.patching import BankaiPatch, PatchFlip, verify_reversibility


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
