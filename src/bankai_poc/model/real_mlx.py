from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load

from bankai_poc.model.patching import BankaiPatch


@dataclass
class RealModelHandle:
    model_ref: str
    model: Any
    tokenizer: Any


def load_real_model(model_ref: str) -> RealModelHandle:
    model, tokenizer = load(model_ref)
    return RealModelHandle(model_ref=model_ref, model=model, tokenizer=tokenizer)


def get_module(model: Any, path: str) -> Any:
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def model_patchable_summary(model: Any) -> dict[str, Any]:
    candidates = []
    for layer in [1, 2, 3, 4, 34]:
        for proj in ["gate_proj", "up_proj"]:
            path = f"model.layers.{layer}.mlp.{proj}"
            try:
                mod = get_module(model, path)
                candidates.append(
                    {
                        "path": path,
                        "weight_shape": tuple(mod.weight.shape),
                        "weight_dtype": str(mod.weight.dtype),
                        "has_scales": hasattr(mod, "scales"),
                        "scales_shape": tuple(mod.scales.shape) if hasattr(mod, "scales") else None,
                    }
                )
            except Exception as exc:
                candidates.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
    def _is_patchable_candidate(item: dict[str, Any]) -> bool:
        dtype = str(item.get("weight_dtype", "")).lower()
        weight_shape = item.get("weight_shape")
        scales_shape = item.get("scales_shape")
        return (
            item.get("has_scales") is True
            and "uint32" in dtype
            and isinstance(weight_shape, tuple)
            and len(weight_shape) == 2
            and weight_shape[1] == 128
            and isinstance(scales_shape, tuple)
            and len(scales_shape) == 2
        )

    patchable = any(_is_patchable_candidate(item) for item in candidates)
    return {"patchable": patchable, "candidates": candidates}


def flip_row(model: Any, layer: int, proj: str, row: int) -> None:
    path = f"model.layers.{layer}.mlp.{proj}"
    mod = get_module(model, path)
    w = mod.weight
    mask = mx.zeros_like(w)
    ones = mx.full((w.shape[1],), 0xFFFFFFFF, dtype=mx.uint32)
    mask = mask.at[row].add(ones)
    new_w = w ^ mask
    model.load_weights([(f"{path}.weight", new_w)], strict=False)


def apply_real_patch(model: Any, patch: BankaiPatch) -> None:
    for flip in patch.flips:
        flip_row(model, flip.layer, flip.proj, flip.row)
    mx.eval(model.parameters())


def revert_real_patch(model: Any, patch: BankaiPatch) -> None:
    apply_real_patch(model, patch)


def generate_text(handle: RealModelHandle, prompt: str, max_tokens: int = 80) -> str:
    return generate(handle.model, handle.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def render_chat_prompt(handle: RealModelHandle, messages: list[dict[str, str]]) -> str:
    tokenizer = handle.tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
