from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List

import torch
import torch.nn as nn

from .affine import fake_quantize_weight, affine_quantize_weight
from .packing import pack_quantized


def get_transformer_blocks(model: nn.Module) -> List[nn.Module]:
   
    candidates = [
        ("model", "layers"),               
        ("model", "decoder", "layers"),   
        ("transformer", "h"),              
    ]
    for path in candidates:
        obj = model
        ok = True
        for p in path:
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)):
            return list(obj)
    raise ValueError("Could not locate Transformer decoder blocks for this model")


def iter_block_linears(block: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


@contextmanager
def fake_quantized_block(block: nn.Module, bits: int, group_size: int = 128, *, reconstruction=None):
    """Temporarily install affine-quantized or reconstructed weights, then restore FP weights.

    `reconstruction` is the projection-name -> design-tensor mapping produced by
    ReconstructionCalibration.collect(block). Codes are fixed while scales/zeros
    are refitted using the very same refine_affine() used by pack_block().
    """
    saved = []
    try:
        for name, linear in iter_block_linears(block):
            original = linear.weight.data
            if reconstruction is None:
                qdq = fake_quantize_weight(original, bits, group_size)
            else:
                from .reconstruction import refine_affine
                from .affine import dequantize_weight
                if name not in reconstruction:
                    raise KeyError(f"Missing reconstruction design for projection {name!r}")
                qt = affine_quantize_weight(original, bits, group_size)
                qt = refine_affine(original, qt, reconstruction[name].to(original.device))
                qdq = dequantize_weight(qt, dtype=original.dtype)
            saved.append((linear, original))
            linear.weight.data = qdq
        yield block
    finally:
        for linear, original in reversed(saved):
            linear.weight.data = original


def pack_block(block: nn.Module, bits: int, group_size: int = 128, *, reconstruction=None) -> dict:
    """Pack every Linear weight in a Transformer block for the CUDA backend."""
    result = {}
    for name, linear in iter_block_linears(block):
        qt = affine_quantize_weight(linear.weight.detach(), bits, group_size)
        if reconstruction is not None:
            from .reconstruction import refine_affine

            qt = refine_affine(linear.weight.detach(), qt, reconstruction[name])
        pw = pack_quantized(qt)
        result[name] = {
            "packed": pw.packed.cpu(),
            "scales": pw.scales.cpu(),
            "zeros": pw.zeros.cpu(),
            "bits": bits,
            "group_size": group_size,
            "out_features": pw.out_features,
            "in_features": pw.in_features,
            "bias": None if linear.bias is None else linear.bias.detach().cpu(),
        }
    return result
