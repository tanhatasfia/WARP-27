#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from warp_kernel.model_utils import get_transformer_blocks, pack_block
from warp_kernel.reconstruction import ReconstructionCalibration, add_calibration_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    add_calibration_arguments(parser)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cpu"
    ).eval()
    blocks = get_transformer_blocks(model)
    calibration = ReconstructionCalibration(model, args) if args.calibration_text else None
    args.out.mkdir(parents=True, exist_ok=True)
    for index, block in enumerate(blocks):
        design = calibration.collect(block) if calibration else None
        for bits in (2, 3, 4):
            path = args.out / f"block_{index:03d}_w{bits}.pt"
            torch.save(pack_block(block, bits, args.group_size, reconstruction=design), path)
            print(path)
        del design
    manifest = {
        "model": args.model,
        "blocks": len(blocks),
        "group_size": args.group_size,
        "precisions": [2, 3, 4],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
