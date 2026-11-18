

import math
from pathlib import Path

import torch

from .model_utils import iter_block_linears


def add_calibration_arguments(parser):
    parser.add_argument("--calibration-text", type=Path,
                        help="Text file with one calibration sequence per nonempty line")
    parser.add_argument("--calibration-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-sequences", type=int, default=4)
    parser.add_argument("--calibration-tokens", type=int, default=128)
    parser.add_argument("--window-tokens", type=int, default=16)
    parser.add_argument("--window-weight", type=float, default=1.0,
                        help="Weight for adjacent-window loss; 0 fits token MSE only")


class _BlockComplete(Exception):
    pass


class ReconstructionCalibration:
    def __init__(self, model, args):
        if args.window_tokens < 1 or args.calibration_sequences < 1:
            raise ValueError("Window size and sequence count must be positive")
        minimum = 2 * args.window_tokens if args.window_weight else 1
        if args.calibration_tokens < minimum or not math.isfinite(args.window_weight) or args.window_weight < 0:
            raise ValueError("Invalid calibration token limit or window weight")
        self.model = model.eval()
        self.device = torch.device(args.calibration_device)
        self.window = args.window_tokens
        self.weight = args.window_weight
        self.sequences = []
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        with args.calibration_text.open(encoding="utf-8") as corpus:
            for line in corpus:
                if not line.strip():
                    continue
                ids = tokenizer(line.strip(), return_tensors="pt", truncation=True,
                                max_length=args.calibration_tokens).input_ids
                if ids.shape[1] >= minimum:
                    self.sequences.append(ids)
                if len(self.sequences) == args.calibration_sequences:
                    break
        if not self.sequences:
            raise ValueError(f"Calibration text needs a line containing at least {minimum} tokens")
        self.model.to(self.device)

    @torch.no_grad()
    def collect(self, block):
        """Collect FP reference inputs to each projection, preserving sequence boundaries."""
        samples = {}
        hooks = []

        def capture(name):
            def hook(_, inputs):
                samples[name].append(inputs[0][0].detach().cpu())
            return hook

        def stop(*_):
            raise _BlockComplete

        try:
            for name, linear in iter_block_linears(block):
                samples[name] = []
                hooks.append(linear.register_forward_pre_hook(capture(name)))
            hooks.append(block.register_forward_hook(stop))
            for ids in self.sequences:
                try:
                    self.model(input_ids=ids.to(self.device), use_cache=False)
                except _BlockComplete:
                    pass
        finally:
            for hook in hooks:
                hook.remove()
        if any(len(inputs) != len(self.sequences) for inputs in samples.values()):
            raise ValueError("Expected one call per projection for each calibration sequence")
        return {name: self.design(inputs) for name, inputs in samples.items()}

    def design(self, inputs):

        tokens = torch.cat(inputs).to(self.device, torch.float32)
        tokens /= math.sqrt(tokens.shape[0])
        if not self.weight:
            return tokens
        differences = []
        for sequence in inputs:
            count = sequence.shape[0] // self.window
            if count >= 2:
                means = sequence[:count * self.window].to(self.device, torch.float32)
                means = means.reshape(count, self.window, -1).mean(dim=1)
                differences.append(means[1:] - means[:-1])
        if not differences:
            raise ValueError("No adjacent calibration windows are available")
        windows = torch.cat(differences)
        return torch.cat((tokens, windows * math.sqrt(self.weight / windows.shape[0])))


@torch.no_grad()
def refine_affine(weight, qt, design, row_batch=32):
    """Fit scale and intercept with fixed codes, then score rounded metadata."""
    device = design.device
    groups = qt.scales.shape[1]
    width = qt.group_size
    padding = groups * width - weight.shape[1]
    grouped = torch.nn.functional.pad(design, (0, padding)).reshape(-1, groups, width)
    sums = grouped.sum(dim=-1)

    scales = qt.scales.to(torch.float16).clone()
    zeros = qt.zeros.to(torch.float16).clone()
    for start in range(0, weight.shape[0], row_batch):
        rows = slice(start, start + row_batch)
        codes = torch.nn.functional.pad(qt.q[rows].to(device, torch.float32), (0, padding))
        codes = codes.reshape(-1, groups, width)
        old_scale = scales[rows].to(device, torch.float32)
        old_zero = zeros[rows].to(device, torch.float32)

        code_feature = torch.einsum("ngk,rgk->rng", grouped, codes)
        centered = code_feature - sums * old_zero[:, None, :]
        features = torch.cat((centered, sums.expand(codes.shape[0], -1, -1)), dim=-1)
        target = weight[rows].to(device, torch.float32) @ design.T
        residual = target - (centered * old_scale[:, None, :]).sum(dim=-1)
        transpose = features.transpose(1, 2)
        gram = transpose @ features
        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        diagonal.add_(diagonal.mean(dim=-1, keepdim=True) * 1e-3 + 1e-6)
        rhs = transpose @ residual.unsqueeze(-1)
        update = torch.linalg.solve(gram, rhs).squeeze(-1)
        scale_change, intercept = update.split(groups, dim=-1)

        new_scale = (old_scale + scale_change).clamp_min(torch.finfo(torch.float16).tiny)
        new_scale = new_scale.to(torch.float16).float()
        new_zero = (old_zero - intercept / new_scale).to(torch.float16).float()
        prediction = (code_feature * new_scale[:, None, :] -
                      sums * (new_scale * new_zero)[:, None, :]).sum(dim=-1)
        old_error = residual.square().sum(dim=-1)
        new_error = (target - prediction).square().sum(dim=-1)
        use_new = (torch.isfinite(new_error) & (new_error < old_error))[:, None]
        scales[rows] = torch.where(use_new, new_scale, old_scale).to(scales.device, scales.dtype)
        zeros[rows] = torch.where(use_new, new_zero, old_zero).to(zeros.device, zeros.dtype)

    qt.scales = scales
    qt.zeros = zeros
    return qt
