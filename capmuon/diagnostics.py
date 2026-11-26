"""Exact F = X^T X / N input eigenspectrum diagnostics for any PyTorch model."""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def inspect_linear_eigenspectra(
    model: nn.Module,
    forward_fn: Callable[[nn.Module], None],
    filter_fn: Optional[Callable[[str, nn.Linear], bool]] = None,
    max_d_in: int = 4096,
    token_mask: Optional[torch.Tensor] = None,
) -> dict[str, dict]:
    """
    Compute exact input second-moment eigenspectra F = X^T X / N across linear layers.

    Args:
        model: Target PyTorch module.
        forward_fn: Callable `forward_fn(model)` that runs one or more forward passes.
        filter_fn: Optional predicate `(name, module) -> bool` selecting layers to probe.
        max_d_in: Skip layers wider than this for fast exact eigh (default: 4096).
        token_mask: Optional 1D boolean mask matching flattened batch*seq length (e.g. non-pad tokens).

    Returns:
        Dict mapping layer name to exact spectral metrics:
        `d_in`, `ratio` (lam1/mean), `cliff` (lam1/lam2), `pct_e1`, `cos_mu`, `top32`.
    """
    captured: dict[str, list[torch.Tensor]] = {}
    handles = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.shape[1] > max_d_in:
            continue
        if filter_fn is not None and not filter_fn(name, mod):
            continue
        captured[name] = []

        def _make_hook(k: str):
            def _hook(_m, inputs):
                if inputs and torch.is_tensor(inputs[0]):
                    captured[k].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu())
            return _hook

        handles.append(mod.register_forward_pre_hook(_make_hook(name)))

    try:
        forward_fn(model)
    finally:
        for h in handles:
            h.remove()

    mask_cpu = token_mask.reshape(-1).bool().cpu() if token_mask is not None else None
    results: dict[str, dict] = {}
    for name, chunks in captured.items():
        if not chunks:
            continue
        X = torch.cat(chunks, dim=0)
        if mask_cpu is not None and X.shape[0] == mask_cpu.shape[0]:
            X = X[mask_cpu]
        N, d = X.shape
        if N < 4:
            continue

        F_mat = (X.T @ X) / float(N)
        evals, evecs = torch.linalg.eigh(F_mat)
        evals = evals.flip(0).clamp_min(0.0)
        v1 = evecs[:, -1]

        mean_eig = max(evals.mean().item(), 1e-12)
        l1 = evals[0].item()
        l2 = max(evals[1].item(), 1e-12) if d > 1 else mean_eig
        mu = X.mean(dim=0)
        cos_mu = abs(float((v1 @ mu).item()) / (mu.norm().item() + 1e-12))
        pct_e1 = 100.0 * l1 / max(evals.sum().item(), 1e-12)

        results[name] = {
            "d_in": d,
            "ratio": l1 / mean_eig,
            "cliff": l1 / l2,
            "pct_e1": pct_e1,
            "cos_mu": cos_mu,
            "top32": (evals[: min(32, d)] / mean_eig).tolist(),
        }

    return results
