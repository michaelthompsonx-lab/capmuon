"""
CapMuon: O(d_in) Capped Input-Geometry Preconditioning for Muon.

Tracks the top-k (default k=1) input eigenmodes of F = E[x x^T] per 2D linear layer
via deterministic, matrix-free O(k * B * d) EMA power/subspace iteration (zero d x d
matrices, CUDA-graph safe) and applies a one-sided contraction cap S <= I:

    S = I - sum_{i=1}^k (1 - s_i) v_i v_i^T,    s_i = min(1, (c * mean(lam) / lam_i)^p)
    dW = -lr * polar(Z S) S * sqrt(max(1, R / C))

Unlike K-FAC or Shampoo whitening, S <= I never amplifies noisy tail eigenvalues.
Inner polar(Z S) redirects orthogonal update energy away from the dominant O(d)
common-mode spike into task-informative features; outer S bounds activation-space
step energy.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """5-step quintic Newton-Schulz polar factor approximation."""
    if G.ndim != 2:
        raise ValueError(f"Expected 2D matrix for Newton-Schulz, got shape {tuple(G.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if (G.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
    X = G.to(work_dtype)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.T
    return X


class CapTracker:
    """
    Matrix-free O(k * B * d) streaming EMA power/subspace iteration for F = E[x x^T].

    Maintains only:
      - V: [d, k] orthonormal estimate of top-k input eigenvectors
      - lam: [k] EMA Rayleigh quotients
      - tr_ema: scalar EMA trace(F)
      - s: [k] one-sided cap factors in (0, 1]
    """

    def __init__(
        self,
        d: int,
        device: torch.device,
        rank: int = 1,
        ema: float = 0.9,
        cap_c: float = 1.0,
        cap_p: float = 0.25,
    ):
        if rank < 1 or rank > d:
            raise ValueError(f"rank must satisfy 1 <= rank <= d (got rank={rank}, d={d})")
        self.d = d
        self.rank = rank
        self.ema = ema
        self.cap_c = cap_c
        self.cap_p = cap_p
        self.steps = 0
        self.V = torch.zeros(d, rank, device=device, dtype=torch.float32)
        self.V[0, 0] = 1.0
        self.lam = torch.zeros(rank, device=device, dtype=torch.float32)
        self.tr_ema = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.s = torch.ones(rank, device=device, dtype=torch.float32)
        self.top_ratio = 1.0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        X = x.detach().reshape(-1, self.d).float()
        B = X.size(0)
        if B == 0:
            return

        if self.steps == 0:
            # Deterministic initialization to batch mean captures common-mode spike immediately
            mu = X.mean(dim=0)
            mn = mu.norm()
            if mn > 1e-12:
                self.V[:, 0] = mu / mn
            if self.rank > 1:
                g = torch.Generator(device=X.device).manual_seed(42)
                R = torch.randn(self.d, self.rank, device=X.device, generator=g)
                R[:, 0] = self.V[:, 0]
                Q, _ = torch.linalg.qr(R)
                self.V.copy_(Q)
            Y = X @ self.V
            self.lam.copy_((Y * Y).mean(dim=0))
            self.tr_ema.copy_((X * X).sum(dim=-1).mean())
            self.steps = 1
            return

        g = 0.0 if self.steps == 1 else self.ema
        Y = X @ self.V                      # [B, k]
        b_lam = (Y * Y).mean(dim=0)         # [k] exact Rayleigh quotients
        b_tr = (X * X).sum(dim=-1).mean()   # scalar exact trace(F_batch)
        GV = (X.T @ Y) / float(B)           # [d, k] exact F_batch @ V

        prev_lam = self.lam.clone()
        self.lam.mul_(g).add_(b_lam, alpha=1.0 - g)
        self.tr_ema.mul_(g).add_(b_tr, alpha=1.0 - g)

        U = (g * prev_lam.unsqueeze(0)) * self.V + (1.0 - g) * GV
        if self.rank == 1:
            un = U[:, 0].norm()
            if un > 1e-12:
                self.V[:, 0] = U[:, 0] / un
        else:
            Q, _ = torch.linalg.qr(U)
            self.V.copy_(Q)

        mean_eig = self.tr_ema / float(self.d)
        thresh = self.cap_c * mean_eig
        if thresh > 1e-12:
            ratio = (thresh / self.lam.clamp_min(1e-12)).clamp_max(1.0)
            self.s.copy_(ratio.pow(self.cap_p))
            self.top_ratio = float((self.lam[0] / mean_eig).item())
        else:
            self.s.fill_(1.0)
            self.top_ratio = 1.0
        self.steps += 1


class CapMuon(torch.optim.Optimizer):
    """
    Muon optimizer with optional O(d_in) one-sided input-geometry capping (CapMuon).

    Args:
        params: Iterable of 2D hidden linear weight parameters.
        lr: Learning rate (scaled by sqrt(max(1, R/C)) per matrix).
        momentum: Momentum factor (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        weight_decay: Decoupled weight decay (default: 0.0).
        cap: Enable input-geometry capping (default: True). Set False for Plain Muon.
        cap_p: Cap exponent p in s_i = min(1, (c*mean/lam_i)^p).
               Recommended: 0.25 (safe universal default) or 0.50 (peak RL / high-LR).
        cap_c: Threshold multiplier c (default: 1.0).
        cap_ema: EMA decay for input power iteration (default: 0.9).
        cap_rank: Number of top input modes to cap (default: 1).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        cap: bool = True,
        cap_p: float = 0.25,
        cap_c: float = 1.0,
        cap_ema: float = 0.9,
        cap_rank: int = 1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.cap = cap
        self.cap_p = cap_p
        self.cap_c = cap_c
        self.cap_ema = cap_ema
        self.cap_rank = cap_rank
        self.trackers: dict[torch.nn.Parameter, CapTracker] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._capture_now = True

    def attach_hooks(self, model: nn.Module, target_names: Optional[Iterable[str]] = None) -> int:
        """
        Attach lightweight forward pre-hooks to managed 2D nn.Linear modules.

        Note on gradient checkpointing: call `set_capture(True)` before the forward
        pass where you want to sample activations (e.g. PPO epoch 0, microbatch 0),
        and `set_capture(False)` immediately before `.backward()` so activation
        recomputation does not double-update trackers.
        """
        managed = {p for g in self.param_groups for p in g["params"]}
        target_list = list(target_names) if target_names is not None else None
        attached = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight in managed:
                if target_list is not None and not any(t in name for t in target_list):
                    continue
                if mod.weight in self.trackers:
                    continue
                d_in = mod.weight.shape[1]
                tracker = CapTracker(
                    d=d_in,
                    device=mod.weight.device,
                    rank=self.cap_rank,
                    ema=self.cap_ema,
                    cap_c=self.cap_c,
                    cap_p=self.cap_p,
                )
                self.trackers[mod.weight] = tracker

                def _make_hook(tr: CapTracker):
                    def _hook(_mod, inputs):
                        if self.cap and self._capture_now and torch.is_grad_enabled():
                            if inputs and torch.is_tensor(inputs[0]):
                                tr.update(inputs[0])
                    return _hook

                self._handles.append(mod.register_forward_pre_hook(_make_hook(tracker)))
                attached += 1
        return attached

    def remove_hooks(self) -> None:
        """Detach all forward pre-hooks registered by `attach_hooks`."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def set_capture(self, flag: bool) -> None:
        """Toggle whether forward hooks update input eigenmode trackers."""
        self._capture_now = flag

    def get_cap_summary(self) -> dict[str, float]:
        """Return summary statistics (max/median lambda_1/mean, min s_1) across layers."""
        if not self.trackers:
            return {"max_ratio": 1.0, "med_ratio": 1.0, "min_s": 1.0}
        ratios = sorted(tr.top_ratio for tr in self.trackers.values())
        s_vals = [float(tr.s[0].item()) for tr in self.trackers.values()]
        mid = len(ratios) // 2
        med = ratios[mid] if len(ratios) % 2 == 1 else 0.5 * (ratios[mid - 1] + ratios[mid])
        return {
            "max_ratio": float(ratios[-1]),
            "med_ratio": float(med),
            "min_s": float(min(s_vals)),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"CapMuon only manages 2D weight matrices, got shape {tuple(p.shape)}")

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                Z = g.add(buf, alpha=mom) if nesterov else buf

                tr = self.trackers.get(p, None) if self.cap else None
                apply_cap = (tr is not None) and (tr.steps >= 2) and bool((tr.s < 1.0).any())

                if apply_cap:
                    V = tr.V.to(Z.dtype)
                    one_minus_s = (1.0 - tr.s).to(Z.dtype)
                    ZV = Z @ V
                    Z_in = Z - (ZV * one_minus_s.unsqueeze(0)) @ V.T
                else:
                    Z_in = Z

                Q = zeropower_via_newtonschulz5(Z_in, steps=ns_steps)

                if apply_cap:
                    V = tr.V.to(Q.dtype)
                    one_minus_s = (1.0 - tr.s).to(Q.dtype)
                    QV = Q @ V
                    Q = Q - (QV * one_minus_s.unsqueeze(0)) @ V.T

                R, C = p.shape
                scale = math.sqrt(max(1.0, R / C))
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(Q.to(p.dtype), alpha=-lr * scale)

        return loss


def split_muon_and_adam_params(
    model: nn.Module,
    exclude_keywords: tuple[str, ...] = ("embed", "lm_head", "wte", "wpe", "visual", "vision"),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Split trainable parameters into:
      - muon_params: 2D nn.Linear hidden weights (excluding embeddings, lm_head, vision)
      - adam_params: embeddings, lm_head, 1D norms/biases/conv1d/SSM scalars

    Safely deduplicates tied weights (e.g., tie_word_embeddings=True in Qwen3.5 / GPT-2).
    """
    excluded_ids: set[int] = set()
    for name, mod in model.named_modules():
        lname = name.lower()
        if isinstance(mod, nn.Embedding) or any(k in lname for k in exclude_keywords):
            for p in mod.parameters(recurse=False):
                excluded_ids.add(id(p))

    muon_params: list[nn.Parameter] = []
    muon_ids: set[int] = set()
    for name, mod in model.named_modules():
        lname = name.lower()
        if isinstance(mod, nn.Linear) and not any(k in lname for k in exclude_keywords):
            w = mod.weight
            if w.requires_grad and w.ndim == 2 and id(w) not in excluded_ids and id(w) not in muon_ids:
                muon_params.append(w)
                muon_ids.add(id(w))

    adam_params: list[nn.Parameter] = []
    seen = set(muon_ids)
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            adam_params.append(p)
            seen.add(id(p))

    return muon_params, adam_params
