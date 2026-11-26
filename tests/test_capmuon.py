"""Unit tests for CapMuon optimizer, CapTracker convergence, and diagnostics."""
import math
import torch
import torch.nn as nn

from capmuon import (
    CapMuon,
    CapTracker,
    inspect_linear_eigenspectra,
    split_muon_and_adam_params,
    zeropower_via_newtonschulz5,
)


def test_newton_schulz_orthogonality():
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    Q = zeropower_via_newtonschulz5(G, steps=5).float()
    sv = torch.linalg.svdvals(Q)
    # Quintic 5-step Muon NS polynomial maps non-zero singular values into ~[0.65, 1.35]
    assert sv.min().item() > 0.65 and sv.max().item() < 1.35, f"Unexpected NS5 singular values: {sv.min():.3f}..{sv.max():.3f}"
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    polar_exact = U @ Vh
    cos = (Q * polar_exact).sum().item() / (Q.norm().item() * polar_exact.norm().item())
    assert cos > 0.98, f"Expected NS5 cosine > 0.98 vs exact polar factor, got {cos:.4f}"


def test_cap_tracker_converges_to_exact_top_eigenpair():
    torch.manual_seed(42)
    d = 64
    B = 2048
    # Construct spiked inputs with known dominant rank-1 common-mode direction
    v_true = torch.randn(d)
    v_true = v_true / v_true.norm()
    X = 0.2 * torch.randn(B, d) + 5.0 * v_true.unsqueeze(0)

    F_exact = (X.T @ X) / B
    evals, evecs = torch.linalg.eigh(F_exact)
    lam1_exact = evals[-1].item()
    v1_exact = evecs[:, -1]
    mean_exact = evals.mean().item()

    tr = CapTracker(d=d, device=torch.device("cpu"), rank=1, ema=0.5, cap_c=1.0, cap_p=0.5)
    for _ in range(12):
        tr.update(X)

    cos = abs(float((tr.V[:, 0] @ v1_exact).item()))
    assert cos > 0.9999, f"Expected |cos(v1, v1_exact)| > 0.9999, got {cos}"
    rel_lam = abs(tr.lam[0].item() - lam1_exact) / lam1_exact
    assert rel_lam < 1e-3, f"Expected relative lam1 error < 1e-3, got {rel_lam}"
    expected_s = min(1.0, math.sqrt(mean_exact / lam1_exact))
    assert abs(tr.s[0].item() - expected_s) < 1e-2
    assert 0.0 < tr.s[0].item() <= 1.0


def test_cap_p0_matches_plain_muon():
    torch.manual_seed(7)
    m1 = nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU(), nn.Linear(32, 16, bias=False))
    m2 = nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU(), nn.Linear(32, 16, bias=False))
    m2.load_state_dict(m1.state_dict())

    opt_plain = CapMuon(m1.parameters(), lr=1e-2, cap=False)
    opt_p0 = CapMuon(m2.parameters(), lr=1e-2, cap=True, cap_p=0.0)
    opt_p0.attach_hooks(m2)

    x = torch.randn(64, 32) + 3.0
    for _ in range(5):
        opt_plain.zero_grad()
        opt_p0.zero_grad()
        m1(x).pow(2).mean().backward()
        m2(x).pow(2).mean().backward()
        opt_plain.step()
        opt_p0.step()

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "cap_p=0 should match Plain Muon identically"
    opt_p0.remove_hooks()


def test_capping_reduces_dominant_mode_activation_step():
    torch.manual_seed(11)
    d = 64
    v = torch.ones(d) / math.sqrt(d)
    x = 0.1 * torch.randn(256, d) + 4.0 * v.unsqueeze(0)

    l_plain = nn.Linear(d, d, bias=False)
    l_cap = nn.Linear(d, d, bias=False)
    l_cap.load_state_dict(l_plain.state_dict())

    opt_p = CapMuon([l_plain.weight], lr=0.05, cap=False)
    opt_c = CapMuon([l_cap.weight], lr=0.05, cap=True, cap_p=0.5)
    opt_c.attach_hooks(l_cap)

    for _ in range(4):
        w0_p = l_plain.weight.detach().clone()
        w0_c = l_cap.weight.detach().clone()
        opt_p.zero_grad()
        opt_c.zero_grad()
        l_plain(x).pow(2).mean().backward()
        l_cap(x).pow(2).mean().backward()
        opt_p.step()
        opt_c.step()

    # Check activation-space shift along dominant mode v is strictly bounded by CapMuon
    dw_p = (l_plain.weight.detach() - w0_p) @ v
    dw_c = (l_cap.weight.detach() - w0_c) @ v
    assert dw_c.norm().item() < 0.25 * dw_p.norm().item()
    opt_c.remove_hooks()


def test_param_split_tied_embeddings_and_vision():
    class MiniVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(8, 16)
            for p in self.visual.parameters():
                p.requires_grad = False
            self.embed_tokens = nn.Embedding(32, 16)
            self.hidden = nn.Linear(16, 16, bias=False)
            self.norm = nn.LayerNorm(16)
            self.lm_head = nn.Linear(16, 32, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

        def forward(self, ids):
            return self.lm_head(self.norm(self.hidden(self.embed_tokens(ids))))

    m = MiniVLM()
    muon_p, adam_p = split_muon_and_adam_params(m)
    assert len(muon_p) == 1 and muon_p[0] is m.hidden.weight
    assert m.lm_head.weight in set(adam_p)
    assert m.visual.weight not in set(muon_p) and m.visual.weight not in set(adam_p)

    # Also test exact eigenspectrum diagnostic
    ids = torch.randint(0, 32, (8, 16))
    spec = inspect_linear_eigenspectra(m, lambda mod: mod(ids))
    assert "hidden" in spec and spec["hidden"]["ratio"] >= 1.0
