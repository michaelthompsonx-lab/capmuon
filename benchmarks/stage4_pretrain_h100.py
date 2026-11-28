"""
Stage 4: From-Scratch Modern LLM Pretraining on FineWeb-Edu (H100 / A100)
  - Supports BOTH:
      (A) Standard 16-layer ModernLM (~250M, RoPE + RMSNorm + SwiGLU + Flash SDPA)
      (B) Weight-tied Looped Transformer (--looped 4: 4 unique layers looped 4x = 16 effective layers!)
  - Data: Deterministic pre-tokenized slice of HuggingFaceFW/fineweb-edu (sample-10BT):
      50M train tokens (1,500 steps x 32,768 tok/step) + 2.1M held-out val tokens.
  - Compares AdamW vs Plain Muon (p=0.00) vs CapMuon (p=0.15, 0.25, 0.35, 0.50, k=4)
    across standard (0.02), aggressive (0.05), and stress (0.10) LRs.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
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
    def __init__(self, d: int, device: torch.device, rank: int = 4, ema: float = 0.9, cap_c: float = 1.0, cap_p: float = 0.25):
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
        if B > 4096:
            stride = B // 4096
            X = X[::stride][:4096]
            B = X.size(0)
        if self.steps == 0:
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
        Y = X @ self.V
        b_lam = (Y * Y).mean(dim=0)
        b_tr = (X * X).sum(dim=-1).mean()
        GV = (X.T @ Y) / float(B)

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
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01,
                 cap=True, cap_p=0.25, cap_c=1.0, cap_ema=0.9, cap_rank=4):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))
        self.cap = cap
        self.cap_p = cap_p
        self.cap_c = cap_c
        self.cap_ema = cap_ema
        self.cap_rank = cap_rank
        self.trackers = {}
        self._handles = []
        self._capture_now = False

    def attach_hooks(self, model: nn.Module, track_only: bool = False) -> int:
        managed = {p for g in self.param_groups for p in g["params"]}
        attached = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight in managed:
                if mod.weight in self.trackers:
                    continue
                tr = CapTracker(mod.weight.shape[1], mod.weight.device, self.cap_rank, self.cap_ema, self.cap_c, self.cap_p)
                self.trackers[mod.weight] = tr

                def _make_hook(tracker):
                    def _hook(_m, inputs):
                        if (track_only or (self.cap and self.cap_p > 0.0)) and self._capture_now and torch.is_grad_enabled():
                            if inputs and torch.is_tensor(inputs[0]):
                                tracker.update(inputs[0])
                    return _hook

                self._handles.append(mod.register_forward_pre_hook(_make_hook(tr)))
                attached += 1
        return attached

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_cap_summary(self):
        if not self.trackers:
            return {"max_ratio": 1.0, "med_ratio": 1.0, "min_s": 1.0}
        ratios = sorted(tr.top_ratio for tr in self.trackers.values())
        s_vals = [float(tr.s[0].item()) for tr in self.trackers.values()]
        mid = len(ratios) // 2
        med = ratios[mid] if len(ratios) % 2 == 1 else 0.5 * (ratios[mid - 1] + ratios[mid])
        return {"max_ratio": float(ratios[-1]), "med_ratio": float(med), "min_s": float(min(s_vals))}

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, nesterov, ns_steps, wd = group["lr"], group["momentum"], group["nesterov"], group["ns_steps"], group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                Z = g.add(buf, alpha=mom) if nesterov else buf

                tr = self.trackers.get(p, None) if (self.cap and self.cap_p > 0.0) else None
                apply_cap = (tr is not None) and (tr.steps >= 2) and bool((tr.s < 1.0).any())
                if apply_cap:
                    V = tr.V.to(Z.dtype)
                    oms = (1.0 - tr.s).to(Z.dtype)
                    Z_in = Z - ((Z @ V) * oms.unsqueeze(0)) @ V.T
                else:
                    Z_in = Z

                Q = zeropower_via_newtonschulz5(Z_in, steps=ns_steps)

                if apply_cap:
                    V = tr.V.to(Q.dtype)
                    oms = (1.0 - tr.s).to(Q.dtype)
                    Q = Q - ((Q @ V) * oms.unsqueeze(0)) @ V.T

                R, C = p.shape
                scale = math.sqrt(max(1.0, R / C))
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(Q.to(p.dtype), alpha=-lr * scale)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def precompute_rope(seq_len: int, head_dim: int, theta: float = 10000.0, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs)[None, None, :, :], torch.sin(freqs)[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d2 = x.size(-1) // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :T, :], sin[:, :, :T, :])
        k = apply_rope(k, cos[:, :, :T, :], sin[:, :, :T, :])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLUMLP(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLUMLP(d, d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class ModernLM(nn.Module):
    def __init__(self, vocab_size=50304, d_model=1024, n_layers=16, n_heads=16, d_ff=2816, seq_len=1024, loops=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.loops = loops
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.register_buffer("rope_cos", torch.empty(1), persistent=False)
        self.register_buffer("rope_sin", torch.empty(1), persistent=False)
        self._init_weights(n_layers * loops)

    def _init_weights(self, eff_layers: int):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                std = 0.02 / math.sqrt(2 * eff_layers) if (name.endswith("o_proj") or name.endswith("down_proj")) else 0.02
                nn.init.normal_(mod.weight, mean=0.0, std=std)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)

    def init_rope(self, device):
        cos, sin = precompute_rope(self.seq_len, self.d_model // self.blocks[0].attn.n_heads, device=device)
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.embed(idx)
        for _ in range(self.loops):
            for blk in self.blocks:
                x = blk(x, self.rope_cos, self.rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def build_or_load_tokens(cache_path: str, n_train_tokens: int, n_val_tokens: int):
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=True)
        if data["train"].numel() >= n_train_tokens and data["val"].numel() >= n_val_tokens:
            print(f"Loaded cached FineWeb-Edu tokens from {cache_path} "
                  f"(train={data['train'].numel()/1e6:.2f}M, val={data['val'].numel()/1e6:.2f}M)", flush=True)
            return data["train"][:n_train_tokens], data["val"][:n_val_tokens]

    print(f"Streaming & tokenizing FineWeb-Edu ({(n_train_tokens + n_val_tokens)/1e6:.1f}M tokens)...", flush=True)
    t0 = time.time()
    from datasets import load_dataset
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token
        encode_fn = lambda s: enc.encode_ordinary(s)
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        eot = tok.eos_token_id
        encode_fn = lambda s: tok.encode(s, add_special_tokens=False)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    total_needed = n_train_tokens + n_val_tokens + 4096
    buf = np.empty(total_needed, dtype=np.uint16)
    pos = 0
    docs = 0
    for row in ds:
        ids = encode_fn(row["text"])
        ids.append(eot)
        n = len(ids)
        if pos + n >= total_needed:
            take = total_needed - pos
            buf[pos:pos + take] = ids[:take]
            pos += take
            break
        buf[pos:pos + n] = ids
        pos += n
        docs += 1
        if docs % 10000 == 0:
            print(f"  tokenized {docs} docs -> {pos/1e6:.2f}M / {total_needed/1e6:.1f}M tokens ({time.time()-t0:.1f}s)", flush=True)

    arr = torch.from_numpy(buf.astype(np.int32))
    val_toks = arr[:n_val_tokens].clone()
    train_toks = arr[n_val_tokens:n_val_tokens + n_train_tokens].clone()
    torch.save({"train": train_toks, "val": val_toks}, cache_path)
    print(f"Cached {train_toks.numel()/1e6:.2f}M train + {val_toks.numel()/1e6:.2f}M val tokens in {time.time()-t0:.1f}s", flush=True)
    return train_toks, val_toks


@torch.no_grad()
def evaluate_val(model: nn.Module, val_toks: torch.Tensor, seq_len: int, batch_size: int, device: str) -> float:
    model.eval()
    n_seqs = (val_toks.numel() - 1) // seq_len
    losses = []
    for start in range(0, n_seqs - batch_size + 1, batch_size):
        idx_list = [val_toks[i * seq_len:(i + 1) * seq_len] for i in range(start, start + batch_size)]
        tgt_list = [val_toks[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(start, start + batch_size)]
        x = torch.stack(idx_list).to(device, dtype=torch.long, non_blocking=True)
        y = torch.stack(tgt_list).to(device, dtype=torch.long, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def run_arm(arm: dict, init_state: dict, train_toks: torch.Tensor, val_toks: torch.Tensor, cfg: dict):
    device = "cuda"
    torch.manual_seed(cfg["SEED"])
    torch.cuda.manual_seed_all(cfg["SEED"])
    torch.cuda.reset_peak_memory_stats()

    model = ModernLM(
        vocab_size=cfg["VOCAB_SIZE"], d_model=cfg["D_MODEL"], n_layers=cfg["N_LAYERS"],
        n_heads=cfg["N_HEADS"], d_ff=cfg["D_FF"], seq_len=cfg["SEQ_LEN"], loops=cfg["LOOPS"],
    ).to(device)
    model.load_state_dict(init_state, strict=True)
    model.init_rope(device)

    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    # Loop-adjusted EMA so L loop calls per forward pass equal effective 0.9 decay per optimizer step
    cap_ema = 0.9 ** (1.0 / float(max(1, cfg["LOOPS"])))
    is_adam_only = arm["type"] == "adamw"
    if is_adam_only:
        opt_adam = torch.optim.AdamW(
            model.parameters(), lr=arm["lr"], betas=(0.9, 0.95), weight_decay=arm.get("wd", 0.1), fused=True
        )
        opt_muon = None
    else:
        opt_adam = torch.optim.AdamW(
            adam_params, lr=arm["lr_adam"], betas=(0.9, 0.95), weight_decay=arm.get("wd_adam", 0.01), fused=True
        )
        opt_muon = CapMuon(
            muon_params, lr=arm["lr_muon"], momentum=0.95, nesterov=True, ns_steps=5,
            weight_decay=arm.get("wd_muon", 0.01),
            cap=arm["cap"], cap_p=arm["cap_p"], cap_c=1.0, cap_ema=cap_ema, cap_rank=arm.get("cap_rank", 4),
        )
        if arm["cap"] and arm["cap_p"] > 0.0:
            opt_muon.attach_hooks(model, track_only=False)
        elif arm.get("track_spikes", False):
            opt_muon.attach_hooks(model, track_only=True)

    print("\n" + "=" * 96, flush=True)
    print(f"PRETRAIN ARM: {arm['name']} | loops={cfg['LOOPS']} cap_ema={cap_ema:.4f} | {arm}", flush=True)
    print("=" * 96, flush=True)

    seq_len, micro_bs, grad_accum = cfg["SEQ_LEN"], cfg["MICRO_BS"], cfg["GRAD_ACCUM"]
    steps, warmup, eval_every = cfg["STEPS"], cfg["WARMUP"], cfg["EVAL_EVERY"]

    def lr_mult(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        prog = float(step - warmup) / float(max(1, steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    init_val = evaluate_val(model, val_toks, seq_len, micro_bs, device)
    print(f"[Step 0000/{steps}] init_val_loss={init_val:.4f} (ppl={math.exp(min(20, init_val)):.1f})", flush=True)

    hist = {
        "arm": arm,
        "steps": [], "train_loss": [], "val_steps": [0], "val_loss": [init_val],
        "max_lam": [], "med_lam": [], "min_s": [], "step_ms": [],
        "diverged": False,
    }

    n_seqs = (train_toks.numel() - 1) // seq_len
    seq_ptr = 0
    t_arm = time.time()
    step_times = []

    for step in range(1, steps + 1):
        t0 = time.time()
        mult = lr_mult(step - 1)
        if is_adam_only:
            for g in opt_adam.param_groups:
                g["lr"] = arm["lr"] * mult
            opt_adam.zero_grad(set_to_none=True)
        else:
            for g in opt_adam.param_groups:
                g["lr"] = arm["lr_adam"] * mult
            for g in opt_muon.param_groups:
                g["lr"] = arm["lr_muon"] * mult
            opt_adam.zero_grad(set_to_none=True)
            opt_muon.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for mb in range(grad_accum):
            if opt_muon is not None:
                opt_muon._capture_now = (mb == 0)
            if seq_ptr + micro_bs > n_seqs:
                seq_ptr = 0
            idx_list = [train_toks[i * seq_len:(i + 1) * seq_len] for i in range(seq_ptr, seq_ptr + micro_bs)]
            tgt_list = [train_toks[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(seq_ptr, seq_ptr + micro_bs)]
            seq_ptr += micro_bs
            x = torch.stack(idx_list).to(device, dtype=torch.long, non_blocking=True)
            y = torch.stack(tgt_list).to(device, dtype=torch.long, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            if opt_muon is not None:
                opt_muon._capture_now = False
            (loss / grad_accum).backward()
            loss_accum += float(loss.item()) / grad_accum

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt_adam.step()
        if opt_muon is not None:
            opt_muon.step()

        torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000.0
        step_times.append(dt_ms)

        if not math.isfinite(loss_accum) or loss_accum > 15.0:
            print(f"  !!! DIVERGED at step {step}: loss={loss_accum}", flush=True)
            hist["diverged"] = True
            break

        if step == 1 or step % 50 == 0 or step == steps:
            cs = opt_muon.get_cap_summary() if opt_muon is not None else {"max_ratio": 1.0, "med_ratio": 1.0, "min_s": 1.0}
            hist["steps"].append(step)
            hist["train_loss"].append(loss_accum)
            hist["max_lam"].append(cs["max_ratio"])
            hist["med_lam"].append(cs["med_ratio"])
            hist["min_s"].append(cs["min_s"])
            hist["step_ms"].append(float(np.mean(step_times[-50:])))
            tok_s = (micro_bs * grad_accum * seq_len) / (hist["step_ms"][-1] / 1000.0)
            extra = f" | max_lam={cs['max_ratio']:6.1f}x med={cs['med_ratio']:5.1f}x min_s={cs['min_s']:.2f}" if opt_muon is not None else ""
            print(f"[Step {step:04d}/{steps}] tr_loss={loss_accum:.4f} | gn={grad_norm:5.2f} | {hist['step_ms'][-1]:5.1f}ms ({tok_s/1e3:5.1f}k tok/s){extra}", flush=True)

        if step % eval_every == 0 or step == steps:
            vl = evaluate_val(model, val_toks, seq_len, micro_bs, device)
            hist["val_steps"].append(step)
            hist["val_loss"].append(vl)
            print(f"  >>> VAL @ step {step:04d}: val_loss={vl:.4f} (ppl={math.exp(min(20, vl)):.2f})", flush=True)

    hist["total_time_s"] = time.time() - t_arm
    hist["mean_step_ms"] = float(np.mean(step_times[5:])) if len(step_times) > 5 else float(np.mean(step_times))
    hist["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    hist["final_val_loss"] = hist["val_loss"][-1]
    hist["best_val_loss"] = min(hist["val_loss"])
    if opt_muon is not None:
        opt_muon.remove_hooks()
    del model, opt_adam, opt_muon
    torch.cuda.empty_cache()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--d-ff", type=int, default=2816)
    ap.add_argument("--loops", type=int, default=1, help="Number of recurrent loops over n_layers (e.g. --n-layers 4 --loops 4)")
    ap.add_argument("--cache", default="/content/fineweb_edu_52m.pt")
    ap.add_argument("--out", default="/content/stage4_pretrain_h100_results.json")
    ap.add_argument("--arms", default="all")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    toks_per_step = args.micro_bs * args.grad_accum * args.seq_len
    n_train_toks = args.steps * toks_per_step + args.seq_len * 64
    n_val_toks = 2_097_152

    cfg = {
        "VOCAB_SIZE": 50304, "D_MODEL": args.d_model, "N_LAYERS": args.n_layers,
        "N_HEADS": args.n_heads, "D_FF": args.d_ff, "SEQ_LEN": args.seq_len,
        "LOOPS": args.loops, "MICRO_BS": args.micro_bs, "GRAD_ACCUM": args.grad_accum,
        "TOKS_PER_STEP": toks_per_step, "STEPS": args.steps,
        "WARMUP": max(25, args.steps // 20), "EVAL_EVERY": args.eval_every, "SEED": 42,
        "GPU": torch.cuda.get_device_name(0),
    }
    print(f"GPU: {cfg['GPU']} | loops={args.loops} (eff_layers={args.n_layers * args.loops}) | "
          f"{toks_per_step:,} tok/step x {args.steps} steps = {args.steps * toks_per_step / 1e6:.1f}M tokens", flush=True)

    train_toks, val_toks = build_or_load_tokens(args.cache, n_train_toks, n_val_toks)

    torch.manual_seed(cfg["SEED"])
    ref_model = ModernLM(cfg["VOCAB_SIZE"], cfg["D_MODEL"], cfg["N_LAYERS"], cfg["N_HEADS"], cfg["D_FF"], cfg["SEQ_LEN"], cfg["LOOPS"])
    n_total = sum(p.numel() for p in ref_model.parameters())
    n_muon = sum(p.numel() for n, p in ref_model.named_parameters() if p.ndim == 2 and "embed" not in n and "lm_head" not in n)
    cfg["N_PARAMS_TOTAL"] = n_total
    cfg["N_PARAMS_MUON"] = n_muon
    print(f"ModernLM params: total={n_total/1e6:.1f}M | 2D Muon hidden={n_muon/1e6:.1f}M", flush=True)
    init_state = {k: v.cpu().clone() for k, v in ref_model.state_dict().items()}
    del ref_model

    all_arms = [
        {"name": "adamw_1e3",          "type": "adamw",   "lr": 1e-3, "wd": 0.1},
        {"name": "adamw_2e3",          "type": "adamw",   "lr": 2e-3, "wd": 0.1},
        {"name": "muon_lr002_p000",    "type": "capmuon", "lr_muon": 0.02, "lr_adam": 2e-3, "cap": False, "cap_p": 0.00, "cap_rank": 4, "track_spikes": True},
        {"name": "cap_lr002_p015",     "type": "capmuon", "lr_muon": 0.02, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.15, "cap_rank": 4},
        {"name": "cap_lr002_p025",     "type": "capmuon", "lr_muon": 0.02, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.25, "cap_rank": 4},
        {"name": "cap_lr002_p035",     "type": "capmuon", "lr_muon": 0.02, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.35, "cap_rank": 4},
        {"name": "muon_lr005_p000",    "type": "capmuon", "lr_muon": 0.05, "lr_adam": 2e-3, "cap": False, "cap_p": 0.00, "cap_rank": 4},
        {"name": "cap_lr005_p025",     "type": "capmuon", "lr_muon": 0.05, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.25, "cap_rank": 4},
        {"name": "cap_lr005_p035",     "type": "capmuon", "lr_muon": 0.05, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.35, "cap_rank": 4},
        {"name": "cap_lr005_p050",     "type": "capmuon", "lr_muon": 0.05, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.50, "cap_rank": 4},
        {"name": "muon_lr010_p000",    "type": "capmuon", "lr_muon": 0.10, "lr_adam": 2e-3, "cap": False, "cap_p": 0.00, "cap_rank": 4},
        {"name": "cap_lr010_p035",     "type": "capmuon", "lr_muon": 0.10, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.35, "cap_rank": 4},
        {"name": "cap_lr010_p050",     "type": "capmuon", "lr_muon": 0.10, "lr_adam": 2e-3, "cap": True,  "cap_p": 0.50, "cap_rank": 4},
    ]
    if args.arms != "all":
        want = set(args.arms.split(","))
        arms = [a for a in all_arms if a["name"] in want]
    else:
        arms = all_arms

    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out)).get("results", {})
            print(f"Resumed {len(results)} completed arms from {args.out}", flush=True)
        except Exception:
            pass

    for arm in arms:
        if arm["name"] in results:
            print(f"Skipping completed {arm['name']}", flush=True)
            continue
        results[arm["name"]] = run_arm(arm, init_state, train_toks, val_toks, cfg)
        with open(args.out, "w") as f:
            json.dump({"config": cfg, "results": results}, f, indent=2)

    print("\n" + "=" * 102, flush=True)
    print(f"STAGE 4 PRETRAIN SUMMARY — ModernLM {n_total/1e6:.0f}M (loops={args.loops}) on FineWeb-Edu ({args.steps * toks_per_step / 1e6:.1f}M tok, {cfg['GPU']})", flush=True)
    print("=" * 102, flush=True)
    for name, r in results.items():
        div = " [DIVERGED]" if r.get("diverged") else ""
        ml = r["max_lam"][-1] if r["max_lam"] else 1.0
        print(
            f"  {name:<18} | finVal={r['final_val_loss']:.4f} (ppl={math.exp(min(20, r['final_val_loss'])):6.2f}) "
            f"| bestVal={r['best_val_loss']:.4f} | maxLam={ml:6.1f}x | {r['mean_step_ms']:5.1f}ms/st "
            f"| vram={r['peak_vram_gb']:.1f}GB | t={r['total_time_s']:.0f}s{div}",
            flush=True,
        )


if __name__ == "__main__":
    main()
