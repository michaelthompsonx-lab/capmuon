"""
Stage 5A: Gemma 4 Scale-Up Spike Anatomy Probe (H100 80 GB / A100 80 GB)
  - Probes:
      1. google/gemma-4-E4B-it       (8.0B total, ~4.5B active dense text backbone)
      2. google/gemma-4-26B-A4B-it   (25.8B total MoE, ~4B active per token — routers + experts!)
  - Decomposes Raw vs Centered vs No-Sink (drop top 2% tokens) vs Both + cos(v1, mu), Top1%T, max|v1|, n_eff.
"""
import argparse
import gc
import json
import time
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

PROMPTS = [
    "Solve step by step: A bookstore sells notebooks for $4 each and pens for $2 each. Maya buys 7 notebooks and 9 pens, then gets a 15% discount. How much does she pay?",
    "Using the numbers [8, 13, 25, 4] each at most once with +, -, *, /, reach the target 173. Show your reasoning clearly.",
    "Explain why attention sinks and non-zero activation means emerge in deep transformer residual streams and how they distort the input covariance spectrum.",
    "Write a concise Python function implementing a matrix-free streaming top-k subspace iteration for tracking dominant eigenvalues of streaming activation batches.",
    "A train leaves station A at 60 mph and another leaves station B 210 miles away at 80 mph traveling toward each other. When and where do they meet?",
    "Compare dense SwiGLU transformers with sparse Mixture-of-Experts routers in terms of activation covariance anisotropy and gradient conditioning.",
    "Prove that for any symmetric positive semidefinite matrix F, replacing F^(-p) with min(1, (c*mean/lam)^p) strictly bounds operator norm by 1.",
    "In reinforcement learning with verifiable rewards (GRPO), explain how clipping ratios and reference KL divergence prevent policy collapse.",
]


@torch.no_grad()
def analyze_matrix(X: torch.Tensor):
    N, d = X.shape
    if N < 16 or d < 64:
        return None
    if N > 2048:
        X = X[:: (N // 2048)][:2048]
        N = X.shape[0]

    def top_eig(M):
        tr = (M * M).sum(dim=-1).mean()
        mean_eig = (tr / d).clamp_min(1e-12)
        if M.shape[0] <= M.shape[1]:
            G = (M @ M.T) / M.shape[0]
            evals, evecs = torch.linalg.eigh(G)
            lam1 = evals[-1].clamp_min(1e-12)
            u1 = evecs[:, -1]
            v1 = M.T @ u1
            v1 = v1 / v1.norm().clamp_min(1e-12)
        else:
            C = (M.T @ M) / M.shape[0]
            evals, evecs = torch.linalg.eigh(C)
            lam1 = evals[-1].clamp_min(1e-12)
            v1 = evecs[:, -1]
        return float((lam1 / mean_eig).item()), v1

    raw_ratio, v1 = top_eig(X)
    mu = X.mean(dim=0)
    mu_n = mu.norm().clamp_min(1e-12)
    cos_mu = float(torch.abs((v1 * (mu / mu_n)).sum()).item())

    Xc = X - mu.unsqueeze(0)
    cent_ratio, _ = top_eig(Xc)

    norms = (X * X).sum(dim=-1)
    k_top1 = max(1, int(0.01 * N))
    top1_energy = float(norms.topk(k_top1).values.sum().item() / norms.sum().clamp_min(1e-12).item())

    k_drop = max(1, int(0.02 * N))
    keep_idx = torch.argsort(norms)[: N - k_drop]
    X_nosnk = X[keep_idx]
    nosnk_ratio, _ = top_eig(X_nosnk)
    both_ratio, _ = top_eig(X_nosnk - X_nosnk.mean(dim=0, keepdim=True))

    max_v = float(v1.abs().max().item())
    n_eff = float(1.0 / (v1.pow(4).sum().clamp_min(1e-12).item()))
    return {
        "d": d, "N": N,
        "raw": round(raw_ratio, 1), "cent": round(cent_ratio, 1),
        "nosnk": round(nosnk_ratio, 1), "both": round(both_ratio, 1),
        "cos_mu": round(cos_mu, 4), "top1_tok_pct": round(top1_energy * 100, 1),
        "max_v": round(max_v, 3), "n_eff": round(n_eff, 1),
    }


def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    is_vlm = hasattr(cfg, "vision_config") or "ConditionalGeneration" in str(getattr(cfg, "architectures", []))
    if is_vlm and AutoModelForImageTextToText is not None:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    return model, tok


def probe_model(model_id: str, max_layers_sample: int = 6):
    print(f"\n==================== GEMMA 4 SCALE PROBE: {model_id} ====================", flush=True)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(model_id)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Loaded {model_id} in {time.time()-t0:.1f}s | peak_vram={vram:.1f}GB", flush=True)

    linears = []
    for name, mod in model.named_modules():
        ln = name.lower()
        if not isinstance(mod, nn.Linear):
            continue
        if any(k in ln for k in ("vision", "visual", "audio", "lm_head", "embed")):
            continue
        linears.append((name, mod))

    layer_nums = sorted({
        int(tok_s) for name, _ in linears for tok_s in name.split(".") if tok_s.isdigit()
    })
    if len(layer_nums) > max_layers_sample:
        picks = sorted({layer_nums[int(i * (len(layer_nums) - 1) / (max_layers_sample - 1))] for i in range(max_layers_sample)})
    else:
        picks = layer_nums
    pick_set = {f".{p}." for p in picks}
    chosen = [(n, m) for n, m in linears if any(ps in f".{n}." for ps in pick_set)]
    filtered = []
    for n, m in chosen:
        if "experts." in n:
            parts = n.split("experts.")[1].split(".")
            if parts[0].isdigit() and int(parts[0]) >= 2:
                continue
        filtered.append((n, m))

    print(f"Total text 2D linears: {len(linears)} | Probing {len(filtered)} across layers {picks}", flush=True)

    captured = {}
    handles = []
    for name, mod in filtered:
        def _mk(nm):
            def _hk(_m, inp):
                if inp and torch.is_tensor(inp[0]):
                    x = inp[0].detach()
                    if x.ndim >= 2 and x.shape[0] > 0:
                        captured[nm] = x.reshape(-1, x.shape[-1]).float().to("cuda:0")
            return _hk
        handles.append(mod.register_forward_pre_hook(_mk(name)))

    enc = tok(PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=256)
    first_dev = next(model.parameters()).device
    enc = {k: v.to(first_dev) for k, v in enc.items()}
    with torch.no_grad():
        _ = model(**enc)

    for h in handles:
        h.remove()

    rows = []
    for name, X in captured.items():
        res = analyze_matrix(X)
        if res is not None:
            res["name"] = name
            rows.append(res)
    rows.sort(key=lambda r: r["raw"], reverse=True)

    print(f"{'Layer':<48} {'d':>6} {'Raw':>8} {'Cent':>8} {'NoSnk':>8} {'Both':>8} | {'cos':>5} {'Top1%':>6} {'max|v|':>6} {'n_eff':>6}", flush=True)
    print("-" * 116, flush=True)
    for r in rows[:20]:
        short = r["name"].replace("model.language_model.layers.", "L").replace("model.layers.", "L")
        print(
            f"{short:<48} {r['d']:>6} {r['raw']:>7.0f}x {r['cent']:>7.0f}x {r['nosnk']:>7.0f}x {r['both']:>7.0f}x | "
            f"{r['cos_mu']:>5.2f} {r['top1_tok_pct']:>5.1f}% {r['max_v']:>6.3f} {r['n_eff']:>6.1f}",
            flush=True,
        )

    del model, captured
    gc.collect()
    torch.cuda.empty_cache()
    return {"model": model_id, "layers_probed": len(rows), "top20": rows[:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/gemma-4-E4B-it,google/gemma-4-26B-A4B-it")
    ap.add_argument("--out", default="/content/stage5a_gemma4_scale_anatomy.json")
    args = ap.parse_args()

    out = {}
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            out[mid] = probe_model(mid)
        except Exception as e:
            print(f"ERROR on {mid}: {e}", flush=True)
            out[mid] = {"error": str(e)}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print("\n=== STAGE 5A GEMMA 4 SCALE ANATOMY COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
