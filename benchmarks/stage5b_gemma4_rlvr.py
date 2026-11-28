"""
Stage 2 Colab L4 Cross-Architecture RLVR Benchmark: google/gemma-4-E2B-it
  - 35-layer 2026 Google Gemma 4 architecture (measured 1,979.4x spike on L34.self_attn.o_proj
    and 1,449.4x / 94.4% raw energy on L34.per_layer_input_gate!)
  - Harder 4-number Countdown (nums 1..25, targets 15..250) so Gemma 4 E2B starts in the
    ~25-50% learning sweet spot instead of saturating at 62.5% on 3-num puzzles.
  - Freezes unused vision/audio towers + static 2.35B-param PLE lookup table (`embed_tokens_per_layer`)
    so 100% of the 35-layer text backbone 2D Linear matrices train in full bf16 inside L4 23.7GB VRAM
    with zero-VRAM CPU-swap DeepSeek-R1 k3 ref KL!
  - Compares Plain Muon (p=0.0) vs CapMuon (p=0.25 k=4, p=0.35 k=4, p=0.50 k=4) [+ AdamW if VRAM fits].
"""
import argparse
import ast
import gc
import json
import math
import os
import random
import re
import time
from collections import Counter
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from capmuon import CapMuon, split_muon_and_adam_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16


def generate_hard_countdown(rng: random.Random, n_nums: int = 4) -> dict:
    hi = 25 if n_nums == 4 else 18
    for _ in range(500):
        nums = [rng.randint(1, hi) for _ in range(n_nums)]
        items = [(Fraction(x), str(x)) for x in nums]
        rng.shuffle(items)
        ok = True
        while len(items) > 1:
            a_val, a_expr = items.pop(rng.randrange(len(items)))
            b_val, b_expr = items.pop(rng.randrange(len(items)))
            ops = ["+", "-", "*"] + (["/"] if b_val != 0 and a_val % b_val == 0 else [])
            op = rng.choice(ops)
            if op == "+":
                val = a_val + b_val
            elif op == "-":
                if a_val < b_val:
                    a_val, b_val, a_expr, b_expr = b_val, a_val, b_expr, a_expr
                val = a_val - b_val
            elif op == "*":
                val = a_val * b_val
            else:
                val = a_val / b_val
            if val <= 0 or val > 400 or val.denominator != 1:
                ok = False
                break
            items.append((val, f"({a_expr} {op} {b_expr})"))
        if ok:
            target = int(items[0][0])
            if 20 <= target <= 250:
                sh = list(nums)
                rng.shuffle(sh)
                return {"nums": sh, "target": target, "ref_expr": items[0][1]}
    return {"nums": [2, 6, 9, 4], "target": 46, "ref_expr": "((9 * 6) - (4 * 2))"}


def make_dataset(n: int, seed: int, n_nums: int = 4) -> list[dict]:
    rng = random.Random(seed)
    return [generate_hard_countdown(rng, n_nums=n_nums) for _ in range(n)]


SYSTEM_PROMPT = (
    "You are a concise arithmetic solver. Using EVERY given number exactly once "
    "with +, -, *, / and parentheses, form an expression equal to the target. "
    "Show brief arithmetic checks inside <reasoning>...</reasoning> and output ONLY "
    "the final expression inside <answer>...</answer>."
)


def format_prompt(tokenizer, puzzle: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Numbers: [3, 5, 8]\nTarget: 37"},
        {"role": "assistant", "content": "<reasoning>Try 8 * 5 = 40. Then 40 - 3 = 37. Matches 37.</reasoning>\n<answer>(8 * 5) - 3</answer>"},
        {"role": "user", "content": "Numbers: [2, 6, 9, 4]\nTarget: 46"},
        {"role": "assistant", "content": "<reasoning>Try 9 * 6 = 54. Remaining [2, 4] give 4 * 2 = 8. 54 - 8 = 46. Matches 46.</reasoning>\n<answer>(9 * 6) - (4 * 2)</answer>"},
        {"role": "user", "content": f"Numbers: {puzzle['nums']}\nTarget: {puzzle['target']}"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _eval_ast(node, used):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, used)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, float) and not node.value.is_integer():
            raise ValueError()
        iv = int(node.value)
        used.append(iv)
        return Fraction(iv)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_ast(node.operand, used)
    if isinstance(node, ast.BinOp):
        l, r = _eval_ast(node.left, used), _eval_ast(node.right, used)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            if r == 0:
                raise ZeroDivisionError()
            return l / r
    raise ValueError()


def score_countdown(response: str, nums: list[int], target: int) -> tuple[float, bool]:
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return 0.0, False
    expr = matches[-1].strip().split("=")[0].strip()
    expr = expr.replace("\u00d7", "*").replace("\u00f7", "/").replace("\u2212", "-")
    if not expr or len(expr) > 80:
        return 0.0, False
    try:
        used = []
        val = _eval_ast(ast.parse(expr, mode="eval"), used)
        if Counter(used) == Counter(nums) and val == Fraction(target):
            return 1.0, True
    except Exception:
        pass
    return 0.0, False


def load_gemma4(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    is_vlm = hasattr(cfg, "vision_config") or "ConditionalGeneration" in str(getattr(cfg, "architectures", []))
    if is_vlm and AutoModelForImageTextToText is not None:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPE, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, trust_remote_code=True)

    # Freeze unused multimodal towers (vision/audio) + static 2.35B PLE lookup table
    # and base token embeddings so 100% of VRAM goes to training the 35-layer text backbone!
    frozen_n, train_n = 0, 0
    for name, p in model.named_parameters():
        ln = name.lower()
        if any(k in ln for k in ("vision", "visual", "audio", "embed_tokens_per_layer", "embed_tokens", "lm_head")):
            p.requires_grad = False
            frozen_n += p.numel()
        else:
            p.requires_grad = True
            train_n += p.numel()

    model.to(DEVICE)
    model.config.use_cache = False
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = False

    muon_p, adam_p = split_muon_and_adam_params(model)
    print(f"Loaded {model_id} | Trainable: {train_n/1e6:.1f}M | Frozen (vision/audio/embeds): {frozen_n/1e6:.1f}M", flush=True)
    print(f"  Muon 2D matrices: {len(muon_p)} ({sum(p.numel() for p in muon_p)/1e6:.1f}M) | AdamW 1D/norms: {len(adam_p)} ({sum(p.numel() for p in adam_p)/1e6:.2f}M)", flush=True)
    return model, tok


def _set_grad_ckpt(model, enabled):
    if enabled:
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "activation_checkpointing_disable"):
            model.activation_checkpointing_disable()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()


def compute_completion_logprobs(model, seq_ids, attn_mask, prompt_len):
    out = model(input_ids=seq_ids, attention_mask=attn_mask, use_cache=False)
    comp_logits = out.logits[:, prompt_len - 1 : -1, :].float()
    comp_targets = seq_ids[:, prompt_len:]
    B_m, T_c, V = comp_logits.shape
    return -F.cross_entropy(
        comp_logits.reshape(B_m * T_c, V), comp_targets.reshape(B_m * T_c), reduction="none"
    ).view(B_m, T_c)


@torch.no_grad()
def evaluate_model(model, tokenizer, val_set, max_new_tokens, batch_size=16):
    model.eval()
    _set_grad_ckpt(model, False)
    solved, tot_len = 0, 0
    for i in range(0, len(val_set), batch_size):
        batch = val_set[i : i + batch_size]
        prompts = [format_prompt(tokenizer, p) for p in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        pl = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        cids = gen[:, pl:]
        txts = tokenizer.batch_decode(cids, skip_special_tokens=True)
        for t, p, ids in zip(txts, batch, cids):
            _, ok = score_countdown(t, p["nums"], p["target"])
            solved += int(ok)
            tot_len += int((ids != tokenizer.pad_token_id).sum().item())
    return {"val_acc": solved / len(val_set), "val_len": tot_len / len(val_set)}


def run_arm(model, init_state, tokenizer, val_set, train_stream, arm, cfg, seed):
    print("\n" + "=" * 84, flush=True)
    print(f"ARM: {arm['name']} | seed={seed} | lr_muon={arm['lr_muon']} lr_adam={arm['lr_adam']} cap={arm['cap']} p={arm['cap_p']} k={arm.get('cap_rank', 4)}", flush=True)
    print("=" * 84, flush=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in init_state:
                p.copy_(init_state[n].to(device=p.device, dtype=p.dtype))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    muon_params, adam_params = split_muon_and_adam_params(model)
    opts = []
    if adam_params:
        opts.append(torch.optim.AdamW(adam_params, lr=arm["lr_adam"], betas=(0.9, 0.95)))
    use_cap = bool(arm["cap"] and arm["cap_p"] > 0.0)
    cap_opt = CapMuon(
        muon_params, lr=arm["lr_muon"], momentum=cfg["MUON_MOMENTUM"], nesterov=True, ns_steps=5,
        cap=use_cap, cap_p=arm["cap_p"], cap_c=1.0, cap_ema=0.9, cap_rank=arm.get("cap_rank", 4),
    )
    attached = cap_opt.attach_hooks(model) if use_cap else 0
    opts.append(cap_opt)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in opts]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizers ready | 2D mats={len(muon_params)} | hooked={attached}", flush=True)

    history = {
        "arm": arm, "seed": seed, "steps": [], "train_acc": [], "approx_kl": [], "ref_kl": [],
        "clipfrac": [], "resp_len": [], "top_ratio": [], "min_s": [], "val_steps": [], "val_acc": [],
        "wall_sec": 0.0, "peak_vram_gb": 0.0,
    }
    t0_all = time.time()
    v0 = evaluate_model(model, tokenizer, val_set, cfg["MAX_NEW_TOKENS"], cfg["EVAL_BATCH"])
    history["val_steps"].append(0); history["val_acc"].append(v0["val_acc"])
    print(f"[Step 000] INIT VAL | val_acc={v0['val_acc']*100:5.1f}% | len={v0['val_len']:.1f}", flush=True)

    P, G, MB = cfg["PROMPTS_PER_STEP"], cfg["GROUP_SIZE"], cfg["MICRO_BATCH"]
    B = P * G
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    kl_beta = cfg["KL_BETA"]

    for step in range(1, cfg["NUM_TRAIN_STEPS"] + 1):
        step_t0 = time.time()
        batch_puzzles = train_stream[(step - 1) * P : step * P]
        repeated = [puz for puz in batch_puzzles for _ in range(G)]
        prompts = [format_prompt(tokenizer, puz) for puz in repeated]

        model.eval(); _set_grad_ckpt(model, False)
        cap_opt.set_capture(False)
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        pl = enc["input_ids"].shape[1]
        with torch.no_grad():
            seq_ids = model.generate(
                **enc, max_new_tokens=cfg["MAX_NEW_TOKENS"], do_sample=True,
                temperature=cfg["TEMPERATURE"], top_p=cfg["TOP_P"],
                pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True,
            )
        cids = seq_ids[:, pl:]
        if cids.shape[1] == 0: continue
        is_eos = cids == eos_id
        eos_cs = is_eos.cumsum(dim=1)
        comp_mask = ((eos_cs == 0) | (is_eos & (eos_cs == 1))).float()
        if pad_id != eos_id:
            comp_mask = comp_mask * (cids != pad_id).float()
        full_attn = torch.cat([enc["attention_mask"], comp_mask.long()], dim=1)

        txts = tokenizer.batch_decode(cids, skip_special_tokens=True)
        r_list = [score_countdown(t, p["nums"], p["target"])[0] for t, p in zip(txts, repeated)]
        rewards = torch.tensor(r_list, device=DEVICE, dtype=torch.float32).view(P, G)
        r_mean, r_std = rewards.mean(dim=1, keepdim=True), rewards.std(dim=1, keepdim=True)
        adv = torch.where(r_std > 1e-6, (rewards - r_mean) / (r_std + 1e-6), torch.zeros_like(rewards)).reshape(B)
        train_acc = float(np.mean(r_list))
        mean_len = float(comp_mask.sum(dim=1).mean().item())

        if adv.abs().max().item() < 1e-6:
            kl_val = ref_kl_val = clip_val = 0.0
        else:
            prog = (step - 1) / max(1, cfg["NUM_TRAIN_STEPS"] - 1)
            lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
            for opt, blrs in zip(opts, base_lrs):
                for g, blr in zip(opt.param_groups, blrs):
                    g["lr"] = blr * lr_mult

            old_chunks = []
            with torch.no_grad():
                for s in range(0, B, MB):
                    e = min(s + MB, B)
                    old_chunks.append(compute_completion_logprobs(model, seq_ids[s:e], full_attn[s:e], pl).detach())
            old_lp = torch.cat(old_chunks, dim=0)

            ref_lp = None
            if kl_beta > 0.0:
                with torch.no_grad():
                    cur_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in init_state}
                    for n, p in model.named_parameters():
                        if n in init_state: p.copy_(init_state[n].to(device=p.device, dtype=p.dtype))
                    ref_chunks = []
                    for s in range(0, B, MB):
                        e = min(s + MB, B)
                        ref_chunks.append(compute_completion_logprobs(model, seq_ids[s:e], full_attn[s:e], pl).detach())
                    for n, p in model.named_parameters():
                        if n in cur_state: p.copy_(cur_state[n].to(device=p.device, dtype=p.dtype))
                    del cur_state
                ref_lp = torch.cat(ref_chunks, dim=0)

            model.train(); _set_grad_ckpt(model, True)
            kl_acc = rkl_acc = clip_acc = tok_acc = 0.0
            tot_tok = comp_mask.sum().clamp_min(1.0)
            for ep in range(cfg["PPO_EPOCHS"]):
                for opt in opts: opt.zero_grad(set_to_none=True)
                perm = torch.randperm(B, device=DEVICE)
                for mbi, s in enumerate(range(0, B, MB)):
                    idx = perm[s : s + MB]
                    mb_mask = comp_mask[idx]
                    mb_toks = mb_mask.sum()
                    if mb_toks.item() < 1.0: continue
                    cap_opt.set_capture(use_cap and ep == 0 and mbi == 0)
                    new_lp = compute_completion_logprobs(model, seq_ids[idx], full_attn[idx], pl)
                    cap_opt.set_capture(False)
                    log_r = (new_lp - old_lp[idx]).clamp(-10.0, 10.0)
                    ratio = torch.exp(log_r)
                    surr1 = ratio * adv[idx].unsqueeze(1)
                    surr2 = ratio.clamp(1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"]) * adv[idx].unsqueeze(1)
                    loss = -(torch.minimum(surr1, surr2) * mb_mask).sum() / tot_tok
                    rkl_mb = 0.0
                    if ref_lp is not None and kl_beta > 0.0:
                        rd = (ref_lp[idx] - new_lp).clamp(-10.0, 10.0)
                        rkl_tok = torch.exp(rd) - rd - 1.0
                        loss = loss + kl_beta * (rkl_tok * mb_mask).sum() / tot_tok
                        rkl_mb = float((rkl_tok.detach() * mb_mask).sum().item())
                    loss.backward()
                    with torch.no_grad():
                        kl_acc += (((ratio - 1.0) - log_r) * mb_mask).sum().item()
                        rkl_acc += rkl_mb
                        clip_acc += (((ratio - 1.0).abs() > cfg["CLIP_EPS"]).float() * mb_mask).sum().item()
                        tok_acc += mb_toks.item()
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["GRAD_CLIP"])
                for opt in opts: opt.step()
            kl_val = kl_acc / max(tok_acc, 1.0)
            ref_kl_val = rkl_acc / max(tok_acc, 1.0)
            clip_val = clip_acc / max(tok_acc, 1.0)

        cs = cap_opt.get_cap_summary() if use_cap else {"max_ratio": 1.0, "min_s": 1.0}
        history["steps"].append(step); history["train_acc"].append(train_acc)
        history["approx_kl"].append(kl_val); history["ref_kl"].append(ref_kl_val)
        history["clipfrac"].append(clip_val); history["resp_len"].append(mean_len)
        history["top_ratio"].append(cs["max_ratio"]); history["min_s"].append(cs["min_s"])
        dt = time.time() - step_t0
        if step == 1 or step % 5 == 0 or step == cfg["NUM_TRAIN_STEPS"]:
            cstr = f" | max_lam={cs['max_ratio']:6.1f}x min_s={cs['min_s']:.2f}" if use_cap else ""
            print(f"[Step {step:03d}/{cfg['NUM_TRAIN_STEPS']}] acc={train_acc*100:5.1f}% | kl={kl_val:.4f} rkl={ref_kl_val:.4f} clip={clip_val*100:4.1f}% | len={mean_len:5.1f} | {dt:.1f}s{cstr}", flush=True)
        if step % cfg["EVAL_EVERY"] == 0 or step == cfg["NUM_TRAIN_STEPS"]:
            vr = evaluate_model(model, tokenizer, val_set, cfg["MAX_NEW_TOKENS"], cfg["EVAL_BATCH"])
            history["val_steps"].append(step); history["val_acc"].append(vr["val_acc"])
            print(f"  >>> VAL @ step {step:03d}: val_acc={vr['val_acc']*100:5.1f}% | len={vr['val_len']:.1f}", flush=True)

    history["wall_sec"] = time.time() - t0_all
    history["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    cap_opt.remove_hooks()
    del opts, cap_opt
    gc.collect(); torch.cuda.empty_cache()
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--val-size", type=int, default=80)
    ap.add_argument("--lr-muon", type=float, default=5e-5)
    ap.add_argument("--out", default="/content/stage2_gemma4_results.json")
    args = ap.parse_args()

    cfg = {
        "MODEL_ID": args.model, "NUM_TRAIN_STEPS": args.steps, "PROMPTS_PER_STEP": 4, "GROUP_SIZE": 8,
        "PPO_EPOCHS": 4, "MICRO_BATCH": 2, "MAX_NEW_TOKENS": 224,
        "TEMPERATURE": 0.6, "TOP_P": 0.9, "CLIP_EPS": 0.2, "KL_BETA": 0.02, "GRAD_CLIP": 1.0,
        "MUON_MOMENTUM": 0.85, "VAL_SIZE": args.val_size, "EVAL_EVERY": 5, "EVAL_BATCH": 16,
    }
    model, tok = load_gemma4(args.model)
    init_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    arms = [
        {"name": "muon_p000",   "lr_adam": 5e-6, "lr_muon": args.lr_muon, "cap": False, "cap_p": 0.00, "cap_rank": 4},
        {"name": "cap_p025_k4", "lr_adam": 5e-6, "lr_muon": args.lr_muon, "cap": True,  "cap_p": 0.25, "cap_rank": 4},
        {"name": "cap_p035_k4", "lr_adam": 5e-6, "lr_muon": args.lr_muon, "cap": True,  "cap_p": 0.35, "cap_rank": 4},
        {"name": "cap_p050_k4", "lr_adam": 5e-6, "lr_muon": args.lr_muon, "cap": True,  "cap_p": 0.50, "cap_rank": 4},
    ]

    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out)).get("results", {})
        except Exception:
            pass

    val_set = make_dataset(cfg["VAL_SIZE"], seed=30042, n_nums=4)
    train_stream = make_dataset(cfg["NUM_TRAIN_STEPS"] * cfg["PROMPTS_PER_STEP"], seed=42, n_nums=4)

    for arm in arms:
        if arm["name"] in results:
            print(f"Skipping completed {arm['name']}", flush=True)
            continue
        results[arm["name"]] = run_arm(model, init_state, tok, val_set, train_stream, arm, cfg, seed=42)
        with open(args.out, "w") as f:
            json.dump({"config": cfg, "results": results}, f, indent=2)

    print("\n" + "=" * 100, flush=True)
    print(f"STAGE 2 GEMMA 4 SUMMARY — {args.model} (4-num hard Countdown, GRPO mu=4, {args.steps} steps)", flush=True)
    for k, r in results.items():
        print(f"  {k:<14} init={r['val_acc'][0]*100:5.1f}% meanVal={np.mean(r['val_acc'][1:])*100:5.1f}% finVal={r['val_acc'][-1]*100:5.1f}% peakVal={max(r['val_acc'])*100:5.1f}% meanTr={np.mean(r['train_acc'])*100:5.1f}% rkl={r['ref_kl'][-1]:.4f} vram={r['peak_vram_gb']:.1f}GB t={r['wall_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
