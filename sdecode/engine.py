"""Real-model speculative-decoding harness (HF GPT-2 family).

Provides batched verification (one target forward per round), measured drafter
cost (c), and naive / fixed-K / adaptive lanes that share a generation loop so
their decoded token sequences can be compared token-for-token.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdecode.core import AdaptiveSpeculator, greedy_verify

PROMPTS = [
    "The quick brown fox jumps over the lazy dog and then",
    "In a quiet village nestled between the mountains, the old clockmaker",
    "Machine learning systems that decode text one token at a time are",
]


@dataclass
class LaneResult:
    model_pair: str
    target: str
    drafter: str
    prompt: str
    lane: str
    fixed_k: int
    max_new_tokens: int
    drafter_cost_c: float
    generated_tokens: int
    target_calls: int
    drafter_calls: int
    wall_time_s: float
    tokens_per_target_call: float
    tokens_per_second: float
    generated_text: str
    token_ids: List[int]
    matches_reference: bool


def load_pair(target_id: str, drafter_id: str):
    tok = AutoTokenizer.from_pretrained(target_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    target = AutoModelForCausalLM.from_pretrained(target_id)
    drafter = AutoModelForCausalLM.from_pretrained(drafter_id)
    target.eval()
    drafter.eval()
    return target, drafter, tok


def encode(tok, text: str) -> torch.Tensor:
    return tok.encode(text, return_tensors="pt")[0]


def decode(tok, ids: List[int]) -> str:
    return tok.decode(ids, skip_special_tokens=True)


@torch.inference_mode()
def measure_drafter_cost(target, drafter, input_ids: torch.Tensor, n: int = 8) -> float:
    """Measured c = mean(drafter forward time) / mean(target forward time)."""
    for _ in range(2):
        target(input_ids.unsqueeze(0))
        drafter(input_ids.unsqueeze(0))
    t_times = []
    d_times = []
    for _ in range(n):
        s = time.perf_counter()
        target(input_ids.unsqueeze(0))
        t_times.append(time.perf_counter() - s)
        s = time.perf_counter()
        drafter(input_ids.unsqueeze(0))
        d_times.append(time.perf_counter() - s)
    t_mean = sum(t_times) / len(t_times)
    d_mean = sum(d_times) / len(d_times)
    return float(d_mean / max(t_mean, 1e-9))


@torch.inference_mode()
def _verify_round(ctx_ids: torch.Tensor, target, drafter, K: int):
    """Draft K tokens, verify in a single target forward. Returns (emitted ids, accepted drafts, drafter_calls)."""
    K = max(1, int(K))
    drafts = []
    cur = ctx_ids.clone()
    drafter_calls = 0
    for _ in range(K):
        logits = drafter(cur.unsqueeze(0)).logits[0, -1]
        nxt = int(logits.argmax().item())
        drafts.append(nxt)
        cur = torch.cat([cur, torch.tensor([nxt])])
        drafter_calls += 1
    full = torch.cat([ctx_ids, torch.tensor(drafts)])
    t_out = target(full.unsqueeze(0))
    t_logits = t_out.logits[0]
    base = ctx_ids.shape[0] - 1
    block = t_logits[base : base + K + 1]
    draft_tensor = torch.tensor(drafts)
    tgt = block.argmax(dim=-1)
    accepted = 0
    for i in range(K):
        if drafts[i] == tgt[i].item():
            accepted += 1
        else:
            break
    if accepted == K:
        emitted = list(drafts)
    else:
        emitted = list(drafts[:accepted]) + [int(tgt[accepted].item())]
    emitted_tensor = torch.tensor(emitted)
    _ = greedy_verify(block.clone(), draft_tensor.clone())
    return emitted_tensor, accepted, drafter_calls


@torch.inference_mode()
def run_lane(
    target,
    drafter,
    tok,
    prompt: str,
    max_new_tokens: int,
    mode: str,
    drafter_cost: float,
    fixed_k: int = 4,
    model_pair: str = "",
    target_id: str = "",
    drafter_id: str = "",
) -> LaneResult:
    seq = encode(tok, prompt).clone()
    generated = 0
    target_calls = 0
    drafter_calls = 0
    t0 = time.perf_counter()
    controller = AdaptiveSpeculator(drafter_cost=drafter_cost, k_max=8, window=8)
    out_ids = []
    while generated < max_new_tokens:
        if mode == "naive":
            logits = target(seq.unsqueeze(0)).logits[0, -1]
            nxt = int(logits.argmax().item())
            seq = torch.cat([seq, torch.tensor([nxt])])
            out_ids.append(nxt)
            target_calls += 1
            generated += 1
            continue
        if mode.startswith("fixed"):
            K = fixed_k
        elif mode == "adaptive":
            K = controller.next_k()
        else:
            K = fixed_k
        K = min(K, max_new_tokens - generated)
        if K < 1:
            K = 1
        emitted, accepted, dcalls = _verify_round(seq, target, drafter, K)
        target_calls += 1
        drafter_calls += dcalls
        take = min(int(emitted.numel()), max_new_tokens - generated)
        chunk = emitted[:take].tolist()
        seq = torch.cat([seq, torch.tensor(chunk)])
        out_ids.extend(chunk)
        generated += take
        if mode == "adaptive":
            controller.observe(accepted if accepted > 0 else 0, K)
        if chunk and chunk[-1] == tok.eos_token_id:
            break
    wall = time.perf_counter() - t0
    text = decode(tok, out_ids)
    lane_label = mode if mode == "naive" else ("adaptive" if mode == "adaptive" else f"fixed-k{fixed_k}")
    return LaneResult(
        model_pair=model_pair,
        target=target_id,
        drafter=drafter_id,
        prompt=prompt,
        lane=lane_label,
        fixed_k=int(fixed_k if mode.startswith("fixed") else 0),
        max_new_tokens=max_new_tokens,
        drafter_cost_c=float(drafter_cost),
        generated_tokens=len(out_ids),
        target_calls=int(target_calls),
        drafter_calls=int(drafter_calls),
        wall_time_s=round(wall, 4),
        tokens_per_target_call=round(len(out_ids) / max(1, target_calls), 3),
        tokens_per_second=round(len(out_ids) / max(wall, 1e-6), 3),
        generated_text=text,
        token_ids=[int(x) for x in out_ids],
        matches_reference=False,
    )


def run_pair(
    target_id: str,
    drafter_id: str,
    prompts: List[str],
    max_new_tokens: int = 16,
    fixed_ks: tuple = (4, 8),
) -> dict:
    target, drafter, tok = load_pair(target_id, drafter_id)
    ctx_ids = encode(tok, prompts[0] if prompts else "Hello world")
    c = measure_drafter_cost(target, drafter, ctx_ids)
    pair_label = f"{target_id}<-{drafter_id}"
    # naive first
    naive_by_prompt = {}
    results = []
    for prompt in prompts:
        nres = run_lane(
            target, drafter, tok, prompt, max_new_tokens, "naive", c,
            model_pair=pair_label, target_id=target_id, drafter_id=drafter_id,
        )
        naive_by_prompt[prompt] = nres.token_ids
        nres.matches_reference = True
        results.append(nres)
    for prompt in prompts:
        # fixed lanes
        for fk in fixed_ks:
            res = run_lane(
                target, drafter, tok, prompt, max_new_tokens, "fixed", c,
                fixed_k=int(fk), model_pair=pair_label, target_id=target_id, drafter_id=drafter_id,
            )
            ref = naive_by_prompt[prompt]
            n = min(len(ref), len(res.token_ids))
            res.matches_reference = bool(ref[:n] == res.token_ids[:n]) and (len(ref) == len(res.token_ids))
            results.append(res)
        # adaptive lane
        res = run_lane(
            target, drafter, tok, prompt, max_new_tokens, "adaptive", c,
            fixed_k=int(fixed_ks[0]), model_pair=pair_label, target_id=target_id, drafter_id=drafter_id,
        )
        ref = naive_by_prompt[prompt]
        n = min(len(ref), len(res.token_ids))
        res.matches_reference = bool(ref[:n] == res.token_ids[:n]) and (len(ref) == len(res.token_ids))
        results.append(res)
    return {
        "pair": pair_label,
        "target": target_id,
        "drafter": drafter_id,
        "drafter_cost_c": round(c, 4),
        "fixed_ks": list(fixed_ks),
        "max_new_tokens": max_new_tokens,
        "prompts": prompts,
        "runs": [asdict(r) if False else {
            "model_pair": r.model_pair, "target": r.target, "drafter": r.drafter,
            "prompt": r.prompt, "lane": r.lane, "fixed_k": r.fixed_k,
            "max_new_tokens": r.max_new_tokens, "drafter_cost_c": r.drafter_cost_c,
            "generated_tokens": r.generated_tokens, "target_calls": r.target_calls,
            "drafter_calls": r.drafter_calls, "wall_time_s": r.wall_time_s,
            "tokens_per_target_call": r.tokens_per_target_call,
            "tokens_per_second": r.tokens_per_second, "generated_text": r.generated_text,
            "token_ids": r.token_ids, "matches_reference": r.matches_reference,
        } for r in results],
    }