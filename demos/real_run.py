"""Real-model receipt run.

Runs the shipped sdecode code (AdaptiveSpeculator + the batched engine path)
against real Hugging Face model pairs on this machine, captures a full trace
per round, verifies token-for-token identity against naive greedy decoding,
and writes receipts/real_run.json.

    python demos/real_run.py --target gpt2 --drafter distilgpt2
    python demos/real_run.py --target gpt2-medium --drafter distilgpt2

Everything in the receipt is measured on this box. There are no GPU claims
here and no projected numbers — a reference harness (no KV-cache reuse across
rounds; the whole prefix is re-forwarded, same as the naive baseline) timed on
CPU with exactly the models named in the receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from sdecode.engine import (  # noqa: E402
    HFEngine,
    adaptive_generate,
    fixed_k_generate,
    measure_drafter_cost,
    naive_generate,
)

PROMPTS = [
    {
        "name": "prose",
        "text": (
            "The expedition left the harbour at dawn, sails heavy with the cold "
            "Atlantic wind. By the third week the crew had settled into a rhythm "
            "of watches and repairs, and the navigator's log grew thick with "
            "careful observations of the coast."
        ),
    },
    {
        "name": "code",
        "text": (
            "def merge_intervals(intervals):\n"
            "    intervals = sorted(intervals, key=lambda x: x[0])\n"
            "    merged = []\n"
            "    for start, end in intervals:\n"
            "        if merged and start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        ),
    },
    {
        "name": "repeat",
        "text": (
            "one small step, two small steps, three small steps, four small "
            "steps, five small steps, six small steps,"
        ),
    },
]


def clean_piece(tok, token_id: int) -> str:
    piece = tok.decode([token_id])
    return piece.replace("\u0120", " ")  # GPT-2 BPE space marker


def lane_record(res: dict, tok) -> dict:
    """Compact + film-ready record for one lane."""
    return {
        "n_tokens": res["n_tokens"],
        "target_forwards": res["target_forwards"],
        "drafter_forwards": res["drafter_forwards"],
        "wall_s": round(res["wall_s"], 3),
        "ms_per_token": round(res["ms_per_token"], 2),
        "tokens_per_target_forward": round(res["tokens_per_target_forward"], 3),
        "text": tok.decode(res["tokens"]),
        "rounds": [
            {
                "round": r["round"],
                "k": r["k"],
                "accepted": r["accepted"],
                "wall_ms": round(r["wall_ms"], 1),
                "draft_pieces": [clean_piece(tok, t) for t in r["draft"]],
                "emitted_pieces": [clean_piece(tok, t) for t in r["emitted"]],
            }
            for r in res["rounds"]
        ],
    }


def run_pair(target_name: str, drafter_name: str, max_tokens: int, out: list) -> None:
    print(f"\n=== pair: {target_name} <- {drafter_name} ===", flush=True)
    t_load = time.perf_counter()
    target = HFEngine(target_name)
    drafter = HFEngine(drafter_name)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f}s", flush=True)

    pair = {
        "target": target_name,
        "drafter": drafter_name,
        "load_s": round(load_s, 1),
        "prompts": [],
    }
    c_measured = measure_drafter_cost(target, drafter)
    pair["drafter_cost_measured"] = round(c_measured, 4)
    print(f"  measured drafter/target cost ratio c = {c_measured:.3f} "
          f"(shipped default assumes 0.1)", flush=True)
    for p in PROMPTS:
        print(f"  prompt: {p['name']}", flush=True)
        x = target.encode(p["text"])
        base = naive_generate(x, target, max_tokens)
        adapt = adaptive_generate(x, target, drafter, max_tokens, drafter_cost=0.1)
        adapt_cal = adaptive_generate(x, target, drafter, max_tokens,
                                      drafter_cost=c_measured)
        k8 = fixed_k_generate(x, target, drafter, max_tokens, k=8)
        n_check = min(adapt["n_tokens"], adapt_cal["n_tokens"], k8["n_tokens"],
                      base["n_tokens"])
        ident = {
            "checked_tokens": n_check,
            "adaptive_matches_naive": adapt["tokens"][:n_check] == base["tokens"][:n_check],
            "adaptive_calibrated_matches_naive": adapt_cal["tokens"][:n_check] == base["tokens"][:n_check],
            "fixed_k8_matches_naive": k8["tokens"][:n_check] == base["tokens"][:n_check],
        }
        assert ident["adaptive_matches_naive"], f"identity broken on {p['name']}"
        assert ident["adaptive_calibrated_matches_naive"], f"identity broken on {p['name']}"
        assert ident["fixed_k8_matches_naive"], f"identity broken on {p['name']}"

        rec = {
            "name": p["name"],
            "prompt": p["text"],
            "prompt_tokens": len(x),
            "naive": lane_record(base, target.tok),
            "adaptive": lane_record(adapt, target.tok),
            "adaptive_calibrated": lane_record(adapt_cal, target.tok),
            "fixed_k8": lane_record(k8, target.tok),
            "identity": ident,
            "p_estimate": adapt["p_estimate"],
            "k_choices": adapt["k_choices"],
            "k_choices_calibrated": adapt_cal["k_choices"],
        }
        pair["prompts"].append(rec)
        print(
            f"    naive {base['ms_per_token']:.1f} ms/tok | "
            f"adaptive(c=0.1) {adapt['ms_per_token']:.1f} "
            f"({adapt['tokens_per_target_forward']:.2f} tok/fwd, K={adapt['k_choices'][-6:]}) | "
            f"adaptive(c={c_measured:.2f}) {adapt_cal['ms_per_token']:.1f} "
            f"(K={adapt_cal['k_choices'][-6:]}) | "
            f"k8 {k8['ms_per_token']:.1f}",
            flush=True,
        )
    out.append(pair)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["gpt2:distilgpt2"],
                    help="target:drafter model pairs")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--out", default="receipts/real_run.json")
    args = ap.parse_args()

    experiments = []
    for spec in args.pairs:
        target_name, drafter_name = spec.split(":")
        run_pair(target_name, drafter_name, args.tokens, experiments)

    receipt = {
        "harness": {
            "generated_by": "demos/real_run.py",
            "note": (
                "Reference harness on CPU: no KV-cache reuse; every step "
                "re-forwards the prefix, identical to the naive baseline's "
                "cost structure. Wall-clock is a single run per lane on a "
                "shared box — treat as indicative, the call-counts are exact."
            ),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "max_tokens": args.tokens,
        },
        "experiments": experiments,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(f"\nwrote {args.out}")

    # compact aggregate table across all prompts of the last experiment
    print("\naggregate (all prompts, mean ms/token):")
    for exp in experiments:
        for lane in ("naive", "adaptive", "adaptive_calibrated", "fixed_k8"):
            vals = [p[lane]["ms_per_token"] for p in exp["prompts"]]
            tpfs = [p[lane]["tokens_per_target_forward"] for p in exp["prompts"]]
            print(
                f"  {exp['target']}<-{exp['drafter']} {lane:20s} "
                f"{sum(vals)/len(vals):7.1f} ms/tok  "
                f"{sum(tpfs)/len(tpfs):.2f} tok/fwd"
            )


if __name__ == "__main__":
    main()
