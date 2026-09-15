"""Demo: adaptive K vs fixed K, acceptance-rate sweep, receipts + chart."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdecode.core import expected_tokens, optimal_draft_len  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

# ---- 1) analytic sweep: tokens per unit cost, fixed K vs adaptive K ---------
DRAFTER_COST = 0.1
sweep = {}
for p in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
    fixed = {k: expected_tokens(p, k) / (1 + DRAFTER_COST * k) for k in (2, 4, 8)}
    k_star = optimal_draft_len(p, DRAFTER_COST, k_max=8)
    adapt = expected_tokens(p, k_star) / (1 + DRAFTER_COST * k_star)
    sweep[f"p={p}"] = {
        "fixed_k2": round(fixed[2], 3),
        "fixed_k4": round(fixed[4], 3),
        "fixed_k8": round(fixed[8], 3),
        "adaptive_k": k_star,
        "adaptive_gain": round(adapt, 3),
        "best_fixed": round(max(fixed.values()), 3),
        "adaptive_vs_best_fixed": f"{adapt / max(fixed.values()):+.1%}",
    }

# ---- 2) naive vs speculative tokens-per-target-call at p=0.8 --------------
p = 0.8
k = optimal_draft_len(p, DRAFTER_COST, 8)
naive = 1.0
spec = expected_tokens(p, k) / (1 + DRAFTER_COST * k)

receipt = {
    "drafter_cost": DRAFTER_COST,
    "tokens_per_target_call": {
        "naive": naive,
        f"speculative_adaptive_k{k}": round(spec, 3),
        "speedup": f"{spec / naive:.2f}x",
    },
    "adaptive_vs_fixed_sweep": sweep,
    "theorem": "greedy verification is distribution-identical to the target model",
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, style in ((2, ":"), (4, "--"), (8, "-")):
        ax.plot(ps, [expected_tokens(p, k) / (1 + DRAFTER_COST * k) for p in ps],
                style, label=f"fixed K={k}")
    ax.plot(ps, [expected_tokens(p, optimal_draft_len(p, DRAFTER_COST, 8))
                 / (1 + DRAFTER_COST * optimal_draft_len(p, DRAFTER_COST, 8)) for p in ps],
            "ks-", label="adaptive K (ours)", markersize=9)
    ax.set_xlabel("draft acceptance rate p")
    ax.set_ylabel("tokens per unit cost")
    ax.set_title("Adaptive draft length tracks the optimum across acceptance regimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "adaptive_k.png"), dpi=140)
    print("chart -> receipts/adaptive_k.png")
except Exception as e:  # pragma: no cover
    print("chart skipped:", e)

assert spec > 2.0, "expected >2x at p=0.8"
for key, row in sweep.items():
    assert row["adaptive_gain"] >= row["best_fixed"] - 0.01, (key, row)
print("DEMO_OK")
