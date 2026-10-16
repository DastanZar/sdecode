"""Generate assets/real_film.gif, assets/real_ledger.png and app/real_run_data.js from receipts/real_run.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sdecode.engine import (  # noqa: E402
    PROMPTS,
    _verify_round,
    encode,
    load_pair,
    measure_drafter_cost,
)

ASSETS = ROOT / "assets"
APP = ROOT / "app"
ASSETS.mkdir(exist_ok=True)
APP.mkdir(exist_ok=True)

receipt_path = ROOT / "receipts" / "real_run.json"
if not receipt_path.exists():
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "demos" / "real_run.py")], check=True)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))


# --- 2x2 receipt-derived ledger ---
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.patch.set_facecolor("#0b0f14")
for ax in axes.ravel():
    ax.set_facecolor("#11161d")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#222c36")

pair_colors = {"gpt2": "#C4F03C", "gpt2-medium": "#7dd3fc"}

# tokens per target call by lane, averaged over prompts, first pair
first = receipt["pairs"][0]
lanes = ["fixed-k4", "fixed-k8", "adaptive", "naive"]
by_lane = {}
for run in first["runs"]:
    by_lane.setdefault(run["lane"], []).append(run["tokens_per_target_call"])
x = np.arange(len(lanes))
vals = [np.mean(by_lane.get(l, [0.0])) for l in lanes]
axes[0, 0].bar(x, vals, color=[pair_colors["gpt2"]] * len(lanes))
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(lanes, rotation=20, color="white")
axes[0, 0].set_ylabel("tokens / target call", color="white")
axes[0, 0].set_title(first["pair"] + " speedup", color="white")

# target calls per 16 tokens
by_lane_calls = {}
for run in first["runs"]:
    by_lane_calls.setdefault(run["lane"], []).append(run["target_calls"])
vals2 = [np.mean(by_lane_calls.get(l, [0.0])) for l in lanes]
axes[0, 1].bar(x, vals2, color="#ff6b6b")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(lanes, rotation=20, color="white")
axes[0, 1].set_ylabel("target calls (16 tok)", color="white")
axes[0, 1].set_title("fewer calls = faster", color="white")

# drafter cost c by pair
pairs = [p["pair"] for p in receipt["pairs"]]
cs = [p["drafter_cost_c"] for p in receipt["pairs"]]
axes[1, 0].bar(np.arange(len(pairs)), cs, color=[pair_colors.get(p.split("<-")[0], "#C4F03C") for p in pairs])
axes[1, 0].set_xticks(np.arange(len(pairs)))
axes[1, 0].set_xticklabels([p.split("<-")[0] for p in pairs], color="white", rotation=15)
axes[1, 0].set_ylabel("c = T_drafter / T_target", color="white")
axes[1, 0].set_title("measured drafter cost", color="white")

# adaptive improvement over naive %
imp = []
for p in receipt["pairs"]:
    nv = np.mean([r["tokens_per_target_call"] for r in p["runs"] if r["lane"] == "naive"])
    ad = np.mean([r["tokens_per_target_call"] for r in p["runs"] if r["lane"] == "adaptive"])
    imp.append((ad - nv) * 100.0)
axes[1, 1].bar(np.arange(len(pairs)), imp, color="#a78bfa")
axes[1, 1].set_xticks(np.arange(len(pairs)))
axes[1, 1].set_xticklabels([p.split("<-")[0] for p in pairs], color="white", rotation=15)
axes[1, 1].set_ylabel("% vs naive tok/call", color="white")
axes[1, 1].set_title("adaptive gain", color="white")

fig.tight_layout()
fig.savefig(ASSETS / "real_ledger.png", dpi=140)
plt.close(fig)
print("wrote real_ledger.png")


# --- real_film.gif: 78 frames from one measured speculative run ---
target, drafter, tok = load_pair("gpt2", "distilgpt2")
c = measure_drafter_cost(target, drafter, encode(tok, PROMPTS[0]))

seq = encode(tok, PROMPTS[0]).clone()
max_new = 16
K = 4
rounds = []
generated = 0
while generated < max_new:
    kk = min(K, max_new - generated)
    emitted, accepted, _ = _verify_round(seq, target, drafter, kk)
    take = min(int(emitted.numel()), max_new - generated)
    chunk = emitted[:take].tolist()
    seq = torch.cat([seq, torch.tensor(chunk, dtype=seq.dtype)])
    generated += take
    rounds.append({"generated": generated, "seq": seq.clone().tolist(), "accepted": accepted, "K": kk, "chunk": chunk})
    if not chunk or chunk[-1] == tok.eos_token_id:
        break


def frame(i, n):
    idx = min(len(rounds) - 1, int(i / max(1, n - 1) * (len(rounds) - 1)))
    st = rounds[idx]
    gen_ids = st["seq"][len(encode(tok, PROMPTS[0])) :]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor="#0b0f14")
    ax.set_facecolor("#11161d")
    ax.axis("off")
    ax.text(0.05, 0.92, f"real decode  gpt2 (target) <- distilgpt2 (drafter)   frame {i + 1}/{n}",
            color="#C4F03C", fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.78, f"tokens generated: {st['generated']} / {max_new}   accepted draft: {st['accepted']}/{st['K']}",
            color="white", fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.62, "output so far:", color="#9ca3af", fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.10, text[-260:], color="white", fontsize=11, transform=ax.transAxes,
            wrap=True, verticalalignment="bottom")
    ax.text(0.05, 0.50, f"target calls: {idx + 1}   naive calls for same: {st['generated']}",
            color="#9ca3af", fontsize=9, transform=ax.transAxes)
    fig.tight_layout()
    return fig


frames = []
for i in range(78):
    fig = frame(i, 78)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    frames.append(Image.fromarray(buf))
    plt.close(fig)

frames[0].save(ASSETS / "real_film.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
print("wrote real_film.gif (" + str(len(frames)) + " frames)")


# --- app data js ---
js = "window.REAL_RUN_DATA = " + json.dumps(receipt) + ";\n"
(APP / "real_run_data.js").write_text(js, encoding="utf-8")
print("wrote app/real_run_data.js")