"""Figures and film from receipts/real_run.json — real model runs only.

    python demos/make_real_figures.py [--experiment 0] [--prompt code]

Produces:
  assets/real_film.gif    animated decode of the chosen prompt's adaptive lane,
                          chips = real tokens, clock = real measured ms
  assets/real_ledger.png  per-prompt ledger: algorithmic win (tok/fwd) and
                          wall-clock (ms/tok) with honest small multiples
  app/real_run_data.js    compact trace for the playback app
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK = "#16213c"
ACCENT = "#4f46e5"
AMBER = "#f59e0b"
TEAL = "#0d9488"
GRAY = "#94a3b8"
PAPER = "#ffffff"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=9)


def load_trace(experiment_ix: int):
    with open(os.path.join(ROOT, "receipts", "real_run.json")) as fh:
        data = json.load(fh)
    return data, data["experiments"][experiment_ix]


def pick_prompt(exp, name):
    for p in exp["prompts"]:
        if p["name"] == name:
            return p
    raise SystemExit(f"prompt {name!r} not in receipt")


def chip(ax, x, y, w, h, text, face, edge, text_color="white", strike=False):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.1, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(box)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.2, color=text_color, zorder=4, family="DejaVu Sans")
    if strike:
        ax.plot([x + 0.01, x + w - 0.01], [y + h / 2, y + h / 2],
                color=text_color, linewidth=1.1, zorder=5, alpha=0.85)
        t.set_alpha(0.75)
    return box


def trunc(piece: str, n: int = 7) -> str:
    s = piece.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def build_film(exp, prompt, lane_key, path):
    pr = pick_prompt(exp, prompt)
    lane = pr[lane_key]
    naive_ms = pr["naive"]["ms_per_token"]
    k8_ms = pr["fixed_k8"]["ms_per_token"]
    rounds = lane["rounds"]

    fig = plt.figure(figsize=(12.4, 7.0), dpi=100)
    fig.patch.set_facecolor(PAPER)

    # ---- geometry
    left, right = 0.03, 0.975
    stream_top, stream_bottom = 0.885, 0.315
    chip_h = 0.052
    gap = 0.014
    cols = 26
    usable = right - left - 0.10
    chip_w = usable / cols - 0.0055
    total_tokens = lane["n_tokens"]
    wall_total = lane["wall_s"] * 1e3

    # per-round chip plans: list of (round_ix, [chips]) where a chip is
    # (piece, settled_style | draft_style is same ghost)
    # settled styles: 'acc' teal fill, 'rej' amber outline + strike,
    # 'cor' indigo fill (correction), 'bon' indigo fill (bonus)
    def plan_round(r):
        acc = r["accepted"]
        chips = []
        for j in range(acc):
            chips.append((r["draft_pieces"][j], "acc"))
        if acc < r["k"]:
            chips.append((r["draft_pieces"][acc], "rej"))
            chips.append((r["emitted_pieces"][-1], "cor"))
        else:
            chips.append((r["emitted_pieces"][-1], "bon"))
        return chips

    plans = [plan_round(r) for r in rounds]
    # flat chip positions
    flat = []  # (round_ix, chip_ix, piece, style)
    for ri, chips in enumerate(plans):
        for ci, (piece, style) in enumerate(chips):
            flat.append((ri, ci, piece, style))
    rows_needed = (len(flat) + cols - 1) // cols
    row_h = chip_h + gap
    rows_capacity = int((stream_top - stream_bottom) / row_h) + 1
    assert rows_needed <= rows_capacity, (
        f"stream overflow: {rows_needed} rows > {rows_capacity}")

    cum = []
    t = 0.0
    for r in rounds:
        t += r["wall_ms"]
        cum.append(t)

    frames = [(i, ph) for i in range(len(rounds)) for ph in (0, 1, 2)]
    frames.append((len(rounds) - 1, 2))

    def chip_at(idx):
        ri, ci, piece, style = flat[idx]
        row = idx // cols
        col = idx % cols
        x = left + 0.10 + col * (chip_w + 0.0055)
        y = stream_top - (row + 1) * row_h
        return x, y, piece, style, ri

    def draw_fig(ri, phase):
        fig.clf()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(left, 0.955, "sdecode — live decode", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(left, 0.918,
                f"{exp['target']}  ←  {exp['drafter']}   ·   prompt: {pr['name']}"
                f"   ·   CPU, real run",
                fontsize=10, color=GRAY)
        ax.text(right, 0.955, f"c measured = {exp['drafter_cost_measured']:.2f}",
                fontsize=10, color=GRAY, ha="right")

        # round markers: label above first chip of each round up to ri
        for r_i in range(ri + 1):
            first_idx = next(i for i, (rj, *_ ) in enumerate(flat) if rj == r_i)
            x, y, *_ = chip_at(first_idx)
            label = f"R{rounds[r_i]['round']} K={rounds[r_i]['k']}"
            ax.text(x, y + chip_h + 0.004, label, fontsize=7.2,
                    color=ACCENT if r_i == ri else GRAY, va="bottom")

        # chips
        for idx, (r_i, ci, piece, style) in enumerate(flat):
            if r_i > ri:
                break
            x, y, _, _, _ = chip_at(idx)
            current = r_i == ri
            settled = r_i < ri or phase == 2
            if current and phase == 0:
                chip(ax, x, y, chip_w, chip_h, trunc(piece), PAPER, AMBER, text_color=INK)
            elif current and phase == 1:
                chip(ax, x, y, chip_w, chip_h, trunc(piece), "#fef3c7", AMBER, text_color=INK)
            else:
                if style == "acc":
                    chip(ax, x, y, chip_w, chip_h, trunc(piece), TEAL, TEAL)
                elif style == "rej":
                    chip(ax, x, y, chip_w, chip_h, trunc(piece), PAPER, AMBER,
                         text_color=AMBER, strike=True)
                else:  # cor / bon
                    chip(ax, x, y, chip_w, chip_h, trunc(piece), ACCENT, ACCENT)
            if settled:
                pass

        # status
        status = "drafting…" if phase == 0 else ("verifying…" if phase == 1 else "settled")
        ax.text(left, 0.245,
                f"round {rounds[ri]['round']}   K={rounds[ri]['k']}   "
                f"accepted {rounds[ri]['accepted']}/{rounds[ri]['k']}   {status}",
                fontsize=11, color=INK)

        # ledger bars
        emitted_so_far = sum(len(rounds[i]["emitted_pieces"]) for i in range(ri + 1))
        done = cum[ri]
        naive_so_far = emitted_so_far * naive_ms
        k8_so_far = emitted_so_far * k8_ms
        scale = max(wall_total, naive_so_far, k8_so_far)
        bar_h = 0.030
        bar_x = left + 0.245
        bar_w_total = right - bar_x - 0.09
        bars = [
            ("adaptive (real)", done, ACCENT),
            ("fixed K=8 would have spent", k8_so_far, AMBER),
            ("naive would have spent", naive_so_far, GRAY),
        ]
        for i, (label, val, color) in enumerate(bars):
            yy = 0.150 - i * (bar_h + 0.014)
            ax.text(bar_x - 0.008, yy + bar_h / 2, label, fontsize=8.6,
                    color=INK, ha="right", va="center")
            frac = max(0.02, val / scale)
            ax.add_patch(FancyBboxPatch((bar_x, yy), frac * bar_w_total, bar_h,
                                        boxstyle="round,pad=0.002,rounding_size=0.008",
                                        facecolor=color, edgecolor="none"))
            ax.text(bar_x + frac * bar_w_total + 0.008, yy + bar_h / 2,
                    f"{val/1000:.1f}s · {emitted_so_far} tok", fontsize=8.6,
                    color=INK, va="center")
        ax.text(left, 0.045,
                f"{total_tokens} tokens · adaptive {wall_total/1000:.1f}s"
                f" ({lane['ms_per_token']:.0f} ms/tok) · naive {pr['naive']['wall_s']:.1f}s"
                f" · fixed K=8 {pr['fixed_k8']['wall_s']:.1f}s"
                f"   · token-identical output (checked {pr['identity']['checked_tokens']} tokens)",
                fontsize=8.8, color=GRAY)

    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=10)
    anim = matplotlib.animation.FuncAnimation(fig, lambda fr: draw_fig(*fr), frames, repeat=False)
    anim.save(path, writer=writer)
    plt.close(fig)
    return len(frames)


def build_ledger(data, path):
    exps = data["experiments"]
    fig, axes = plt.subplots(len(exps), 2, figsize=(12.4, 4.4 * len(exps)), dpi=110)
    if len(exps) == 1:
        axes = np.array([axes])
    prompts = [p["name"] for p in exps[0]["prompts"]]
    x = np.arange(len(prompts))
    lanes = [("naive", GRAY), ("adaptive", ACCENT), ("adaptive_calibrated", "#7c3aed"),
             ("fixed_k8", AMBER)]
    for row, exp in enumerate(exps):
        axl, axr = axes[row]
        w = 0.2
        for j, (lane, color) in enumerate(lanes):
            vals = [p[lane]["tokens_per_target_forward"] for p in exp["prompts"]]
            axl.bar(x + (j - 1.5) * w, vals, w, color=color,
                    label=lane.replace("adaptive_calibrated", "adaptive (c measured)"))
        axl.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")
        axl.set_title(f"{exp['target']} ← {exp['drafter']} — tokens per target forward",
                      fontsize=10.5, color=INK)
        axl.set_xticks(x, prompts)
        axl.set_ylim(0, max(6.5, 1.15 * max(
            p[lane]["tokens_per_target_forward"] for p in exp["prompts"] for lane, _ in lanes)))
        clean(axl)
        if row == 0:
            axl.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
        for j, (lane, color) in enumerate(lanes):
            vals = [p[lane]["ms_per_token"] for p in exp["prompts"]]
            axr.bar(x + (j - 1.5) * w, vals, w, color=color)
        axr.set_title(f"{exp['target']} ← {exp['drafter']} — real ms per token (CPU)",
                      fontsize=10.5, color=INK)
        axr.set_xticks(x, prompts)
        clean(axr)
        axr.text(0.99, 0.97, f"c measured = {exp['drafter_cost_measured']:.2f}",
                 transform=axr.transAxes, ha="right", va="top", fontsize=9, color=GRAY)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def build_app_data(data, path):
    """Compact trace for the app playback (pieces + clocks + summary)."""
    compact = {
        "harness": data["harness"],
        "experiments": [],
    }
    for exp in data["experiments"]:
        e = {
            "target": exp["target"],
            "drafter": exp["drafter"],
            "drafter_cost_measured": exp["drafter_cost_measured"],
            "prompts": [],
        }
        for p in exp["prompts"]:
            e["prompts"].append({
                "name": p["name"],
                "text": p["prompt"],
                "p_estimate": p["p_estimate"],
                "identity": p["identity"],
                "naive": {"ms_per_token": p["naive"]["ms_per_token"],
                          "wall_s": p["naive"]["wall_s"],
                          "tokens_per_target_forward": p["naive"]["tokens_per_target_forward"]},
                "fixed_k8": {"ms_per_token": p["fixed_k8"]["ms_per_token"],
                             "wall_s": p["fixed_k8"]["wall_s"],
                             "tokens_per_target_forward": p["fixed_k8"]["tokens_per_target_forward"]},
                "adaptive": {"ms_per_token": p["adaptive"]["ms_per_token"],
                             "wall_s": p["adaptive"]["wall_s"],
                             "tokens_per_target_forward": p["adaptive"]["tokens_per_target_forward"],
                             "p_estimate": p["p_estimate"],
                             "k_choices": p["k_choices"],
                             "rounds": p["adaptive"]["rounds"]},
                "adaptive_calibrated": {"ms_per_token": p["adaptive_calibrated"]["ms_per_token"],
                                        "wall_s": p["adaptive_calibrated"]["wall_s"],
                                        "tokens_per_target_forward": p["adaptive_calibrated"]["tokens_per_target_forward"],
                                        "k_choices": p["k_choices_calibrated"],
                                        "rounds": p["adaptive_calibrated"]["rounds"]},
            })
        compact["experiments"].append(e)
    js = "window.REAL_RUN_DATA = " + json.dumps(compact) + ";\n"
    with open(path, "w") as fh:
        fh.write(js)
    return len(js)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=int, default=0)
    ap.add_argument("--prompt", default="code")
    ap.add_argument("--lane", default="adaptive_calibrated")
    args = ap.parse_args()

    data, exp = load_trace(args.experiment)
    assets = os.path.join(ROOT, "assets")
    n_frames = build_film(exp, args.prompt, args.lane,
                          os.path.join(assets, "real_film.gif"))
    build_ledger(data, os.path.join(assets, "real_ledger.png"))
    n_js = build_app_data(data, os.path.join(ROOT, "app", "real_run_data.js"))
    print(f"film frames: {n_frames}")
    print(f"assets -> ['real_film.gif', 'real_ledger.png'], app data: {n_js} bytes")


if __name__ == "__main__":
    main()
