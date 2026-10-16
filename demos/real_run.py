"""Run the speculative-decoding harness on real HF model pairs and write receipts/real_run.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdecode.engine import PROMPTS, run_pair  # noqa: E402

PAIRS = [
    ("gpt2", "distilgpt2"),
    ("gpt2-medium", "distilgpt2"),
]

MAX_NEW_TOKENS = 16
FIXED_KS = (4, 8)


def main() -> None:
    receipts = []
    for target_id, drafter_id in PAIRS:
        print(f"[real_run] running {target_id} <- {drafter_id} ...", flush=True)
        result = run_pair(
            target_id,
            drafter_id,
            PROMPTS,
            max_new_tokens=MAX_NEW_TOKENS,
            fixed_ks=FIXED_KS,
        )
        # assert token-for-token identity vs naive for every non-naive run
        for run in result["runs"]:
            if run["lane"] != "naive" and not run["matches_reference"]:
                raise RuntimeError(
                    f"token mismatch for {result['pair']} prompt={run['prompt']!r} lane={run['lane']}"
                )
        receipts.append(result)
        print(
            f"   c={result['drafter_cost_c']}  "
            f"naive tok/call=1.0  "
            f"adaptive={[r['tokens_per_target_call'] for r in result['runs'] if r['lane']=='adaptive']}",
            flush=True,
        )

    out = ROOT / "receipts" / "real_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "fixed_ks": list(FIXED_KS),
        "prompts": PROMPTS,
        "pairs": receipts,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    # default torch threading
    import os

    main()