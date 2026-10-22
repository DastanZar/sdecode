"""Real-model receipts: token-for-token identity and schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RECEIPT = ROOT / "receipts" / "real_run.json"


def test_receipt_schema_and_identity():
    assert RECEIPT.exists(), "run demos/real_run.py first"
    data = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert isinstance(data["pairs"], list) and len(data["pairs"]) >= 1
    for pair in data["pairs"]:
        assert "drafter_cost_c" in pair
        assert "runs" in pair and len(pair["runs"]) > 0
        for run in pair["runs"]:
            for key in (
                "lane",
                "prompt",
                "target_calls",
                "drafter_calls",
                "generated_tokens",
                "tokens_per_target_call",
                "token_ids",
                "matches_reference",
                "generated_text",
            ):
                assert key in run, (pair["pair"], key)
            assert isinstance(run["token_ids"], list)
            # naive is trivially identical; speculative lanes must match naive
            assert run["matches_reference"] is True, (pair["pair"], run["lane"], run["prompt"])


def test_engine_identity_small_run():
    # Guard for environments without cached HF weights / network.
    try:
        import transformers  # noqa: F401
    except Exception:
        pytest.skip("transformers not available")
    from sdecode.engine import load_pair, run_lane, measure_drafter_cost, encode

    target, drafter, tok = load_pair("gpt2", "distilgpt2")
    c = measure_drafter_cost(target, drafter, encode(tok, "The quick brown fox"))
    prompt = "The quick brown fox jumps over"
    naive = run_lane(target, drafter, tok, prompt, 8, "naive", c,
                     model_pair="gpt2<-distilgpt2", target_id="gpt2", drafter_id="distilgpt2")
    fixed = run_lane(target, drafter, tok, prompt, 8, "fixed", c, fixed_k=4,
                     model_pair="gpt2<-distilgpt2", target_id="gpt2", drafter_id="distilgpt2")
    assert fixed.token_ids == naive.token_ids