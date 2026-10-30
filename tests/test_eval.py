"""Acceptance eval against a running server (skipped unless EVAL_BASE_URL is set)."""

import os
import time
from pathlib import Path

import httpx
import pytest
import yaml

BASE_URL = os.environ.get("EVAL_BASE_URL")
MAX_LATENCY_S = float(os.environ.get("EVAL_MAX_LATENCY_S", "20"))
CASES = yaml.safe_load((Path(__file__).parent / "eval_questions.yaml").read_text())["questions"]

pytestmark = pytest.mark.skipif(not BASE_URL, reason="set EVAL_BASE_URL to run the eval against a server")


def _source_matches(citation: dict, expected: dict) -> bool:
    if expected["title_contains"].lower() not in citation["title"].lower():
        return False
    return "page" not in expected or citation["page"] == expected["page"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_eval_question(case):
    if case.get("todo"):
        pytest.skip("placeholder question; fill in eval_questions.yaml")

    start = time.perf_counter()
    r = httpx.post(f"{BASE_URL.rstrip('/')}/api/chat", json={"question": case["question"]}, timeout=120)
    elapsed = time.perf_counter() - start
    r.raise_for_status()
    body = r.json()

    assert body["foundInMaterials"] is case["expect_found"], body["answer"]
    if case["expect_found"]:
        expected = case.get("expect_sources", [])
        assert not expected or any(
            _source_matches(c, e) for c in body["citations"] for e in expected
        ), f"expected one of {expected}, got {body['citations']}"
        for phrase in case.get("answer_contains", []):
            assert phrase.lower() in body["answer"].lower()
    else:
        assert body["citations"] == []
    assert elapsed <= MAX_LATENCY_S, f"took {elapsed:.1f}s"
