"""Zero-API tests for the frozen G1 fairness protocol."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_g1_audit_passes_without_network_or_api_key() -> None:
    result = json.loads(
        (ROOT / "experiments/results/g1_fairness_audit_v8.json").read_text(encoding="utf-8")
    )
    assert result["passed"] is True
    assert result["checks"]["same_candidate_validator_for_common_arms"] is True
