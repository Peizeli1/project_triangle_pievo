"""Zero-API tests for protocol freeze and explicit formal blocking."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments import sota_gate


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_g1_freeze_is_bound_to_g0_artifacts() -> None:
    protocol = load(ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json")
    readiness = load(ROOT / "experiments/results/official_four_task_readiness_v8.json")
    preflight = load(ROOT / "experiments/results/official_formal_preflight_afe_v8.json")
    freeze = load(ROOT / "experiments/results/matched_qwen235b_protocol_freeze_afe_v8.json")
    assert sota_gate.freeze_matches(protocol, readiness, preflight, freeze)


def test_stale_g1_protocol_hash_fails_closed() -> None:
    protocol = load(ROOT / "experiments/configs/radp_search_afe_matched_qwen235b_protocol_v8.json")
    readiness = load(ROOT / "experiments/results/official_four_task_readiness_v8.json")
    preflight = load(ROOT / "experiments/results/official_formal_preflight_afe_v8.json")
    freeze = copy.deepcopy(
        load(ROOT / "experiments/results/matched_qwen235b_protocol_freeze_afe_v8.json")
    )
    freeze["protocol_sha256"] = "0" * 64
    assert not sota_gate.freeze_matches(protocol, readiness, preflight, freeze)
