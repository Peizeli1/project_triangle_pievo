from __future__ import annotations

import json
import os
from pathlib import Path

from pievo.tools import tool_chembl_official as chembl_tool


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, candidate: str) -> float:
        self.calls.append(candidate)
        return 4.25


def test_afe_authorization_repairs_model_rewrite_without_spending_invalid_primary(
    tmp_path: Path, monkeypatch
) -> None:
    predictor = _Predictor()
    monkeypatch.setattr(chembl_tool, "_PREDICTOR", predictor)
    monkeypatch.setattr(chembl_tool, "_CALL_INDEX", 0)
    monkeypatch.setattr(chembl_tool, "_INVALID_ATTEMPTS", 0)
    monkeypatch.setattr(chembl_tool, "_SUCCESSFUL_VALUES", {})
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))
    monkeypatch.setenv("PIEVO_AFE_AUTHORIZED_CANDIDATE", "c1ccccc1")
    monkeypatch.setenv("PIEVO_AFE_DUPLICATE_RETRY_REQUESTED", "0")
    monkeypatch.setenv("PIEVO_MAX_PRIMARY_EVALS", "6")
    monkeypatch.setenv("PIEVO_MBO_FORBIDDEN_CANDIDATES", "")
    monkeypatch.setenv("PIEVO_CORRUPTION_MODE", "none")

    result = chembl_tool.evaluate_chembl("not a smiles [")

    assert result["success"] is True
    assert result["input"] == "c1ccccc1"
    assert result["candidate_mismatch_repaired"] is True
    assert result["call_index"] == 1
    assert predictor.calls == ["c1ccccc1"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "official_chembl_tool_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    mismatch = [row for row in rows if row.get("record_type") == "candidate_mismatch"]
    primaries = [
        row
        for row in rows
        if row.get("tool_name") == "evaluate_chembl"
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
    ]
    assert len(mismatch) == 1
    assert mismatch[0]["primary_consumed"] is False
    assert len(primaries) == 1


def test_afe_duplicate_requests_signal_fresh_candidate(tmp_path: Path, monkeypatch) -> None:
    predictor = _Predictor()
    monkeypatch.setattr(chembl_tool, "_PREDICTOR", predictor)
    monkeypatch.setattr(chembl_tool, "_CALL_INDEX", 0)
    monkeypatch.setattr(chembl_tool, "_INVALID_ATTEMPTS", 0)
    monkeypatch.setattr(chembl_tool, "_SUCCESSFUL_VALUES", {})
    monkeypatch.setenv("WORKING_RESULT_DIR", str(tmp_path))
    monkeypatch.setenv("PIEVO_AFE_AUTHORIZED_CANDIDATE", "c1ccccc1")
    monkeypatch.setenv("PIEVO_AFE_DUPLICATE_RETRY_REQUESTED", "0")
    monkeypatch.setenv("PIEVO_MAX_PRIMARY_EVALS", "6")
    monkeypatch.setenv("PIEVO_MBO_FORBIDDEN_CANDIDATES", "")
    monkeypatch.setenv("PIEVO_CORRUPTION_MODE", "none")

    first = chembl_tool.evaluate_chembl("c1ccccc1")
    duplicate = chembl_tool.evaluate_chembl("c1ccccc1")

    assert first["success"] is True
    assert duplicate["duplicate_suppressed"] is True
    assert duplicate["retry_requested"] is True
    assert predictor.calls == ["c1ccccc1"]
    assert os.getenv("PIEVO_AFE_DUPLICATE_RETRY_REQUESTED") == "1"
