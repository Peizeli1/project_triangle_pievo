import json
from pathlib import Path

from experiments.formal_run_approval import radp_afe_audit_ok


def _write_rows(path: Path, schemas: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"schema_version": schema}) + "\n" for schema in schemas),
        encoding="utf-8",
    )


def test_spo_afe_audit_schema_is_accepted(tmp_path: Path) -> None:
    audit = tmp_path / "radp_candidate_portfolio_audit.jsonl"
    _write_rows(
        audit,
        ["radp_afe_portfolio.spo_v1", "radp_afe_acquisition.spo_v1"],
    )

    assert radp_afe_audit_ok(audit) is True


def test_partial_afe_audit_schema_is_rejected(tmp_path: Path) -> None:
    audit = tmp_path / "radp_candidate_portfolio_audit.jsonl"
    _write_rows(audit, ["radp_afe_portfolio.spo_v1"])

    assert radp_afe_audit_ok(audit) is False
