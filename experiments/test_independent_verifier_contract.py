"""Offline tests for the independent/held-out provider attestation contract."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.independent_verifier_contract import (
    PROVENANCE_STEMS,
    SCHEMA_VERSION,
    load_and_validate_provider_attestation,
    validate_provider_attestation,
)


HASH = "a" * 64


def valid_attestation() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "MBO",
        "provider": "official_mbo",
        "primary_channel": {
            "id": "primary_surrogate_v1",
            "model_fingerprint": "model-primary",
            "data_partition": "train-primary",
        },
        "independent_channel": {
            "id": "independent_reference_v1",
            "model_fingerprint": "model-independent",
            "data_partition": "train-independent",
            "shared_training_data": False,
            "independence_audited": True,
        },
        "heldout_probe": {
            "data_partition": "heldout-mbo-v1",
            "disjoint_from_primary": True,
            "disjoint_from_independent": True,
            "source_labels_hidden": True,
        },
        "generalization_evidence": {
            "generalized_verifier": True,
            "exact_lookup_only": False,
            "protocol_frozen_before_fit": True,
            "two_model_training_partitions_disjoint": True,
            "heldout_partition_unused_for_training": True,
            "document_overlap_count_zero": True,
            "molecule_overlap_count_zero": True,
            "heldout_metrics_passed": True,
            "applicability_calibrated": True,
            "representative_coverage_passed": True,
            "on_policy_coverage_passed": True,
            "repeat_consistency_passed": True,
            "cross_fit_passed": True,
            "counterfactual_predictions_passed": True,
            "heldout_sample_count": 100,
            "representative_candidate_count": 20,
            "on_policy_candidate_count": 12,
        },
        "verification_cost_accounted": True,
        "provenance": {f"{stem}_sha256": HASH for stem in PROVENANCE_STEMS},
    }


def test_valid_provider_attestation_passes() -> None:
    result = validate_provider_attestation(
        valid_attestation(), task="MBO", provider="official_mbo"
    )
    assert result["passed"] is True
    assert result["errors"] == []


def test_same_model_and_shared_data_fail_closed() -> None:
    value = valid_attestation()
    value["independent_channel"]["model_fingerprint"] = "model-primary"  # type: ignore[index]
    value["independent_channel"]["shared_training_data"] = True  # type: ignore[index]
    result = validate_provider_attestation(
        value, task="MBO", provider="official_mbo"
    )
    assert result["passed"] is False
    assert any("model fingerprints" in error for error in result["errors"])
    assert any("shared_training_data=false" in error for error in result["errors"])


def test_heldout_partition_and_source_visibility_are_required() -> None:
    value = valid_attestation()
    value["heldout_probe"]["data_partition"] = "train-primary"  # type: ignore[index]
    value["heldout_probe"]["source_labels_hidden"] = False  # type: ignore[index]
    result = validate_provider_attestation(
        value, task="MBO", provider="official_mbo"
    )
    assert result["passed"] is False
    assert any("held-out data partition" in error for error in result["errors"])
    assert any("source labels" in error for error in result["errors"])


def test_exact_lookup_or_missing_on_policy_coverage_fails_closed() -> None:
    value = valid_attestation()
    value["generalization_evidence"]["exact_lookup_only"] = True  # type: ignore[index]
    value["generalization_evidence"]["on_policy_coverage_passed"] = False  # type: ignore[index]
    result = validate_provider_attestation(
        value, task="MBO", provider="official_mbo"
    )
    assert result["passed"] is False
    assert any("exact_lookup_only=false" in error for error in result["errors"])
    assert any("on_policy_coverage_passed=true" in error for error in result["errors"])


def test_wrong_task_or_provider_cannot_be_relabelled() -> None:
    value = copy.deepcopy(valid_attestation())
    result = validate_provider_attestation(
        value, task="NHO", provider="official_nho"
    )
    assert result["passed"] is False
    assert any("task does not match" in error for error in result["errors"])
    assert any("provider does not match" in error for error in result["errors"])


def test_legacy_sparse_attestation_cannot_pass_v2_contract() -> None:
    value = valid_attestation()
    value["schema_version"] = "independent_verifier_attestation.v1"
    value.pop("generalization_evidence")
    result = validate_provider_attestation(
        value, task="MBO", provider="official_mbo"
    )
    assert result["passed"] is False
    assert any("schema version" in error for error in result["errors"])
    assert any("generalization_evidence" in error for error in result["errors"])


def test_loaded_attestation_requires_matching_provenance_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = {stem: root / f"{stem}.bin" for stem in PROVENANCE_STEMS}
        for path in files.values():
            path.write_text(json.dumps({"path": path.name}) + "\n", encoding="utf-8")
        value = valid_attestation()
        provenance = value["provenance"]
        for stem, path in files.items():
            provenance[f"{stem}_path"] = path.name
            provenance[f"{stem}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        attestation = root / "attestation.json"
        attestation.write_text(json.dumps(value) + "\n", encoding="utf-8")

        result = load_and_validate_provider_attestation(
            attestation, task="MBO", provider="official_mbo"
        )
        assert result["passed"] is True

        files["independent_manifest"].write_text("tampered\n", encoding="utf-8")
        result = load_and_validate_provider_attestation(
            attestation, task="MBO", provider="official_mbo"
        )
        assert result["passed"] is False
        assert any("independent_manifest sha256" in error for error in result["errors"])
