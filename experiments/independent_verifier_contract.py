"""Fail-closed attestation contract for independent and held-out providers.

The released PiEvo checkpoints are useful for matched evaluator smoke tests,
but they are not independent evidence.  This module defines the minimum
machine-readable proof a future provider must supply before RADP can be marked
formal-ready.  It performs no network access and never infers independence
from a provider name alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "independent_verifier_attestation.v2"

PROVENANCE_STEMS = (
    "primary_manifest",
    "independent_manifest",
    "heldout_manifest",
    "independence_audit",
    "protocol",
    "dataset_manifest",
    "metrics",
    "coverage",
    "model_a",
    "model_b",
    "repeat",
    "cross_fit",
    "counterfactual",
    "feature_encoder_manifest",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_provider_attestation(
    value: Mapping[str, Any] | None,
    *,
    task: str,
    provider: str,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a provider's independence and held-out evidence declaration.

    A valid declaration requires distinct channel identities, distinct model
    and data fingerprints, explicit no-shared-training-data evidence, a
    disjoint held-out partition, hidden source labels, and auditable cost and
    provenance hashes.  Missing fields fail closed.
    """

    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "provider": provider,
            "passed": False,
            "errors": ["provider attestation is missing or not an object"],
        }

    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("provider attestation schema version is missing or unsupported")
    if value.get("task") != task:
        errors.append(f"provider attestation task does not match {task}")
    if value.get("provider") != provider:
        errors.append(f"provider attestation provider does not match {provider}")

    primary = value.get("primary_channel")
    independent = value.get("independent_channel")
    heldout = value.get("heldout_probe")
    provenance = value.get("provenance")
    generalization = value.get("generalization_evidence")
    if not isinstance(primary, Mapping):
        errors.append("primary_channel declaration is missing")
        primary = {}
    if not isinstance(independent, Mapping):
        errors.append("independent_channel declaration is missing")
        independent = {}
    if not isinstance(heldout, Mapping):
        errors.append("heldout_probe declaration is missing")
        heldout = {}
    if not isinstance(provenance, Mapping):
        errors.append("provenance declaration is missing")
        provenance = {}
    if not isinstance(generalization, Mapping):
        errors.append("generalization_evidence declaration is missing")
        generalization = {}

    primary_id = primary.get("id")
    independent_id = independent.get("id")
    if not _nonempty_string(primary_id) or not _nonempty_string(independent_id):
        errors.append("primary and independent channel ids are required")
    elif primary_id == independent_id:
        errors.append("primary and independent channel ids must differ")

    primary_model = primary.get("model_fingerprint")
    independent_model = independent.get("model_fingerprint")
    if not _nonempty_string(primary_model) or not _nonempty_string(independent_model):
        errors.append("primary and independent model fingerprints are required")
    elif primary_model == independent_model:
        errors.append("primary and independent model fingerprints must differ")

    primary_partition = primary.get("data_partition")
    independent_partition = independent.get("data_partition")
    if not _nonempty_string(primary_partition) or not _nonempty_string(independent_partition):
        errors.append("primary and independent data partitions are required")
    elif primary_partition == independent_partition:
        errors.append("primary and independent data partitions must differ")

    if independent.get("shared_training_data") is not False:
        errors.append("independent channel must explicitly declare shared_training_data=false")
    if independent.get("independence_audited") is not True:
        errors.append("independent channel must have an explicit independence audit")

    heldout_partition = heldout.get("data_partition")
    if not _nonempty_string(heldout_partition):
        errors.append("held-out data partition is required")
    elif heldout_partition in {primary_partition, independent_partition}:
        errors.append("held-out data partition must be disjoint from both channels")
    if heldout.get("disjoint_from_primary") is not True:
        errors.append("held-out probe must declare disjoint_from_primary=true")
    if heldout.get("disjoint_from_independent") is not True:
        errors.append("held-out probe must declare disjoint_from_independent=true")
    if heldout.get("source_labels_hidden") is not True:
        errors.append("held-out probe must hide source labels from the policy")

    required_generalization_flags = (
        "generalized_verifier",
        "protocol_frozen_before_fit",
        "two_model_training_partitions_disjoint",
        "heldout_partition_unused_for_training",
        "document_overlap_count_zero",
        "molecule_overlap_count_zero",
        "heldout_metrics_passed",
        "applicability_calibrated",
        "representative_coverage_passed",
        "on_policy_coverage_passed",
        "repeat_consistency_passed",
        "cross_fit_passed",
        "counterfactual_predictions_passed",
    )
    for name in required_generalization_flags:
        if generalization.get(name) is not True:
            errors.append(f"generalization evidence requires {name}=true")
    if generalization.get("exact_lookup_only") is not False:
        errors.append("generalized verifier must declare exact_lookup_only=false")
    for name in ("heldout_sample_count", "representative_candidate_count", "on_policy_candidate_count"):
        count = generalization.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            errors.append(f"generalization evidence requires positive {name}")

    if value.get("verification_cost_accounted") is not True:
        errors.append("verification_cost_accounted must be true")
    for stem in PROVENANCE_STEMS:
        if not _is_sha256(provenance.get(f"{stem}_sha256")):
            errors.append(f"{stem} sha256 is missing or invalid")

    if base_dir is not None:
        for stem in PROVENANCE_STEMS:
            hash_key = f"{stem}_sha256"
            path_value = provenance.get(f"{stem}_path")
            if not _nonempty_string(path_value):
                errors.append(f"{stem} path is missing")
                continue
            path = Path(str(path_value))
            if not path.is_absolute():
                path = base_dir / path
            actual = sha256(path)
            if actual is None:
                errors.append(f"{stem} path does not exist")
            elif actual != provenance.get(hash_key):
                errors.append(f"{stem} sha256 does not match its recorded file")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "provider": provider,
        "passed": not errors,
        "errors": errors,
    }


def load_and_validate_provider_attestation(
    path: Path,
    *,
    task: str,
    provider: str,
) -> dict[str, Any]:
    """Load a local attestation and validate it without network access."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "provider": provider,
            "passed": False,
            "errors": [f"could not load provider attestation: {type(exc).__name__}"],
        }
    return validate_provider_attestation(
        value, task=task, provider=provider, base_dir=path.parent
    )


def sha256(path: Path) -> str | None:
    """Hash helper used by future provider builders and tests."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
