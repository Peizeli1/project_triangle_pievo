#!/usr/bin/env python3
"""Frozen, structure-only applicability domain for the released MBO surrogate.

The envelope is derived from the canonical SMILES structure distribution of the
50,000-row ChEMBL35 development pool.  Only structure-derived descriptors are
used at runtime; activity columns are intentionally ignored.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors


CONTRACT_ID = "mbo_structural_envelope_v2"
REFERENCE_POOL_SHA256 = "9eb38b040dd03a4bfc7495cda9ae4642a7de7ccea2ab6b5a7c4af6c5218e2ee4"
REFERENCE_POOL_ROWS = 50000
REFERENCE_VALID_UNIQUE_ROWS = 28299
THRESHOLDS = {
    "heavy_atoms_max": 44,
    "rotatable_bonds_max": 13,
    "exact_mol_wt_max": 615.34206954,
    "logp_max": 7.5703,
}


def assess_mbo_applicability(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return {
            "contract_id": CONTRACT_ID,
            "accepted": False,
            "reason": "RDKit could not parse canonical SMILES",
            "features": {},
        }
    features = {
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "exact_mol_wt": float(rdMolDescriptors.CalcExactMolWt(molecule)),
        "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
    }
    violations = [
        key
        for key, threshold_key, operator in (
            ("heavy_atoms", "heavy_atoms_max", lambda value, limit: value > limit),
            ("rotatable_bonds", "rotatable_bonds_max", lambda value, limit: value > limit),
            ("exact_mol_wt", "exact_mol_wt_max", lambda value, limit: value > limit),
            ("logp", "logp_max", lambda value, limit: value > limit),
        )
        if operator(features[key], THRESHOLDS[threshold_key])
    ]
    return {
        "contract_id": CONTRACT_ID,
        "accepted": not violations,
        "reason": "; ".join(f"{key} exceeds frozen reference envelope" for key in violations) or None,
        "violations": violations,
        "features": features,
        "thresholds": THRESHOLDS,
        "reference_pool_sha256": REFERENCE_POOL_SHA256,
        "reference_pool_rows": REFERENCE_POOL_ROWS,
        "reference_valid_unique_rows": REFERENCE_VALID_UNIQUE_ROWS,
    }
