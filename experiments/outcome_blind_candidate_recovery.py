#!/usr/bin/env python3
"""Deterministic structural recovery candidates for exhausted MBO searches.

The stream contains hand-specified, locally validated structures only.  It has
no evaluator outputs, affinity labels, or model-derived ordering.  A method
may use it only after its normal proposal pool has been exhausted by duplicate
suppression; it is a liveness repair rather than a replacement search policy.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path


CONTRACT_ID = "outcome_blind_structural_recovery_v2"

# These are deliberately conventional, neutral organic structures.  The list
# was authored without consulting the released evaluator or any activity data.
_MBO_STRUCTURES = (
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "CCCc1ccccc1",
    "Oc1ccccc1",
    "COc1ccccc1",
    "CCOc1ccccc1",
    "Nc1ccccc1",
    "CNc1ccccc1",
    "CCNc1ccccc1",
    "Fc1ccccc1",
    "Clc1ccccc1",
    "Brc1ccccc1",
    "N#Cc1ccccc1",
    "O=C(O)c1ccccc1",
    "NC(=O)c1ccccc1",
    "COC(=O)c1ccccc1",
    "CCOC(=O)c1ccccc1",
    "CC(=O)c1ccccc1",
    "CS(=O)(=O)c1ccccc1",
    "c1ccc2ccccc2c1",
    "Cc1ccc2ccccc2c1",
    "Oc1ccc2ccccc2c1",
    "COc1ccc2ccccc2c1",
    "Nc1ccc2ccccc2c1",
    "Fc1ccc2ccccc2c1",
    "Clc1ccc2ccccc2c1",
    "Brc1ccc2ccccc2c1",
    "C1CCCCCC1",
    "CC1CCCCCC1",
    "OC1CCCCCC1",
    "NC1CCCCCC1",
    "FC1CCCCCC1",
    "C1CCCCCCC1",
    "CC1CCCCCCC1",
    "OC1CCCCCCC1",
    "NC1CCCCCCC1",
    "C1CCCCCCCCCCC1",
    "CC1CCCCCCCCCCC1",
    "OC1CCCCCCCCCCC1",
    "NC1CCCCCCCCCCC1",
    "O=C1CCCCCCCCCCC1",
    "NC(=O)C1CCCCCCCCCCC1",
    "C1CCC2CCCCC2C1",
    "CC1CCC2CCCCC2C1",
    "OC1CCC2CCCCC2C1",
    "NC1CCC2CCCCC2C1",
    "C1C2CC3CC1CC(C2)C3",
    "CC1C2CC3CC1CC(C2)C3",
    "OC1C2CC3CC1CC(C2)C3",
    "c1ccncc1",
    "Cc1ccncc1",
    "Oc1ccncc1",
    "Nc1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "CCN(CC)Cc1ccccc1",
    "CN(C)Cc1ccccc1",
    "COc1ccc(C)cc1",
    "CCOc1ccc(C)cc1",
    "COc1ccc(Cl)cc1",
    "COc1ccc(F)cc1",
    "CC(=O)Oc1ccccc1",
)

_NHO_AXES = (
    (24, 30, 36, 45, 54),
    (24, 36, 48, 60, 78),
    (3, 4, 5, 6, 8, 10),
    (60, 80, 100, 120, 150, 180),
    (400, 450, 500, 600, 700, 800),
    (0, 1, 2),
    (0, 1),
)

_SPO_STRUCTURES = (
    "La2Sr1Cu1O4", "La1.8Sr0.2Cu1O4", "La1.5Sr0.5Cu1O4",
    "La1Sr1Cu1O4", "La0.5Sr1.5Cu1O4", "La2Sr1Cu2O7",
    "La1Sr1Cu2O7", "La2Sr2Cu3O7", "La2Ba1Cu1O4",
    "La1.5Ba0.5Cu1O4", "La1Ba1Cu1O4", "La2Ba1Cu2O7",
    "La1Ba1Cu2O7", "La2Ba2Cu3O7", "Y1Ba2Cu3O6",
    "Y1Ba2Cu3O7", "Y1Ba2Cu2O6", "Y1Ba1Cu2O5",
    "Bi2Sr2Ca1Cu2O8", "Bi2Sr2Ca2Cu3O9", "Bi2Sr2Ca3Cu4O10",
    "Bi1Sr2Ca1Cu2O8", "Bi2Sr1Ca1Cu2O8", "Bi2Sr2Ca1Cu1O6",
    "Bi2Sr2Ca2Cu4O10", "Bi1Sr2Ca2Cu3O9", "Tl1Ba2Ca1Cu2O7",
    "Tl1Ba2Ca2Cu3O9", "Tl1Ba2Ca3Cu4O11", "Tl2Ba2Ca1Cu2O8",
    "Tl2Ba2Ca1Cu3O9", "Tl2Ba2Ca2Cu3O10", "Hg1Ba2Ca1Cu2O7",
    "Hg1Ba2Ca2Cu3O10", "Hg1Ba2Ca3Cu4O12", "Hg1Ba2Ca4Cu5O12",
    "Hg2Ba2Ca1Cu2O8", "Hg2Ba2Ca2Cu3O10", "Hg2Ba2Ca3Cu4O12",
    "Y2Ba2Cu3O7", "La2Sr1Cu3O8", "La2Ba1Cu3O8",
    "Bi2Sr2Ca3Cu5O12", "Tl1Ba2Ca2Cu4O11",
)

_TMC_NEUTRAL = (
    "RUCBEY-subgraph-1", "WECJIA-subgraph-3", "KEYRUB-subgraph-1",
    "NURKEQ-subgraph-2", "MEBXUN-subgraph-1", "BIFMOV-subgraph-1",
    "CUJYEL-subgraph-2", "EZEXEM-subgraph-1", "FOMVUB-subgraph-2",
    "EFIHEJ-subgraph-3", "LETTEL-subgraph-1", "KAKKIR-subgraph-3",
    "BICRIQ-subgraph-3", "UPEGAZ-subgraph-2", "CEVJAP-subgraph-2",
    "BABTUT-subgraph-3", "ZEJJEF-subgraph-3", "KULGAZ-subgraph-2",
    "CIGDAA-subgraph-1", "HOVMIP-subgraph-3", "ULUSIE-subgraph-1",
    "IBEKUV-subgraph-1", "REQSUD-subgraph-2", "BOSJIF-subgraph-1",
    "GUVMEP-subgraph-0",
)
_TMC_ANIONIC = (
    "MAZJIJ-subgraph-0", "OBONEA-subgraph-1", "CORTOU-subgraph-2",
    "LEVGUO-subgraph-2", "REBWEB-subgraph-2", "DOGPAS-subgraph-1",
    "IJIMIX-subgraph-3", "PEJGAN-subgraph-1", "BIFZEX-subgraph-0",
    "IRIXUC-subgraph-3", "SAYGOO-subgraph-0", "UROGIS-subgraph-1",
    "MAQKEX-subgraph-1", "LUQWUQ-subgraph-1", "QAYDID-subgraph-2",
    "MOYDOV-subgraph-3", "NIZQUK-subgraph-1", "SAYHIJ-subgraph-1",
    "CIQGOY-subgraph-0", "VUFZUT-subgraph-3", "ZOQFIU-subgraph-0",
    "GUQBUQ-subgraph-0", "LEZYUM-subgraph-2", "RAJXUX-subgraph-2",
    "QEWZOH-subgraph-3",
)


def _task_candidates(task: str) -> tuple[str, ...]:
    normalized = str(task).strip().upper()
    if normalized == "MBO":
        return _MBO_STRUCTURES
    if normalized == "NHO":
        return tuple(
            (
                f"fiber_radius={fiber},helix_radius={helix},n_turns={turns},"
                f"pitch={pitch},wavelength={wavelength},x_y_z={x_y_z},"
                f"direction={direction}"
            )
            for fiber, helix, turns, pitch, wavelength, x_y_z, direction in itertools.product(*_NHO_AXES)
        )
    if normalized == "SPO":
        return _SPO_STRUCTURES
    if normalized == "TMC":
        try:
            from TMC.src.ligands import LIGAND_POOL

            neutral = sorted(
                name for name, row in LIGAND_POOL.items()
                if float(row.get("charge", 0.0)) == 0.0
            )
            anionic = sorted(
                name for name, row in LIGAND_POOL.items()
                if float(row.get("charge", 0.0)) == -1.0
            )
        except Exception as exc:
            raise RuntimeError("TMC ligand pool is required for recovery") from exc
        return tuple(
            "Pd_" + "_".join((neutral_left, neutral_right, anion_left, anion_right))
            for neutral_left, neutral_right in itertools.combinations_with_replacement(neutral, 2)
            for anion_left, anion_right in itertools.combinations_with_replacement(anionic, 2)
        )
    raise ValueError(f"unsupported recovery task: {task}")


def generator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def recovery_stream(*, task: str, seed: int) -> list[str]:
    """Return a seed-specific permutation without reading any outcomes."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    candidates = _task_candidates(task)
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{CONTRACT_ID}:{seed}:{candidate}".encode("utf-8")
        ).hexdigest(),
    )
