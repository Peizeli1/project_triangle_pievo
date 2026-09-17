#!/usr/bin/env python3
"""Outcome-blind common candidate generator frozen by the G1 protocol.

This is intentionally small and deterministic. It is a fairness control, not
the final AFE generator from G3/G4. Both common-generator arms receive exactly
the same ordered structures for a given task, seed, and slot.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from pathlib import Path

GENERATOR_ID = "radp_common_generator_v1"
TASKS = ("MBO", "NHO", "SPO", "TMC")
_CANDIDATES = (
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "COc1ccccc1",
    "Clc1ccccc1",
    "c1ccc2ccccc2c1",
    "Cc1ccc2ccccc2c1",
    "Oc1ccc2ccccc2c1",
    "c1ccc2c(c1)oc1ccccc12",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "C1CCCCCCC1",
    "CC1CCCCCCC1",
    "OC1CCCCCCC1",
    "C1CCCCCCCCCCC1",
    "CC1CCCCCCCCCCC1",
    "O=C1CCCCCCCCCCC1",
    "C1CCC2CCCCC2C1",
    "C1C2CC3CC1CC(C2)C3",
    "C1CCC2(CC1)CCCCC2",
    "CC1CC2CCC1C(C)C2",
    "O=C1CC2(c3ccccc3)CCC1C2",
    "OC1CC2(c3ccccc3)CCC1C2",
    "NC(=O)C1CCCCCCCCCCC1",
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

_SPO_CANDIDATES = (
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
    "Hg1Ba2Ca2Cu3O9", "Hg1Ba2Ca3Cu4O10", "Hg1Ba2Ca4Cu5O12",
    "Hg2Ba2Ca1Cu2O8", "Hg2Ba2Ca2Cu3O10", "Hg2Ba2Ca3Cu4O12",
    "Y1Ba2Cu3O8", "Y2Ba2Cu3O7", "La2Sr1Cu3O8",
    "La2Ba1Cu3O8", "Bi2Sr2Ca3Cu5O12", "Tl1Ba2Ca2Cu4O11",
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
    "IJIMIX-subgraph-1", "PEJGAN-subgraph-1", "BIFZEX-subgraph-0",
    "IRIXUC-subgraph-3", "SAYGOO-subgraph-0", "UROGIS-subgraph-1",
    "MAQKEX-subgraph-1", "LUQWUQ-subgraph-1", "QAYDID-subgraph-2",
    "MOYDOV-subgraph-3", "NIZQUK-subgraph-1", "SAYHIJ-subgraph-1",
    "CIQGOY-subgraph-0", "VUFZUT-subgraph-1", "ZOQFIU-subgraph-0",
    "GUQBUQ-subgraph-0", "LEZYUM-subgraph-2", "RAJXUX-subgraph-2",
    "QEWZOH-subgraph-3",
)


def _task_candidates(task: str) -> tuple[str, ...]:
    normalized = str(task).strip().upper()
    if normalized == "MBO":
        return _CANDIDATES
    if normalized == "NHO":
        return tuple(
            f"fiber_radius={fiber},helix_radius={helix},n_turns={turns},"
            f"pitch={pitch},wavelength={wavelength},x_y_z={x_y_z},direction={direction}"
            for fiber, helix, turns, pitch, wavelength, x_y_z, direction in itertools.product(*_NHO_AXES)
        )
    if normalized == "SPO":
        return _SPO_CANDIDATES
    if normalized == "TMC":
        return tuple(
            "Pd_" + "_".join((neutral_left, neutral_right, anion_left, anion_right))
            for neutral_left, neutral_right in itertools.combinations_with_replacement(_TMC_NEUTRAL, 2)
            for anion_left, anion_right in itertools.combinations_with_replacement(_TMC_ANIONIC, 2)
        )
    raise ValueError(f"unsupported common-generator task: {task}")


def generator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_candidate(smiles: str) -> bool:
    try:
        from pievo.tools.candidate_validation import validate_mbo

        validate_mbo(smiles)
        return True
    except Exception:
        return False


def generate(*, task: str, seed: int, slot: int, count: int = 3) -> list[str]:
    normalized = str(task).strip().upper()
    candidates = _task_candidates(normalized)
    if seed < 0 or slot < 0 or count <= 0 or count > len(candidates):
        raise ValueError("invalid common-generator dimensions")
    order = list(candidates)
    random.Random((int(seed) + 1) * 1009 + int(slot) * 9176).shuffle(order)
    output: list[str] = []
    for candidate in order:
        canonical = candidate
        if normalized == "MBO":
            try:
                from pievo.tools.candidate_validation import validate_mbo

                canonical = validate_mbo(candidate).canonical
            except Exception:
                canonical = None
        if canonical and canonical not in output:
            output.append(canonical)
        if len(output) == count:
            break
    if len(output) != count:
        raise RuntimeError("frozen common generator could not produce enough valid candidates")
    return output


def replay(*, task: str, seed: int, slots: int = 8, count: int = 3) -> list[list[str]]:
    return [generate(task=task, seed=seed, slot=slot, count=count) for slot in range(slots)]
