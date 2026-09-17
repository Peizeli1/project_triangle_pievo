#!/usr/bin/env python3
"""Merge and audit the preregistered Gemini-2.5-Flash-Lite refresh attempts.

The refresh was executed in fixed tracks.  This audit never ranks retries by
score: for each cell it accepts the first complete artifact in the fixed
attempt order and records every incomplete artifact.  A cell is formal only
when it contains exactly 24 primary evaluator rows with contiguous indices and
the bridge attests the native Lite model, budget-zero configuration, and seed.

Artifact completion and protocol comparability are reported separately.  The
baseline launcher used one shared bridge log for parallel cells, so a baseline
cell with 24 evaluator rows is not treated as strictly comparable unless its
bridge calls can be attributed to that cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("MBO", "NHO", "SPO", "TMC")
BASELINE_METHODS = (
    "vanilla_mas",
    "react",
    "ai_scientist_v1",
    "ai_scientist_v2",
    "ai_researcher",
    "piflow",
)
METHODS = BASELINE_METHODS + ("pievo", "radp")
SEEDS = (0, 1, 2)
MODEL_ALIAS = "gemini-2.5-flash-lite-no-thinking"
NATIVE_MODEL = "gemini-2.5-flash-lite"
MU_REF = {"MBO": 6.5, "NHO": 2.0, "SPO": 298.5, "TMC": 493.8}
AUDIT_NAMES = {
    "MBO": "official_chembl_tool_audit.jsonl",
    "NHO": "official_nho_tool_audit.jsonl",
    "SPO": "official_spo_tool_audit.jsonl",
    "TMC": "official_tmc_tool_audit.jsonl",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_rows(task: str, run_dir: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(run_dir / AUDIT_NAMES[task])
    selected = [
        row
        for row in rows
        if str(row.get("tool_name", "")).startswith("evaluate_")
        and row.get("success") is True
        and row.get("duplicate_suppressed") is not True
        and row.get("record_type") not in {"duplicate_suppressed", "verification"}
        and finite(row.get("output"))
    ]
    return sorted(selected, key=lambda row: int(row.get("call_index", 0)))


def metrics(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["output"]) for row in rows]
    running: list[float] = []
    current = -math.inf
    for value in values:
        current = max(current, value)
        running.append(current)
    return {
        "primary_evaluations": len(rows),
        "primary_indices": [int(row.get("call_index", -1)) for row in rows],
        "sq_percent": max(values) / MU_REF[task] * 100.0 if values else None,
        "auoc_percent": statistics.fmean(running) / MU_REF[task] * 100.0 if running else None,
        "best_value": max(values) if values else None,
    }


def candidate_record(task: str, method: str, seed: int, run_dir: Path, attempt: str) -> dict[str, Any]:
    manifest = load_json(run_dir / "run_manifest.json") or {}
    summary = load_json(run_dir / "summary.json") or {}
    identity = manifest.get("run_identity", {}) if isinstance(manifest, dict) else {}
    runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
    generation = runtime.get("generation_config", {}) if isinstance(runtime, dict) else {}
    rows = primary_rows(task, run_dir)
    bridge_path = run_dir / "gemini_no_thinking_bridge.jsonl"
    if not bridge_path.is_file() and run_dir.parent.joinpath("gemini_no_thinking_bridge.jsonl").is_file():
        bridge_path = run_dir.parent / "gemini_no_thinking_bridge.jsonl"
    bridge_rows = load_jsonl(bridge_path)
    bridge_model_ok = bool(bridge_rows) and all(row.get("native_model") == NATIVE_MODEL for row in bridge_rows)
    bridge_thinking_ok = bool(bridge_rows) and all(row.get("no_thinking_verified") is True for row in bridge_rows)
    bridge_seed_ok = bool(bridge_rows) and all(
        row.get("seed_forwarded") is True and row.get("seed_requested") == seed
        for row in bridge_rows
    )
    is_baseline_schema = str(manifest.get("schema_version", "")).startswith("baseline_table1_")
    baseline_config_ok = bool(
        is_baseline_schema
        and generation.get("model") == MODEL_ALIAS
        and generation.get("enable_thinking") is False
        and runtime.get("seed_requested") == seed
        and runtime.get("seed_supported_by_provider") is True
    )
    artifact_complete = bool(
        manifest.get("status") == "completed"
        and identity.get("task") == task
        and identity.get("method") == method
        and identity.get("seed") == seed
        and identity.get("primary_budget") == 24
        and identity.get("model") == MODEL_ALIAS
        and len(rows) == 24
        and [int(row.get("call_index", -1)) for row in rows] == list(range(1, 25))
        and (runtime.get("native_model") == NATIVE_MODEL or generation.get("model") == MODEL_ALIAS)
        and (runtime.get("thinking_budget", 0) == 0)
        and (
            baseline_config_ok
            or (
                runtime.get("all_bridge_calls_native_model", bridge_model_ok) is True
                and runtime.get("all_bridge_calls_no_thinking", bridge_thinking_ok) is True
                and runtime.get("all_bridge_calls_forwarded_seed", bridge_seed_ok) is True
                and bridge_model_ok
                and bridge_thinking_ok
                and bridge_seed_ok
            )
        )
    )
    # A shared parent-level bridge log is useful provenance, but it does not
    # prove which bridge records belong to this particular parallel cell.
    # Strict comparability therefore requires a per-cell bridge log with at
    # least one call and an exact seed/model/no-thinking attestation.
    per_cell_bridge = bridge_path.parent.resolve() == run_dir.resolve()
    strict_bridge_ok = bool(
        per_cell_bridge
        and bridge_rows
        and all(row.get("native_model") == NATIVE_MODEL for row in bridge_rows)
        and all(row.get("no_thinking_verified") is True for row in bridge_rows)
        and all(row.get("seed_forwarded") is True and row.get("seed_requested") == seed for row in bridge_rows)
    )
    protocol_comparable = bool(
        artifact_complete
        and (
            strict_bridge_ok
            or (
                # Matched cells have a dedicated bridge log.  Baseline cells
                # deliberately do not pass through this fallback because a
                # shared bridge log cannot be attributed to a single cell.
                not is_baseline_schema
                and runtime.get("all_bridge_calls_native_model") is True
                and runtime.get("all_bridge_calls_no_thinking") is True
                and runtime.get("all_bridge_calls_forwarded_seed") is True
                and bridge_model_ok
                and bridge_thinking_ok
                and bridge_seed_ok
            )
        )
    )
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "attempt": attempt,
        "run_dir": str(run_dir.resolve()),
        "manifest_status": manifest.get("status"),
        "summary_status": summary.get("status"),
        "complete": artifact_complete,
        "protocol_comparable": protocol_comparable,
        "metrics": metrics(task, rows),
        "identity": {
            "model": identity.get("model"),
            "native_model": identity.get("native_model"),
            "track": identity.get("track"),
            "stage": identity.get("stage"),
            "candidate_recovery_contract": identity.get("candidate_recovery_contract"),
        },
        "runtime": {
            "native_model": runtime.get("native_model") or NATIVE_MODEL,
            "thinking_budget": runtime.get("thinking_budget"),
            "all_bridge_calls_native_model": runtime.get("all_bridge_calls_native_model", bridge_model_ok),
            "all_bridge_calls_no_thinking": runtime.get("all_bridge_calls_no_thinking", bridge_thinking_ok),
            "all_bridge_calls_forwarded_seed": runtime.get("all_bridge_calls_forwarded_seed", bridge_seed_ok),
            "generation_model": generation.get("model"),
            "generation_enable_thinking": generation.get("enable_thinking"),
            "bridge_calls": len(bridge_rows),
            "bridge_model_ok": bridge_model_ok,
            "bridge_thinking_ok": bridge_thinking_ok,
            "bridge_seed_ok": bridge_seed_ok,
            "baseline_config_ok": baseline_config_ok,
            "per_cell_bridge": per_cell_bridge,
            "strict_bridge_ok": strict_bridge_ok,
            "bridge_path": str(bridge_path.resolve()),
        },
        "failure": {
            "exit_code": manifest.get("exit_code"),
            "error": summary.get("error"),
            "primary_count": len(rows),
            "bridge_violations": sum(not bool(row.get("no_thinking_verified")) for row in bridge_rows),
        },
    }


def path_from_map(run_map: dict[str, Any], key: str) -> Path | None:
    value = run_map.get(key)
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def collect_matched(base: Path, task: str, method: str, seed: int) -> list[dict[str, Any]]:
    root = base / "pievo_radp"
    run_map = load_json(root / "matched_run_map.json") or {}
    key = f"{method}:{task}:{seed}"
    path = path_from_map(run_map, key)
    if path is None:
        return []
    return [candidate_record(task, method, seed, path, "recovery02")]


def collect_baseline(roots: list[tuple[str, Path]], task: str, method: str, seed: int) -> list[dict[str, Any]]:
    key = f"{method}:{task}:{seed}"
    candidates: list[dict[str, Any]] = []
    for attempt, root in roots:
        run_map = load_json(root / "baselines" / "baseline_run_map.json") or {}
        path = path_from_map(run_map, key)
        if path is not None and path.is_dir():
            candidates.append(candidate_record(task, method, seed, path, attempt))
    return candidates


def choose(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Fixed attempt order only.  This deliberately does not inspect SQ/AUOC.
    for candidate in candidates:
        if candidate["complete"]:
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matched_root = args.matched_root.resolve()
    baseline_roots = [(f"baseline_attempt_{index + 1}", root.resolve()) for index, root in enumerate(args.baseline_root)]
    cells: list[dict[str, Any]] = []
    for task in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                if method in {"pievo", "radp"}:
                    candidates = collect_matched(matched_root, task, method, seed)
                else:
                    candidates = collect_baseline(baseline_roots, task, method, seed)
                selected = choose(candidates)
                cells.append(
                    {
                        "key": f"{method}:{task}:{seed}",
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "selected": selected,
                        "candidates": candidates,
                        "complete": selected is not None,
                    }
                )

    complete_cells = [cell for cell in cells if cell["complete"]]
    task_leaders: dict[str, dict[str, str | None]] = {}
    comparable_task_leaders: dict[str, dict[str, str | None]] = {}
    for task in TASKS:
        task_leaders[task] = {}
        comparable_task_leaders[task] = {}
        for metric in ("sq_percent", "auoc_percent"):
            values = [
                (method, selected["metrics"][metric])
                for cell in complete_cells
                if cell["task"] == task
                and (selected := cell.get("selected")) is not None
                and selected["method"] in METHODS
                and selected["metrics"].get(metric) is not None
            ]
            task_leaders[task][metric] = max(values, key=lambda item: float(item[1]))[0] if values else None
            comparable_values = [
                (method, selected["metrics"][metric])
                for cell in complete_cells
                if cell["task"] == task
                and (selected := cell.get("selected")) is not None
                and selected.get("protocol_comparable") is True
                and selected["metrics"].get(metric) is not None
            ]
            comparable_task_leaders[task][metric] = (
                max(comparable_values, key=lambda item: float(item[1]))[0]
                if comparable_values else None
            )

    comparable_cells = [
        cell for cell in complete_cells
        if isinstance(cell.get("selected"), dict)
        and cell["selected"].get("protocol_comparable") is True
    ]

    result = {
        "schema_version": "gemini25flashlite_nothinking_refresh_audit.v1",
        "model_alias": MODEL_ALIAS,
        "native_model": NATIVE_MODEL,
        "thinking_budget": 0,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "primary_budget_per_cell": 24,
        "cell_count": len(cells),
        "complete_cell_count": len(complete_cells),
        "protocol_comparable_cell_count": len(comparable_cells),
        "complete_but_not_protocol_comparable_cell_count": len(complete_cells) - len(comparable_cells),
        "incomplete_cell_count": len(cells) - len(complete_cells),
        "matched_root": str(matched_root),
        "baseline_roots": [{"attempt": attempt, "root": str(root)} for attempt, root in baseline_roots],
        "selection_rule": "fixed attempt order; select only complete artifacts; never rank or select by SQ/AUOC",
        "task_metric_leaders_over_artifact_complete_cells": task_leaders,
        "task_metric_leaders_over_protocol_comparable_cells": comparable_task_leaders,
        "cells": cells,
        "claim_boundary": {
            "formal_matrix_complete": len(complete_cells) == len(cells) and len(comparable_cells) == len(cells),
            "supported": "Only artifact-complete cells with 24 contiguous primary evaluations and cell-attributable strict Lite bridge attestation are comparable.",
            "not_supported": [
                "A full 96-cell Gemini matrix when any cell is incomplete.",
                "Comparing baseline cells whose shared bridge log cannot be attributed to that cell.",
                "Universal SOTA or overall SOTA claims.",
                "Treating partial trajectories as 24-evaluation results.",
                "Treating the no-thinking bridge as evidence that the upstream provider always honored budget zero when positive thought tokens were observed.",
            ],
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = output.with_suffix(".md")
    lines = [
        "# Gemini-2.5-Flash-Lite refresh audit",
        "",
        f"- Artifact-complete cells: **{result['complete_cell_count']}/{result['cell_count']}**",
        f"- Strictly protocol-comparable cells: **{result['protocol_comparable_cell_count']}/{result['cell_count']}**",
        f"- Complete but not cell-attributable: **{result['complete_but_not_protocol_comparable_cell_count']}**",
        f"- Native model: `{NATIVE_MODEL}`",
        "- Thinking contract: `thinkingBudget=0`, `includeThoughts=false`, strict positive-token rejection",
        "- Cell selection: fixed attempt order; no score-based retry selection",
        "",
        "## Task-wise leaders among protocol-comparable cells",
        "",
        "| Task | SQ leader | AUOC leader |",
        "|---|---|---|",
    ]
    for task in TASKS:
        lines.append(f"| {task} | {comparable_task_leaders[task]['sq_percent'] or 'N/A'} | {comparable_task_leaders[task]['auoc_percent'] or 'N/A'} |")
    lines.extend(["", "## Incomplete cells", "", "| Cell | Failure summary |", "|---|---|"])
    for cell in cells:
        if cell["complete"]:
            continue
        summaries = []
        for candidate in cell["candidates"]:
            failure = candidate.get("failure", {})
            summaries.append(f"{candidate['attempt']}: primary={failure.get('primary_count')}, error={failure.get('error')}")
        lines.append(f"| `{cell['key']}` | {'; '.join(summaries) or 'no artifact'} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "complete_cells": len(complete_cells), "cell_count": len(cells)}, ensure_ascii=False))
    return 0 if len(complete_cells) == len(cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
