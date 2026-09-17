#!/usr/bin/env python3
"""Audit source attribution and active-gating metrics for the zero-API benchmark.

The input is a deterministic evaluator-side benchmark.  This script does not
choose seeds, methods, or outcomes; it scores every recorded event under the
frozen protocol and writes a paper-ineligible exploratory artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_CLASSES = (
    "observation_noise",
    "surrogate_local_failure",
    "local_or_non_generalizing_mechanism",
    "true_mechanism_change",
)
ANOMALOUS_CONDITIONS = {
    "noise": "observation_noise",
    "surrogate_failure": "surrogate_local_failure",
    "local_artifact": "local_or_non_generalizing_mechanism",
    "true_mechanism": "true_mechanism_change",
    # The initial event is a real mechanism and the later reversal is scored
    # separately through final-state accuracy and backtracking metrics.
    "mechanism_reversal": "true_mechanism_change",
}
DECISION_TO_SOURCE = {
    "repeat_disagrees_noise": "observation_noise",
    "independent_disagrees_surrogate": "surrogate_local_failure",
    "heldout_fails_local_or_spurious": "local_or_non_generalizing_mechanism",
    "heldout_generalizes": "true_mechanism_change",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return statistics.mean(values) if values else None


def macro_f1(y_true: list[str], y_pred: list[str | None]) -> dict[str, Any]:
    per_class: dict[str, float] = {}
    for label in SOURCE_CLASSES:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred) if b is not None)
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "macro_f1": mean(per_class.values()),
        "per_class_f1": per_class,
        "n": len(y_true),
        "predicted_n": sum(value is not None for value in y_pred),
        "coverage": sum(value is not None for value in y_pred) / len(y_true) if y_true else 0.0,
    }


def calibration(rows: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
    """Compute multiclass Brier and confidence ECE for recorded posteriors."""
    if not rows:
        return {"n": 0, "brier": None, "ece_10bin": None}
    briers: list[float] = []
    confidence_correct: list[tuple[float, bool]] = []
    for truth, posterior in rows:
        probs = [float(posterior.get(label, 0.0)) for label in SOURCE_CLASSES]
        briers.append(sum((prob - float(label == truth)) ** 2 for prob, label in zip(probs, SOURCE_CLASSES)))
        prediction = SOURCE_CLASSES[max(range(len(probs)), key=probs.__getitem__)]
        confidence_correct.append((max(probs), prediction == truth))
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for confidence, correct in confidence_correct:
        index = min(9, int(confidence * 10.0))
        bins[index].append((confidence, correct))
    ece = 0.0
    for bucket in bins:
        if bucket:
            ece += len(bucket) / len(confidence_correct) * abs(
                statistics.mean(c for c, _ in bucket) - statistics.mean(float(x) for _, x in bucket)
            )
    return {"n": len(rows), "brier": mean(briers), "ece_10bin": ece}


def audit_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_records: list[dict[str, Any]] = []
    for row in rows:
        condition = row["condition"]
        if condition not in ANOMALOUS_CONDITIONS:
            continue
        truth = ANOMALOUS_CONDITIONS[condition]
        for event in row.get("event_decisions", []):
            decision = event.get("decision")
            event_records.append({
                "policy": row["policy"],
                "condition": condition,
                "budget": row["budget"],
                "seed": row["seed"],
                "truth": truth,
                "prediction": DECISION_TO_SOURCE.get(decision),
                "decision": decision,
                "allowed": bool(event.get("allowed")),
                "visible_residual": event.get("visible_residual"),
                "posterior": event.get("source_posterior"),
                "verification_cost": event.get("verification_cost"),
                "action_trace": event.get("action_trace", []),
            })

    metrics: dict[str, Any] = {}
    for policy in sorted({row["policy"] for row in rows}):
        subset = [record for record in event_records if record["policy"] == policy]
        y_true = [record["truth"] for record in subset]
        y_pred = [record["prediction"] for record in subset]
        posterior_rows = [
            (record["truth"], record["posterior"])
            for record in subset
            if isinstance(record["posterior"], dict)
        ]
        metrics[policy] = {
            "source_attribution": macro_f1(y_true, y_pred),
            "posterior_calibration": calibration(posterior_rows),
            "mean_event_verification_cost": mean(
                record["verification_cost"] for record in subset if finite(record["verification_cost"])
            ),
            "terminal_action_accuracy": mean(
                float(record["prediction"] == record["truth"])
                for record in subset
            ),
            "event_count": len(subset),
        }

    residuals: dict[str, dict[str, Any]] = {}
    for condition in ANOMALOUS_CONDITIONS:
        values = [
            float(record["visible_residual"])
            for record in event_records
            if record["condition"] == condition and finite(record["visible_residual"])
        ]
        residuals[condition] = {
            "n": len(values),
            "mean": mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {"by_policy": metrics, "matched_visible_residuals": residuals, "event_records": event_records}


def aggregate_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["condition"], int(row["budget"]))].append(row)
    output: dict[str, Any] = {}
    for (policy, condition, budget), group in sorted(grouped.items()):
        key = f"{policy}|{condition}|budget{budget}"
        true_rows = [row for row in group if condition in {"true_mechanism", "mechanism_reversal"}]
        correct_update_times = []
        for row in true_rows:
            if condition == "true_mechanism" and row.get("final_state_correct") and row.get("update_round") is not None:
                correct_update_times.append(row["update_round"])
            elif condition == "mechanism_reversal" and row.get("final_state_correct") and row.get("backtrack_rounds"):
                correct_update_times.append(row["backtrack_rounds"][-1])
        output[key] = {
            "n": len(group),
            "false_principle_revision_rate": mean(float(row["false_expansion"]) for row in group),
            "true_mechanism_miss_rate": mean(float(row["missed_true_expansion"]) for row in group),
            "final_state_accuracy": mean(float(row["final_state_correct"]) for row in group),
            "mean_verification_cost": mean(row["verification_cost"] for row in group),
            "mean_total_calls": mean(row["total_calls"] for row in group),
            "time_to_correct_revision": mean(correct_update_times),
            "mean_post_update_regret": mean(row["post_update_regret"] for row in group),
            "backtrack_rate": mean(float(row["backtrack_count"] > 0) for row in group),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    rows = load(input_path / "trial_results.json")
    result = {
        "schema_version": "active_source_metrics.v1",
        "paper_eligible": False,
        "input": str(input_path),
        "input_sha256": hashlib.sha256((input_path / "trial_results.json").read_bytes()).hexdigest(),
        "protocol_scope": "all recorded seeds, conditions, policies, and budgets; no outcome-based selection",
        "source_classes": list(SOURCE_CLASSES),
        "trial_metrics": aggregate_trials(rows),
        **audit_events(rows),
        "claim_boundary": {
            "supported": [
                "matched visible residuals are present across the five anomalous source conditions",
                "source attribution and action behavior can be scored under the controlled hidden-law evaluator",
                "false revision, true-mechanism miss, cost, backtracking, and post-update regret are reported over all rows",
            ],
            "not_supported": [
                "real laboratory or physical chemistry accuracy",
                "external benchmark performance",
                "universal SOTA",
                "paper inclusion without a separate evidence decision",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    lines = [
        "# Exploratory source-attribution audit",
        "",
        "This artifact is paper-ineligible and uses all rows from the deterministic zero-API benchmark.",
        "",
        "## Source attribution and calibration",
        "",
        "| Policy | Macro-F1 | Coverage | Brier | ECE | Terminal action accuracy | Mean event cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, values in result["by_policy"].items():
        source = values["source_attribution"]
        cal = values["posterior_calibration"]
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {policy} | {fmt(source['macro_f1'])} | {fmt(source['coverage'])} | "
            f"{fmt(cal['brier'])} | {fmt(cal['ece_10bin'])} | "
            f"{fmt(values['terminal_action_accuracy'])} | {fmt(values['mean_event_verification_cost'])} |"
        )
    lines += ["", "## Matched visible residuals", "", "| Condition | n | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|---:|"]
    for condition, values in result["matched_visible_residuals"].items():
        lines.append(
            f"| {condition} | {values['n']} | {values['mean']:.6f} | {values['std']:.6f} | "
            f"{values['min']:.6f} | {values['max']:.6f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "events": len(result["event_records"]), "policies": list(result["by_policy"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
