#!/usr/bin/env python3
"""Stateful evidence-admission CLI for Harbor/AARRI experiments.

The tool is task-agnostic. It stores claims, evidence provenance, channel trust,
and principle lifecycle state in a local JSON file. It never reads benchmark
verifiers, hidden tests, or reference solutions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

DEFAULT_STATE = "/tmp/triage-evidence-state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path() -> Path:
    return Path(os.environ.get("TRIAGE_STATE", DEFAULT_STATE))


def empty_state() -> dict:
    return {
        "version": 1,
        "created_at": now(),
        "channels": {},
        "claims": {},
        "events": [],
    }


def load_state() -> dict:
    path = state_path()
    if not path.exists():
        return empty_state()
    return json.loads(path.read_text())


def save_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def emit(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def get_channel(state: dict, name: str) -> dict:
    return state["channels"].setdefault(name, {"alpha": 1.0, "beta": 1.0})


def trust(ch: dict) -> float:
    return ch["alpha"] / (ch["alpha"] + ch["beta"])


def cmd_init(args: argparse.Namespace) -> None:
    state = empty_state()
    save_state(state)
    emit({"status": "initialized", "state": str(state_path())})


def cmd_claim(args: argparse.Namespace) -> None:
    state = load_state()
    if args.id in state["claims"] and not args.replace:
        raise SystemExit(f"claim {args.id!r} already exists; pass --replace to overwrite")
    state["claims"][args.id] = {
        "statement": args.statement,
        "parent": args.parent,
        "status": "proposed",
        "created_at": now(),
        "evidence": [],
        "decision_history": [],
    }
    state["events"].append({"time": now(), "type": "claim", "claim_id": args.id})
    save_state(state)
    emit({"claim_id": args.id, "status": "proposed", "statement": args.statement})


def cmd_evidence(args: argparse.Namespace) -> None:
    state = load_state()
    claim = state["claims"].get(args.claim_id)
    if claim is None:
        raise SystemExit(f"unknown claim {args.claim_id!r}")
    if any(e["id"] == args.id for e in claim["evidence"]):
        raise SystemExit(f"evidence {args.id!r} already exists for claim {args.claim_id!r}")
    ch = get_channel(state, args.source)
    rec = {
        "id": args.id,
        "source": args.source,
        "stance": args.stance,
        "independent": bool(args.independent),
        "heldout": bool(args.heldout),
        "reliability": args.reliability,
        "note": args.note,
        "channel_trust_at_record": trust(ch),
        "time": now(),
    }
    claim["evidence"].append(rec)
    if claim["status"] == "proposed" and args.stance == "support":
        claim["status"] = "provisional"
    state["events"].append({
        "time": now(), "type": "evidence", "claim_id": args.claim_id,
        "evidence_id": args.id, "source": args.source,
    })
    save_state(state)
    emit({"recorded": rec, "claim_status": claim["status"]})


def cmd_channel(args: argparse.Namespace) -> None:
    state = load_state()
    ch = get_channel(state, args.source)
    if args.outcome == "agree":
        ch["alpha"] += args.weight
    else:
        ch["beta"] += args.weight
    state["events"].append({
        "time": now(), "type": "channel_update", "source": args.source,
        "outcome": args.outcome, "weight": args.weight,
    })
    save_state(state)
    emit({"source": args.source, "alpha": ch["alpha"], "beta": ch["beta"], "trust": trust(ch)})


def evaluate_claim(state: dict, claim_id: str) -> dict:
    claim = state["claims"].get(claim_id)
    if claim is None:
        raise SystemExit(f"unknown claim {claim_id!r}")
    ev = claim["evidence"]
    heldout_contra = [e for e in ev if e["stance"] == "contradict" and e["independent"] and e["heldout"]]
    indep_held_support = [e for e in ev if e["stance"] == "support" and e["independent"] and e["heldout"]]
    independent_sources = sorted({e["source"] for e in indep_held_support})
    uncertain = [e for e in ev if e["stance"] == "uncertain"]
    low_trust_sources = sorted({
        e["source"] for e in ev
        if trust(get_channel(state, e["source"])) < 0.5
    })

    if heldout_contra:
        action = "QUARANTINE"
        status = "retracted"
        reason = "independent held-out contradiction"
    elif len(indep_held_support) >= 2:
        action = "ADMIT"
        status = "accepted"
        reason = "at least two records marked independent and held out support the claim"
    elif ev:
        action = "VERIFY"
        status = "provisional" if any(e["stance"] == "support" for e in ev) else claim["status"]
        reason = "evidence exists but independent held-out support is insufficient"
    else:
        action = "VERIFY"
        status = "proposed"
        reason = "no evidence recorded"

    return {
        "claim_id": claim_id,
        "action": action,
        "status": status,
        "reason": reason,
        "support_count": sum(e["stance"] == "support" for e in ev),
        "contradict_count": sum(e["stance"] == "contradict" for e in ev),
        "independent_heldout_support_count": len(indep_held_support),
        "independent_heldout_sources": independent_sources,
        "heldout_contradiction_count": len(heldout_contra),
        "uncertain_count": len(uncertain),
        "low_trust_sources": low_trust_sources,
    }


def retract_descendants(state: dict, parent_id: str) -> list[str]:
    changed = []
    frontier = [parent_id]
    while frontier:
        current = frontier.pop()
        for cid, c in state["claims"].items():
            if c.get("parent") == current and c.get("status") != "retracted":
                c["status"] = "retracted"
                changed.append(cid)
                frontier.append(cid)
    return changed


def cmd_decide(args: argparse.Namespace) -> None:
    state = load_state()
    result = evaluate_claim(state, args.claim_id)
    claim = state["claims"][args.claim_id]
    old_status = claim["status"]
    claim["status"] = result["status"]
    descendants = []
    if result["action"] == "QUARANTINE":
        descendants = retract_descendants(state, args.claim_id)
    record = {"time": now(), **result, "previous_status": old_status, "retracted_descendants": descendants}
    claim["decision_history"].append(record)
    state["events"].append({"time": now(), "type": "decision", "claim_id": args.claim_id, "action": result["action"]})
    save_state(state)
    emit(record)


def cmd_status(args: argparse.Namespace) -> None:
    state = load_state()
    out = {
        "state": str(state_path()),
        "channels": {k: {**v, "trust": trust(v)} for k, v in state["channels"].items()},
        "claims": state["claims"],
    }
    emit(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="triage-evidence", description="Stateful evidence admission for scientific-agent decisions")
    p.add_argument("--state", help="override state file path")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("claim")
    s.add_argument("--id", required=True)
    s.add_argument("--statement", required=True)
    s.add_argument("--parent")
    s.add_argument("--replace", action="store_true")
    s.set_defaults(func=cmd_claim)

    s = sub.add_parser("evidence")
    s.add_argument("--claim-id", required=True)
    s.add_argument("--id", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--stance", choices=["support", "contradict", "uncertain"], required=True)
    s.add_argument("--independent", action="store_true")
    s.add_argument("--heldout", action="store_true")
    s.add_argument("--reliability", choices=["low", "medium", "high"], default="medium")
    s.add_argument("--note", default="")
    s.set_defaults(func=cmd_evidence)

    s = sub.add_parser("channel")
    s.add_argument("--source", required=True)
    s.add_argument("--outcome", choices=["agree", "disagree"], required=True)
    s.add_argument("--weight", type=float, default=1.0)
    s.set_defaults(func=cmd_channel)

    s = sub.add_parser("decide")
    s.add_argument("--claim-id", required=True)
    s.set_defaults(func=cmd_decide)

    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.state:
        os.environ["TRIAGE_STATE"] = args.state
    args.func(args)


if __name__ == "__main__":
    main()
