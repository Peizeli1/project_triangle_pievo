#!/usr/bin/env python3
"""Write a no-network method-runtime attestation before a matched run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_runtime_contract import TASKS, METHODS, validate_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--recovery",
        required=False,
        type=Path,
        help="Optional outcome-blind recovery implementation to hash into the attestation.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--development",
        action="store_true",
        help="Check wiring/features without granting formal-run eligibility.",
    )
    args = parser.parse_args()

    result = validate_runtime(
        method=args.method,
        task=args.task,
        seed=args.seed,
        repository=args.repository,
        input_paths={
            "task_config": args.task_config,
            "model_config": args.model_config,
            "protocol": args.protocol,
            **({"recovery": args.recovery} if args.recovery is not None else {}),
        },
        require_formal_ready=not args.development,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "errors": result["errors"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
