#!/usr/bin/env python3
"""Small CLI exposing the controlled source-aware gate to external harnesses."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.source_aware_gate import SourceAwareGate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=float, required=True)
    parser.add_argument("--repeat", type=float, required=True)
    parser.add_argument("--independent", type=float, required=True)
    parser.add_argument("--prediction", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    decision = SourceAwareGate(tolerance=0.25).decide(
        primary=args.primary,
        repeat=args.repeat,
        independent=args.independent,
        surprisal=abs(args.primary - args.prediction),
        threshold=args.threshold,
    )
    print(json.dumps(asdict(decision), sort_keys=True))


if __name__ == "__main__":
    main()
