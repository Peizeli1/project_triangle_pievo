"""Parse PiEvo's cumulative token snapshots without double counting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _count_leaf(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        return (0, 0, 0)
    prompt = value.get("prompt", value.get("prompt_tokens", value.get("input_tokens")))
    completion = value.get(
        "completion", value.get("completion_tokens", value.get("output_tokens"))
    )
    if _number(prompt) and _number(completion):
        prompt_value = int(prompt)
        completion_value = int(completion)
        return prompt_value, completion_value, prompt_value + completion_value
    total = value.get("total", value.get("total_tokens"))
    if _number(total):
        return 0, 0, int(total)
    return (0, 0, 0)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def snapshot_usage(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        leaf = _count_leaf(value)
        if leaf != (0, 0, 0):
            return {"prompt": leaf[0], "completion": leaf[1], "total": leaf[2]}
        prompt = completion = total = 0
        for key, item in value.items():
            if key in {"round", "timestamp", "usage_total"}:
                continue
            child = snapshot_usage(item)
            prompt += child["prompt"]
            completion += child["completion"]
            total += child["total"]
        return {"prompt": prompt, "completion": completion, "total": total}
    if isinstance(value, list):
        prompt = completion = total = 0
        for item in value:
            child = snapshot_usage(item)
            prompt += child["prompt"]
            completion += child["completion"]
            total += child["total"]
        return {"prompt": prompt, "completion": completion, "total": total}
    return {"prompt": 0, "completion": 0, "total": 0}


def last_cumulative_usage(value: Any) -> dict[str, int | None]:
    """Return the last nonempty cumulative snapshot, never the sum of rounds."""
    if isinstance(value, list):
        for index in range(len(value) - 1, -1, -1):
            usage = snapshot_usage(value[index].get("usage", value[index]) if isinstance(value[index], dict) else value[index])
            if usage["total"] > 0:
                return {**usage, "snapshot_index": index}
        return {"prompt": 0, "completion": 0, "total": 0, "snapshot_index": None}
    usage = snapshot_usage(value.get("usage", value) if isinstance(value, dict) else value)
    return {**usage, "snapshot_index": 0 if usage["total"] > 0 else None}


def read_cumulative_usage(path: Path) -> dict[str, int | None]:
    if not path.is_file():
        return {"prompt": 0, "completion": 0, "total": 0, "snapshot_index": None}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"prompt": 0, "completion": 0, "total": 0, "snapshot_index": None}
    return last_cumulative_usage(value)
