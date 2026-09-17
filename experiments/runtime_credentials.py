"""Ephemeral credential lookup for local experiment launchers.

The experiment artifacts must record the endpoint and model, never the API
credential.  On this workstation the key is kept in macOS Keychain under the
``zjuapi.com`` service.  This module intentionally returns the value only to
the in-memory client process and never writes it to disk.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


DEFAULT_KEYCHAIN_SERVICE = "zjuapi.com"
DEFAULT_KEYCHAIN_ACCOUNT = "ai4ai"


def _keychain_value(*, service: str, account: str) -> str:
    command = ["security", "find-generic-password", "-a", account, "-s", service, "-w"]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "API key was not found in the configured keychain entry; "
            "set ZJU_TASK_API_KEY_FILE or configure the local Keychain item"
        ) from exc
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("the configured Keychain API-key entry is empty")
    return value


def read_api_key(
    path: Path | None = None,
    *,
    keychain_service: str | None = None,
    keychain_account: str | None = None,
) -> str:
    """Read a key from an explicit file, environment, or macOS Keychain.

    File and environment overrides preserve compatibility with existing
    runners.  The normal path is Keychain lookup, so no plaintext key file is
    needed for a local run.
    """

    if path is not None and path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
        raise RuntimeError(f"API key file is empty: {path}")

    environment_value = os.getenv("ZJU_TASK_API_KEY", "").strip()
    if environment_value:
        return environment_value

    return _keychain_value(
        service=keychain_service or os.getenv(
            "ZJU_TASK_KEYCHAIN_SERVICE", DEFAULT_KEYCHAIN_SERVICE
        ),
        account=keychain_account or os.getenv(
            "ZJU_TASK_KEYCHAIN_ACCOUNT", DEFAULT_KEYCHAIN_ACCOUNT
        ),
    )
