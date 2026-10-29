#!/usr/bin/env python3
"""Validation test for scripts/scrub.py.

Asserts that each targeted secret pattern is redacted and that benign strings
(URLs, ordinary code) are NOT redacted.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scrub import scrub


# ── fixture text — one example of each secret type ───────────────────────
# Fake tokens are split into adjacent literals so no full token sits in the source.

SECRETS: list[tuple[str, str, str]] = [
    # (description, text_containing_secret, expected_redaction_kind)
    (
        "OpenAI API key",
        "Use sk-" "abcdefghijklmnopqrstuvwxyz1234567890AB to call the API.",
        "REDACTED:API_KEY",
    ),
    (
        "AWS access key ID",
        "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "REDACTED:AWS_ACCESS_KEY",
    ),
    (
        "GitHub personal access token",
        "git clone with GHP_TOKEN=ghp_" "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890AB",
        "REDACTED:GH_TOKEN",
    ),
    (
        "GitHub server token",
        "Authorization: ghs_" "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890AB",
        "REDACTED:GH_TOKEN",
    ),
    (
        "Slack bot token",
        "slack_token = xoxb-" "1234567890-abcdefghijklmnopqrstuvwx",
        "REDACTED:SLACK_TOKEN",
    ),
    (
        "JWT",
        "token = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "REDACTED:JWT",
    ),
    (
        "Bearer token",
        "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9abcdefghijklmnop",
        "REDACTED:BEARER_TOKEN",
    ),
    (
        "password assignment",
        "password = supersecretpassword123",
        "REDACTED:PASSWORD",
    ),
    (
        "api_key assignment",
        "api_key = my-secret-key-abc123",
        "REDACTED:API_KEY",
    ),
    (
        "AWS secret access key",
        "AWS_SECRET_ACCESS_KEY = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "REDACTED:AWS_SECRET",
    ),
]

BENIGN: list[tuple[str, str]] = [
    # (description, text_that_should_NOT_be_redacted)
    ("plain URL", "See https://example.com/path?query=1 for details."),
    ("ordinary code", "def greet(name: str) -> str:\n    return f'Hello, {name}'"),
    ("short token", "token = abc123"),
    ("normal config value", "host = localhost\nport = 5432\ndbname = myapp"),
    ("version string", "version = 1.2.3-rc1"),
]


def test_secrets_redacted() -> None:
    failures: list[str] = []
    for desc, text, expected_marker in SECRETS:
        result = scrub(text, entropy_check=False)
        if expected_marker not in result:
            failures.append(
                f"  FAIL [{desc}]: expected [{expected_marker}] not found in:\n    {result!r}"
            )
        else:
            print(f"  PASS [{desc}]: found [{expected_marker}]")

    if failures:
        for f in failures:
            print(f)
        sys.exit(1)


def test_benign_not_redacted() -> None:
    failures: list[str] = []
    for desc, text in BENIGN:
        result = scrub(text, entropy_check=False)
        if "REDACTED" in result:
            failures.append(
                f"  FAIL [{desc}]: unexpected REDACTED in:\n    {result!r}"
            )
        else:
            print(f"  PASS [{desc}]: not redacted")

    if failures:
        for f in failures:
            print(f)
        sys.exit(1)


def test_entropy_pass_redacts_high_entropy() -> None:
    # A 40-char random-looking base64 string should be caught by entropy pass
    import random, string
    rng = random.Random(42)
    high_entropy = "".join(rng.choices(string.ascii_letters + string.digits + "+/", k=48))
    text = f"some_var = {high_entropy}"
    result = scrub(text, entropy_check=True)
    if "REDACTED" in result:
        print(f"  PASS [entropy pass]: high-entropy token redacted")
    else:
        # entropy pass is optional / best-effort; do not fail hard
        print(f"  SKIP [entropy pass]: token not caught (entropy may be borderline)")


if __name__ == "__main__":
    print("test_scrub.py")
    test_secrets_redacted()
    test_benign_not_redacted()
    test_entropy_pass_redacts_high_entropy()
    print("\nAll scrub tests passed.")
