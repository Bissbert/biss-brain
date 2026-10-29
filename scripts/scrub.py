"""Secret redaction. Public API: scrub(text, entropy_check=True) -> str.

Replaces known secret patterns with [REDACTED:KIND] markers.
Optionally runs a Shannon entropy second pass for unknown high-entropy tokens.
"""
from __future__ import annotations

import math
import re

# ── pattern registry ─────────────────────────────────────────────────────
# Each entry: (compiled pattern, replacement label)
# Order matters: more specific patterns first.

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # PEM private key block (RSA, EC, OPENSSH, PKCS#8, encrypted)
    (
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)",
            re.ASCII | re.DOTALL,
        ),
        "PRIVATE_KEY",
    ),

    # Anthropic API / admin key (sk-ant-api03-..., sk-ant-admin01-...)
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}", re.ASCII), "API_KEY"),

    # OpenAI project / service keys (sk-proj-..., sk-svcacct-...)
    (re.compile(r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}", re.ASCII), "API_KEY"),

    # OpenAI / Anthropic API key
    (re.compile(r"sk-[A-Za-z0-9]{32,}", re.ASCII), "API_KEY"),

    # GitHub fine-grained personal access token
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}", re.ASCII), "GH_TOKEN"),

    # GitHub OAuth / user-to-server / refresh tokens
    (re.compile(r"gh[our]_[A-Za-z0-9]{36}", re.ASCII), "GH_TOKEN"),

    # GitLab personal access token
    (re.compile(r"glpat-[A-Za-z0-9_-]{20,}", re.ASCII), "GITLAB_TOKEN"),

    # Credentials in a URL (scheme://user:password@host): keep scheme and host
    (
        re.compile(r"(?<=://)[^\s/:@]+:[^\s/@]+(?=@)", re.ASCII),
        "URL_CREDENTIALS",
    ),

    # AWS access key ID
    (re.compile(r"AKIA[A-Z0-9]{16}", re.ASCII), "AWS_ACCESS_KEY"),

    # GitHub personal access token
    (re.compile(r"ghp_[A-Za-z0-9]{36}", re.ASCII), "GH_TOKEN"),

    # GitHub app / server-to-server token
    (re.compile(r"ghs_[A-Za-z0-9]{36}", re.ASCII), "GH_TOKEN"),

    # Slack bot token
    (re.compile(r"xoxb-[A-Za-z0-9-]+", re.ASCII), "SLACK_TOKEN"),

    # JWT (three base64url segments separated by dots)
    (
        re.compile(
            r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
            re.ASCII,
        ),
        "JWT",
    ),

    # Bearer token in Authorization header (≥20 chars)
    (
        re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.ASCII | re.IGNORECASE),
        "BEARER_TOKEN",
    ),

    # Inline password assignment  (password = "...", password: something)
    (
        re.compile(
            r"password\s*[=:]\s*\S+",
            re.IGNORECASE | re.ASCII,
        ),
        "PASSWORD",
    ),

    # API key assignment  (api_key = ..., apikey: ...)
    (
        re.compile(
            r"api[_\-]?key\s*[=:]\s*\S+",
            re.IGNORECASE | re.ASCII,
        ),
        "API_KEY",
    ),

    # AWS secret access key: 40-char base64-ish value near the keyword
    (
        re.compile(
            r"(?:AWS_SECRET_ACCESS_KEY|aws_secret_access_key|secret_access_key)"
            r"\s*[=:]\s*[A-Za-z0-9/+]{40}",
            re.ASCII,
        ),
        "AWS_SECRET",
    ),

    # Generic secret/token assignment with a long value
    (
        re.compile(
            r"(?:secret|token)\s*[=:]\s*[A-Za-z0-9._\-/+]{30,}",
            re.IGNORECASE | re.ASCII,
        ),
        "SECRET",
    ),
]

# ── entropy helpers ──────────────────────────────────────────────────────
# Token characters that look like encoded values
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{30,}")


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _entropy_redact(text: str, threshold: float = 4.5, min_len: int = 30) -> str:
    """Replace high-entropy tokens with [REDACTED:HIGH_ENTROPY]."""
    def _replace(m: re.Match[str]) -> str:
        tok = m.group(0)
        if len(tok) >= min_len and _shannon_entropy(tok) >= threshold:
            return "[REDACTED:HIGH_ENTROPY]"
        return tok

    return _TOKEN_RE.sub(_replace, text)


# ── public API ───────────────────────────────────────────────────────────

def scrub(text: str, entropy_check: bool = True) -> str:
    """Redact secrets from *text*.

    Applies regex patterns first, then optionally a Shannon entropy pass.
    Returns the sanitised string.
    """
    for pattern, label in _PATTERNS:
        text = pattern.sub(f"[REDACTED:{label}]", text)

    if entropy_check:
        text = _entropy_redact(text)

    return text
