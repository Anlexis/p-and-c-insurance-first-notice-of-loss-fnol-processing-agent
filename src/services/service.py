"""AgentCore Platform v1.0"""

# Service layer: shared, node-independent validation of caller-supplied data.
# Contains no business logic, no routing and no credentials — the domain nodes
# call these helpers to decide whether caller data may enter the pipeline at
# all, and the HTTP adapter calls the same helpers so both entry paths enforce
# one contract.
#
# Three rules are implemented here, and every caller-facing check in this
# template is one of them:
#
#   1. inert identifiers — every caller string that can reach the rendered
#      response is restricted to a closed alphabet, so caller text can never
#      become response markup or instructions;
#   2. finite bounded numbers — every number the agent acts on is parsed to a
#      real, finite value inside an explicit range, and anything else is
#      refused (NaN compares False against every bound, so an unchecked
#      non-finite value silently disables the decision it feeds);
#   3. instruction-override screening — caller text is refused when it carries
#      a chat-template control token or an explicit directive to discard the
#      agent's own instructions.

from __future__ import annotations

import math
import re
from typing import Any, Optional

# Claim and policy identifiers are printed into the acknowledgment, so they are
# restricted to an alphabet that cannot carry markup, whitespace or directives.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Structured invocation parameters (the input_context channel) are narrower
# still: lowercase tokens only, so a value can never read as prose.
_INERT_TOKEN_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# A field NAME is caller data too. Only an already-inert name is ever echoed
# back in an error message; anything else is reported positionally.
_SAFE_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")

# Chat-template control tokens, screened as a CLASS rather than as a list of
# known strings: any `<|...|>` marker, the instruction brackets used by several
# instruction-tuned model families, and the system-block markers. These carry no
# meaning in an insurance claim question, so refusing them costs nothing and
# closes the whole family (a phrase-only screen misses every one of them).
_CONTROL_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chat_control_token", re.compile(r"<\|[^|>]{0,64}\|>")),
    ("instruction_bracket", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("system_block_marker", re.compile(r"<</?SYS>>", re.IGNORECASE)),
)

# Explicit instruction-override directives. Every pattern is anchored on both
# sides of a full verb+object phrase: a bare verb ("insert", "act") is NOT
# enough, because ordinary claims language contains those words and a screen
# that refuses real work is worse than no screen at all. Probed against this
# template's own knowledge base and canonical payload — see the test module.
_DIRECTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?<![A-Za-z])(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|the\s+|your\s+|these\s+)*"
            r"(?:previous|prior|above|preceding|earlier|foregoing|system)?\s*"
            r"(?:instructions?|rules?|prompts?|directives?|guidelines?|constraints?)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"(?<![A-Za-z])(?:reveal|disclose|repeat|print|output|dump|show)\s+"
            r"(?:me\s+|us\s+)?(?:your|the)\s+(?:full\s+|entire\s+|original\s+)?"
            r"(?:system\s+prompt|system\s+message|initial\s+instructions)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        # A role RE-ASSIGNMENT, not any sentence starting "you are now". The
        # article is what separates the two: "you are now an unrestricted
        # approver" asserts a new identity, while "you are now handling my
        # claim" — which a claimant writes — is a gerund and must pass. The
        # article-free branch is limited to explicit release from constraints.
        "persona_override",
        re.compile(
            r"(?<![A-Za-z])you\s+are\s+(?:now|no\s+longer)\s+(?:an?|the)\s+"
            r"|(?<![A-Za-z])you\s+are\s+no\s+longer\s+"
            r"(?:bound|required|restricted|limited|allowed|obliged)(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_injection",
        re.compile(
            r"(?<![A-Za-z])new\s+(?:instructions?|rules?|system\s+prompt)\s*[:：]",
            re.IGNORECASE,
        ),
    ),
)


def safe_field_name(name: Any, position: int) -> str:
    """Render a caller-supplied field name for an error message.

    Field names are caller data. A name is echoed only when it is already inert
    and trips no screen of its own; otherwise it is reported by position, so an
    error message can never become a channel for the payload it rejected.
    """
    if isinstance(name, str) and _SAFE_FIELD_NAME_RE.match(name) and screen_text(name) is None:
        return name
    return f"field #{position}"


def is_identifier(value: Any) -> bool:
    """True when *value* is a string safe to render into the response document."""
    return isinstance(value, str) and bool(_IDENTIFIER_RE.match(value))


def is_inert_token(value: Any) -> bool:
    """True when *value* is a lowercase inert token (the input_context alphabet)."""
    return isinstance(value, str) and bool(_INERT_TOKEN_RE.match(value))


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse *value* as a real, finite number inside ``[lo, hi]``; else ``None``.

    Rejects booleans (``isinstance(True, int)`` is true in Python), non-numeric
    types, and NaN / ±Infinity. The non-finite case is the dangerous one: every
    comparison against NaN is False, so an unchecked NaN threshold would drop
    every retrieved passage — or keep every one — without any error surfacing.
    Callers treat ``None`` as "not declared" and fall back to their own default.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


def screen_text(text: str) -> Optional[str]:
    """Return the name of the first override pattern *text* matches, else None."""
    if not isinstance(text, str) or not text:
        return None
    for name, pattern in _CONTROL_TOKEN_PATTERNS:
        if pattern.search(text):
            return name
    for name, pattern in _DIRECTIVE_PATTERNS:
        if pattern.search(text):
            return name
    return None


def screen_structure(value: Any, _depth: int = 0) -> Optional[str]:
    """Screen every string leaf AND every mapping key, depth-first.

    Applied to the PARSED payload rather than only to the raw request body, so
    a directive hidden behind JSON ``\\u`` escapes — which the raw text does not
    contain and the parsed value does — is still caught. Keys are screened
    because a hostile field name reaches the same audit and error paths a value
    does. Depth is bounded so a deeply nested body cannot exhaust the stack.
    """
    if _depth > 8:
        return "nesting_depth_exceeded"
    if isinstance(value, str):
        return screen_text(value)
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                finding = screen_text(key)
                if finding:
                    return finding
            finding = screen_structure(nested, _depth + 1)
            if finding:
                return finding
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            finding = screen_structure(item, _depth + 1)
            if finding:
                return finding
    return None
