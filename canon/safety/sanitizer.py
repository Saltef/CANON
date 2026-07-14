from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from canon.safety.scanner import (
    EXFILTRATION_PATTERNS,
    HIDDEN_TEXT_PATTERNS,
    INSTRUCTION_PATTERNS,
    ROLE_CONFUSION_PATTERNS,
    TOOL_CONTROL_PATTERNS,
    scan_text,
)


@dataclass(frozen=True)
class SanitizationResult:
    original_text: str
    sanitized_text: str
    changed: bool
    decision: str
    removed_categories: list[str]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


PATTERN_GROUPS = {
    "instruction_override": INSTRUCTION_PATTERNS,
    "role_confusion": ROLE_CONFUSION_PATTERNS,
    "exfiltration": EXFILTRATION_PATTERNS,
    "tool_control": TOOL_CONTROL_PATTERNS,
    "hidden_text": HIDDEN_TEXT_PATTERNS,
}


def sanitize_retrieved_text(text: str, provenance: str = "academic") -> SanitizationResult:
    scan = scan_text(text, provenance=provenance)
    if scan.decision in {"allow", "block", "quarantine"}:
        return SanitizationResult(
            original_text=text,
            sanitized_text=text,
            changed=False,
            decision=scan.decision,
            removed_categories=[],
            note="no_sanitization_applied",
        )
    sanitized = unicodedata.normalize("NFKC", text)
    removed: list[str] = []
    for category, patterns in PATTERN_GROUPS.items():
        before = sanitized
        for pattern in patterns:
            sanitized = re.sub(pattern, "[removed unsafe instruction]", sanitized, flags=re.IGNORECASE | re.DOTALL)
        if sanitized != before:
            removed.append(category)
    sanitized = strip_control_format_chars(sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return SanitizationResult(
        original_text=text,
        sanitized_text=sanitized,
        changed=sanitized != text,
        decision="sanitize",
        removed_categories=removed or scan.categories,
        note="sanitized_text_is_allowed_only_as_evidence_data",
    )


def strip_control_format_chars(text: str) -> str:
    return "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"Cf", "Cc"} or char in {"\n", "\r", "\t"}
    )


def retrieved_instruction_exception_policy(
    source_authorized: bool,
    bounded_workflow: bool,
    external_action_requested: bool,
) -> dict:
    allowed = source_authorized and bounded_workflow and not external_action_requested
    return {
        "exception_allowed": allowed,
        "requirements": [
            "source_trusted_or_user_authorized",
            "workflow_scope_bounded",
            "no_direct_external_actions_or_secret_requests",
            "instruction_use_logged",
        ],
        "decision": "allow_bounded_exception" if allowed else "deny_exception",
    }
