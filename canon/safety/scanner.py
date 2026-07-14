from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


INSTRUCTION_PATTERNS = [
    r"\bignore (all )?(previous|above|earlier) (instructions|directions|rules)\b",
    r"\bdisregard (the )?(previous|above|earlier) (instructions|directions|rules)\b",
    r"\boverride (the )?(system|developer|user) (prompt|message|instructions)\b",
    r"\breveal (the )?(system prompt|developer message|hidden instructions)\b",
    r"\bdo not (answer|follow|obey|mention|cite)\b",
    r"\byou are now\b",
    r"\bact as\b",
]

ROLE_CONFUSION_PATTERNS = [
    r"(^|\n)\s*(system|developer|assistant|user)\s*:",
    r"\b(system|developer|assistant|user)\s*:",
    r"<\s*/?\s*(system|developer|assistant|user)\s*>",
    r"\[/?(system|developer|assistant|user)\]",
]

EXFILTRATION_PATTERNS = [
    r"\b(api[_ -]?key|token|password|credential|secret|private key)\b",
    r"\b(send|email|post|upload|exfiltrate|leak)\b.*\b(secret|token|credential|password)\b",
    r"\bchain[- ]of[- ]thought\b",
]

TOOL_CONTROL_PATTERNS = [
    r"\b(run|execute|call|invoke)\b.*\b(shell|command|tool|function|python|powershell)\b",
    r"\bclick\b.*\blink\b",
    r"\bbrowse to\b",
    r"\bdownload\b.*\bfile\b",
]

HIDDEN_TEXT_PATTERNS = [
    r"display\s*:\s*none",
    r"visibility\s*:\s*hidden",
    r"font-size\s*:\s*0",
    r"color\s*:\s*(white|#fff|#ffffff)",
    r"<!--.*?-->",
]


@dataclass(frozen=True)
class SafetyScan:
    risk_score: float
    decision: str
    categories: list[str]
    matched_patterns: list[str]

    def as_dict(self) -> dict[str, float | str | list[str]]:
        return {
            "risk_score": self.risk_score,
            "decision": self.decision,
            "categories": self.categories,
            "matched_patterns": self.matched_patterns,
        }


def scan_text(text: str, provenance: str = "academic") -> SafetyScan:
    lower = normalize_for_scan(text)
    categories: list[str] = []
    matched: list[str] = []

    for category, patterns in [
        ("instruction_override", INSTRUCTION_PATTERNS),
        ("role_confusion", ROLE_CONFUSION_PATTERNS),
        ("exfiltration", EXFILTRATION_PATTERNS),
        ("tool_control", TOOL_CONTROL_PATTERNS),
        ("hidden_text", HIDDEN_TEXT_PATTERNS),
    ]:
        hits = matching_patterns(lower, patterns)
        if hits:
            categories.append(category)
            matched.extend(hits)

    if has_unicode_anomaly(text):
        categories.append("unicode_anomaly")
        matched.append("unicode_control_or_format")

    if has_repetition_attack_shape(lower):
        categories.append("retrieval_trigger_shape")
        matched.append("excessive_repetition")

    risk = risk_score(categories, provenance)
    return SafetyScan(
        risk_score=risk,
        decision=safety_decision(risk, categories),
        categories=categories,
        matched_patterns=matched,
    )


def normalize_for_scan(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def matching_patterns(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.DOTALL)]


def has_unicode_anomaly(text: str) -> bool:
    suspicious = 0
    for char in text:
        category = unicodedata.category(char)
        if category in {"Cf", "Cc"} and char not in {"\n", "\r", "\t"}:
            suspicious += 1
    return suspicious >= 2


def has_repetition_attack_shape(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9_:-]{4,}", text)
    if len(tokens) < 24:
        return False
    most_common = max(tokens.count(token) for token in set(tokens))
    return most_common / len(tokens) >= 0.25


def risk_score(categories: list[str], provenance: str) -> float:
    weights = {
        "instruction_override": 0.45,
        "role_confusion": 0.3,
        "exfiltration": 0.35,
        "tool_control": 0.3,
        "hidden_text": 0.25,
        "unicode_anomaly": 0.15,
        "retrieval_trigger_shape": 0.2,
    }
    score = sum(weights.get(category, 0.0) for category in categories)
    if provenance in {"web", "user_upload", "comment", "forum_comment", "social_media", "unknown"} and categories:
        score += 0.1
    return round(min(1.0, score), 6)


def safety_decision(risk: float, categories: list[str]) -> str:
    if "exfiltration" in categories and ("instruction_override" in categories or "tool_control" in categories):
        return "block"
    if risk >= 0.55:
        return "quarantine"
    if risk >= 0.3:
        return "sanitize"
    return "allow"
