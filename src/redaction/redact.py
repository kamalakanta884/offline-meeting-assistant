"""
Live PII / sensitive-information redaction.

Dev fallback: regex-based detection of common structured identifiers
(account-number-shaped digit sequences, phone numbers, ID-number-shaped
sequences). This runs with zero extra dependencies so the pipeline works
out of the box.

Deployment target: augment with a quantized on-device NER model (converted
from an open-source model) to also catch unstructured sensitive mentions
(names in a patient/case context, etc). See docs/architecture.md#redaction.
"""

import re
from dataclasses import dataclass
from typing import List

# Structured-identifier patterns. These are intentionally conservative
# starting points — tune thresholds/patterns based on real meeting data
# during the hackathon build.
PATTERNS = {
    "id_number_12_digit": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),  # e.g. Aadhaar-shaped
    "account_number": re.compile(r"\b\d{9,18}\b"),
    "phone_number": re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?\d{10}\b"),
}


@dataclass
class RedactionSpan:
    start: int
    end: int
    category: str
    original_text: str


def find_sensitive_spans(text: str) -> List[RedactionSpan]:
    spans: List[RedactionSpan] = []
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append(
                RedactionSpan(
                    start=match.start(),
                    end=match.end(),
                    category=category,
                    original_text=match.group(),
                )
            )
    return spans


def _merge_overlapping_spans(spans: List[RedactionSpan]) -> List[RedactionSpan]:
    """
    Multiple patterns can match overlapping regions (e.g. a 12-digit ID
    number also matches the looser account-number pattern). Keep only the
    widest span for each overlapping cluster so redaction markers don't get
    interleaved/corrupted.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    merged: List[RedactionSpan] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start < last.end:  # overlaps with the last kept span
            if (span.end - span.start) > (last.end - last.start):
                merged[-1] = span  # prefer the wider match
        else:
            merged.append(span)
    return merged


def redact_text(text: str) -> str:
    """Returns the text with sensitive spans replaced by a redaction marker."""
    spans = _merge_overlapping_spans(find_sensitive_spans(text))
    spans.sort(key=lambda s: s.start, reverse=True)
    redacted = text
    for span in spans:
        marker = f"[REDACTED:{span.category}]"
        redacted = redacted[: span.start] + marker + redacted[span.end :]
    return redacted


# TODO: NER-based redaction
# Add a quantized on-device NER model here to catch unstructured sensitive
# mentions (e.g. "the patient, Mr. Sharma, ...") that regex alone will miss.
# Combine its output spans with `find_sensitive_spans` above before calling
# `redact_text`.
