"""
Meeting summarization: produces a short summary, key decisions, and action
items from the (redacted) transcript.

Dev fallback: a simple extractive placeholder (no LLM dependency) so the
pipeline runs end-to-end without requiring a local model download.

Deployment target: a small quantized local LLM sourced/exported via
Qualcomm AI Hub tooling. See docs/qualcomm_ai_hub_setup.md.
"""

from dataclasses import dataclass, field
from typing import List

from src.diarization.diarize import SpeakerTurn


@dataclass
class MeetingSummary:
    summary: str
    action_items: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)


ACTION_KEYWORDS = ("will ", "action item", "todo", "follow up", "by tomorrow", "assign")
DECISION_KEYWORDS = ("we decided", "agreed", "let's go with", "final call", "decision")


def extractive_summarize(turns: List[SpeakerTurn]) -> MeetingSummary:
    """
    Placeholder summarizer: pulls out lines matching simple keyword
    heuristics for action items / decisions, and uses the first and last
    few lines as a naive "summary". Replace with a real local-LLM call once
    the QNN-backed summarizer is wired up (see TODO below).
    """
    full_text_lines = [t.text for t in turns]

    action_items = [
        line for line in full_text_lines
        if any(k in line.lower() for k in ACTION_KEYWORDS)
    ]
    decisions = [
        line for line in full_text_lines
        if any(k in line.lower() for k in DECISION_KEYWORDS)
    ]

    head = " ".join(full_text_lines[:2])
    tail = " ".join(full_text_lines[-2:]) if len(full_text_lines) > 2 else ""
    summary = (head + " ... " + tail).strip() if tail else head

    return MeetingSummary(summary=summary, action_items=action_items, decisions=decisions)


# TODO: local LLM summarizer
# def llm_summarize(turns: List[SpeakerTurn]) -> MeetingSummary:
#     """
#     Feed the (redacted) transcript into a small quantized local LLM
#     (sourced via Qualcomm AI Hub) and parse its structured output into a
#     MeetingSummary. See docs/qualcomm_ai_hub_setup.md for model setup.
#     """
#     raise NotImplementedError


def get_default_summarizer():
    return extractive_summarize
