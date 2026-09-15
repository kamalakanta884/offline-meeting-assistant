"""
End-to-end CLI pipeline runner for development/demo purposes.

Usage:
    python -m src.app.main --audio demo/sample_meeting.wav

This runs: ASR -> diarization (with persistent speaker memory) -> live PII
redaction -> summarization, and prints a redacted transcript plus a meeting
summary. This is a CLI harness for development; the actual submission app
should wrap this pipeline in a desktop UI showing the live transcript,
speaker labels, and redaction highlighting as described in the README.
"""

import argparse
import json

import dataclasses

from src.asr.transcribe import get_default_backend
from src.diarization.diarize import SpeakerStore, diarize_segments
from src.redaction.redact import redact_text
from src.summarization.summarize import get_default_summarizer


def run_pipeline(audio_path: str) -> dict:
    asr = get_default_backend()
    segments = asr.transcribe(audio_path)

    store = SpeakerStore.load()
    turns = diarize_segments(segments, audio_path=audio_path, store=store)

    # Redact each turn's text in place before it goes anywhere else in the
    # pipeline (display AND summarization) — sensitive spans must never
    # reach the summarizer, or they could leak into the meeting summary.
    redacted_turns_objs = []
    for turn in turns:
        redacted_turn = dataclasses.replace(turn, text=redact_text(turn.text))
        redacted_turns_objs.append(redacted_turn)

    redacted_turns = [
        {
            "speaker": t.speaker_label,
            "start": t.start,
            "end": t.end,
            "text": t.text,
        }
        for t in redacted_turns_objs
    ]

    summarizer = get_default_summarizer()
    summary = summarizer(redacted_turns_objs)

    return {
        "transcript": redacted_turns,
        "summary": summary.summary,
        "action_items": summary.action_items,
        "decisions": summary.decisions,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline meeting assistant pipeline")
    parser.add_argument("--audio", required=True, help="Path to a .wav meeting audio file")
    parser.add_argument("--out", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    result = run_pipeline(args.audio)

    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
