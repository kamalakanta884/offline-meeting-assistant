# Demo

Before submission, add here:

1. `sample_meeting.wav` — a short (1–2 min) sample recording, ideally with 2+ speakers and one Hindi-English code-switched sentence, to show the ASR handling it correctly
2. A screen recording / GIF of the live app showing:
   - Live transcript appearing as people speak
   - A sensitive number (e.g. a fake account number) getting redacted live
   - A second meeting where a returning speaker is auto-labeled by name
   - Wi-Fi/network turned OFF during the whole demo, to prove it's fully offline
3. `demo_script.md` — a short script of what to say while presenting, timed to ~3 minutes

## Running the CLI demo

```bash
python -m src.app.main --audio demo/sample_meeting.wav --out demo/output.json
```

This prints the redacted transcript with speaker labels, plus a summary, action items, and decisions extracted from the meeting.
