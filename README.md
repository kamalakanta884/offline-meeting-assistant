# SahaySar — On-Device Meeting Intelligence

**Fully offline meeting transcription, speaker recognition, live PII redaction, and summarization — running entirely on the NPU of Snapdragon-powered HP PCs. No audio, text, or metadata ever leaves the device.**

Built for the Snapdragon® AI Lab Build & Present Challenge.

---

## Why this exists

Cloud meeting assistants (Otter, Zoom AI Companion, MS Copilot) are unusable in legal, healthcare, finance, and government settings where audio can't leave the building — and they handle Indian code-switched speech (Hindi-English, Tamil-English, etc.) poorly because their pipelines are tuned for monolingual cloud audio. SahaySar is designed specifically to close both gaps, on-device, using Qualcomm AI Hub models optimized for the Snapdragon NPU.

## What makes this different from a generic "AI notetaker"

| Feature | Why it matters |
|---|---|
| **Code-switched ASR** | Tuned/evaluated for Hindi-English and other regional code-switching, not just clean monolingual English |
| **Live on-device PII redaction** | Automatically flags/redacts account numbers, ID numbers, patient/case identifiers as they're spoken — visible in real time in the transcript |
| **Persistent local speaker memory** | Learns a voice-embedding profile per speaker on first meeting; recognizes them by name in later meetings — entirely stored on-device |
| **100% offline** | Every model (ASR, diarization, NER, summarization) runs on-device via the Qualcomm AI Engine / NPU. No network calls at inference time. |
| **Local meeting Q&A (stretch goal)** | Ask questions across past meeting transcripts using on-device RAG — no cloud, no data exposure |

## Architecture

```
  Microphone/System Audio
          │
          ▼
  ┌───────────────┐
  │  ASR (Whisper) │  Qualcomm AI Hub, quantized (w8a16), NPU-accelerated
  └───────┬───────┘
          │ transcript stream
          ▼
  ┌───────────────────┐      ┌────────────────────────┐
  │ Speaker Diarization │ ──▶ │ Persistent Speaker Store │ (local embeddings, on-disk)
  └───────┬───────────┘      └────────────────────────┘
          │
          ▼
  ┌───────────────┐
  │ PII Redaction  │  local NER model, flags/redacts sensitive spans live
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │ Summarization  │  quantized local LLM → action items, decisions, summary
  └───────┬───────┘
          │
          ▼
  Local transcript store (per-meeting, on-disk, never uploaded)
          │
          ▼
  (stretch) Local RAG Q&A over past meetings
```

See [`docs/architecture.md`](docs/architecture.md) for details on each component and [`docs/qualcomm_ai_hub_setup.md`](docs/qualcomm_ai_hub_setup.md) for how models are sourced from Qualcomm AI Hub and modified for this use case.

## Models used / modified

- **ASR:** `whisper-small-quantized` (Qualcomm AI Hub) — evaluated and re-tuned on code-switched Hindi-English samples (see `docs/architecture.md#asr`)
- **Diarization:** open-source speaker-embedding model, converted/quantized for on-device inference (not on AI Hub by default — this satisfies the "add models from open-source platforms" requirement)
- **PII/NER redaction:** lightweight open-source NER model, quantized for local inference
- **Summarization:** small quantized local LLM (see setup doc for current pick)

This project was built for the Challenge Submission Period and modifies/optimizes these models specifically for on-device Snapdragon NPU deployment.

## Status

🚧 Early build — see [Project board / Issues] for current progress. This repo currently contains:
- [x] Repo scaffold + architecture
- [x] CPU-fallback ASR pipeline (for development on non-Snapdragon machines)
- [x] Diarization + persistent speaker memory (MFCC-based, validated with tests — see `docs/architecture.md`)
- [x] Live PII redaction (regex-based structured identifiers; NER upgrade planned)
- [x] Summarization (extractive baseline; local-LLM upgrade planned)
- [ ] Qualcomm AI Hub QNN/NPU integration
- [ ] Neural speaker-embedding model (upgrade from MFCC baseline)
- [ ] Local RAG Q&A (stretch)

## Getting started (development mode, CPU fallback)

```bash
git clone <this-repo>
cd offline-meeting-assistant
pip install -r requirements.txt
python -m src.app.main --audio demo/sample_meeting.wav
```

> Note: the default pipeline in this repo runs on CPU/ONNX Runtime for development on any machine. See `docs/qualcomm_ai_hub_setup.md` for swapping each stage to the Qualcomm AI Hub–optimized, NPU-accelerated model for deployment on a Snapdragon-powered HP PC.

## Demo

See `demo/` for a sample run and (add your demo GIF/video link here before submission).

## License

MIT — see [LICENSE](LICENSE).
