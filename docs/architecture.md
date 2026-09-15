# Architecture

## Overview

SahaySar is a pipeline of four on-device models plus a persistent local store. Each stage is designed to run standalone (for development/testing on any machine) and to be swapped for its Qualcomm AI Hub–optimized, NPU-accelerated counterpart for deployment on Snapdragon-powered HP PCs.

## 1. ASR (Automatic Speech Recognition) {#asr}

- **Base model:** Whisper-Small
- **Source:** `qualcomm/Whisper-Small-Quantized` on Qualcomm AI Hub (w8a16 quantized, encoder split for NPU/DSP execution)
- **Modification for this project:** evaluated on code-switched Hindi-English audio samples; where accuracy drops on code-switched segments, apply targeted fine-tuning / prompt-conditioning using an open-source Hindi-English code-switched speech dataset (e.g. a subset of a public Indian-language ASR corpus)
- **Dev fallback:** `faster-whisper` (CPU) for local development before deploying to the QNN/NPU runtime

## 2. Speaker Diarization

- **Current implementation:** real audio-based speaker discrimination using MFCC (Mel-Frequency Cepstral Coefficient) statistics as a speaker embedding, computed with `librosa`. Segments within a meeting are clustered with agglomerative clustering (cosine distance) to group them by speaker. This requires no downloaded model weights and has been validated to correctly separate distinct voices and recognize repeated speakers within and across meetings (see `tests/test_diarize.py`).
- **Persistent speaker memory:** each cluster's averaged embedding is stored locally (`~/.sahaysar/speakers.json`), keyed by a locally-generated speaker ID with a user-assigned display name. On subsequent meetings, new embeddings are compared against stored ones via cosine similarity (threshold-based match) to auto-label returning speakers by name, once the user has labeled them once. This is the "returning speaker recognition" differentiator demoed live.
- **Planned upgrade path:** swap the MFCC embedding for a neural speaker-embedding model (e.g. an x-vector or ECAPA-TDNN style model), converted to ONNX and quantized for on-device inference — this is the "open-source model added and optimized" component that satisfies the challenge's proposal requirements, and would improve accuracy on real (noisier, more varied) meeting audio beyond what the MFCC baseline provides. The clustering and persistent-memory logic is already embedding-agnostic, so this is a drop-in swap in `src/diarization/embeddings.py`.

## 3. PII / Sensitive-Information Redaction

- **Approach:** run a lightweight local NER model over the live transcript stream; flag spans matching sensitive categories (ID numbers, account numbers, patient/case identifiers, phone numbers) using a combination of the NER model and regex patterns for structured identifiers (e.g. Aadhaar-shaped numbers, account-number formats)
- **Output:** the live transcript UI shows flagged spans redacted (blacked out) by default, with an option to reveal for the meeting owner only
- **Everything runs locally** — no span or flagged content is ever transmitted

## 4. Summarization

- **Approach:** a small quantized local LLM ingests the (redacted or full, per user preference) transcript and produces: a short summary, key decisions, and action items with owners where mentioned
- **Source:** small quantized LLM available via Qualcomm AI Hub (see setup doc for current recommended model given size/latency constraints on target hardware)

## 5. Local storage

All transcripts, summaries, and speaker embeddings are stored on-disk under a local app data directory. No component in this system makes a network call at inference time. The only network access anywhere in the project is at build/setup time, to download model weights once.

## 6. Stretch goal: Local meeting Q&A (RAG)

Once multiple meetings are stored locally, a simple on-device retrieval step (embedding search over past transcripts) feeds relevant past context into the local LLM, enabling questions like "what did we decide about the vendor contract last week?" — entirely offline.
