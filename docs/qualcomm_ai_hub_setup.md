# Qualcomm AI Hub Setup

This document explains how to move each pipeline stage from its CPU dev-fallback to the Qualcomm AI Hub–optimized model running on the NPU of a Snapdragon-powered HP PC.

## 1. Install the AI Hub tooling

```bash
pip install qai-hub qai_hub_models
qai-hub configure --api_token <YOUR_TOKEN>   # from your Qualcomm AI Hub account
```

## 2. ASR: swap to Whisper-Small-Quantized

- Model page: `qualcomm/Whisper-Small-Quantized` (Qualcomm AI Hub / Hugging Face)
- Fetch a pre-exported deployable asset, or export/compile your own with custom config using the `qai_hub_models` Python library
- Replace the `WhisperBackend` implementation in `src/asr/transcribe.py` (see the `# TODO: QNN backend` marker) with a call into the compiled QNN model via ONNX Runtime's QNN Execution Provider

## 3. Diarization: convert and quantize

- This model is **not** provided directly by Qualcomm AI Hub — export your chosen open-source speaker-embedding model to ONNX, then use `qai_hub_models`' generic conversion/quantization tooling to prepare it for the AI Engine (NPU/DSP)
- This step is the clearest example of "significantly modifying an existing open-source model to add it via Qualcomm AI Hub tooling," which satisfies the challenge's proposal requirements

## 4. Redaction (NER): quantize for on-device

- Export the chosen NER model to ONNX and quantize (int8/w8a16) following the same pattern as the diarization model

## 5. Summarization: pick a small local LLM

- Check current Qualcomm AI Hub catalog for the smallest available instruction-tuned LLM that fits your target device's memory budget
- Compile/export via `qai_hub_models`, then swap the `Summarizer` implementation in `src/summarization/summarize.py`

## 6. Runtime notes

- Target execution provider: ONNX Runtime with the **QNN Execution Provider**, or the Qualcomm AI Engine Direct SDK directly for maximum control
- Test on a hosted Qualcomm AI Hub device first (via `qai-hub` cloud access) before deploying to physical hardware
- Remember: **only 64-bit x64 Python is supported on Snapdragon X Elite/X2 Elite Windows devices** — ARM64 Python will fail during install of some dependencies

## 7. Verifying "fully offline"

Before recording your demo, disable networking on the device and confirm the full pipeline (ASR → diarization → redaction → summarization) still runs end-to-end. This is a key differentiator to show live to judges.
