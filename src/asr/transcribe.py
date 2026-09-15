"""
Speech-to-text stage.

Dev fallback: uses `faster-whisper` on CPU so the pipeline is runnable on any
machine during development.

Deployment target: Qualcomm AI Hub's `Whisper-Small-Quantized` model, running
on the NPU via the QNN Execution Provider. See docs/qualcomm_ai_hub_setup.md.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class ASRBackend:
    """Interface every ASR backend must implement."""

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        raise NotImplementedError


class FasterWhisperBackend(ASRBackend):
    """CPU dev-fallback backend using faster-whisper."""

    def __init__(self, model_size: str = "small"):
        # Imported lazily so the rest of the repo can be inspected/tested
        # without requiring this dependency to be installed.
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        segments, _info = self.model.transcribe(audio_path, beam_size=5)
        return [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments
        ]


class QNNWhisperBackend(ASRBackend):
    """
    Deployment backend: Qualcomm AI Hub Whisper-Small-Quantized on the NPU.

    TODO: QNN backend
    Replace this stub with a real implementation once the model has been
    exported/compiled via `qai_hub_models` and is available as an ONNX
    Runtime session using the QNN Execution Provider.

    See docs/qualcomm_ai_hub_setup.md for step-by-step setup.
    """

    def __init__(self, encoder_path: str, decoder_path: str):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        raise NotImplementedError(
            "QNN Whisper backend not yet wired up — see docs/qualcomm_ai_hub_setup.md"
        )

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        raise NotImplementedError


def get_default_backend() -> ASRBackend:
    """Returns the CPU dev-fallback backend by default."""
    return FasterWhisperBackend()
