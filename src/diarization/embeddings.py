"""
Speaker embedding extraction from raw audio.

This uses MFCC (Mel-Frequency Cepstral Coefficient) statistics as a speaker
embedding. This is a classical, well-established technique for capturing
voice-characteristic information (timbre, pitch range) from short audio
segments, and it requires no downloaded model weights — everything runs
locally with librosa + numpy.

This is a real, working speaker-discrimination signal (much better than
pause-based grouping), while still being lighter-weight than a full neural
speaker-embedding model. See docs/architecture.md for the option to swap in
a neural embedding model (e.g. a converted/quantized x-vector or ECAPA-TDNN
model) for higher accuracy once one is integrated for the NPU.
"""

import numpy as np
import librosa


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """Loads an audio file, resampled to `target_sr`, mono."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def extract_segment_embedding(
    audio: np.ndarray, sr: int, start: float, end: float, n_mfcc: int = 20
) -> np.ndarray:
    """
    Extracts a fixed-length embedding for the audio between `start` and
    `end` seconds, by computing MFCCs across the segment and summarizing
    them with mean + standard deviation per coefficient (a standard way to
    turn a variable-length MFCC sequence into a fixed-length vector).

    Returns a zero vector if the segment is too short to analyze (so
    callers can safely handle edge cases without crashing).
    """
    start_sample = max(0, int(start * sr))
    end_sample = min(len(audio), int(end * sr))

    if end_sample - start_sample < sr * 0.1:  # shorter than 100ms — too short
        return np.zeros(n_mfcc * 2)

    segment = audio[start_sample:end_sample]

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    embedding = np.concatenate([mean, std])
    # L2-normalize so cosine similarity comparisons are well-behaved
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
