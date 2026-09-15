import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.asr.transcribe import TranscriptSegment
from src.diarization.diarize import SpeakerStore, diarize_segments

SR = 16000


def _make_voice(freq: float, duration: float, sr: int = SR) -> np.ndarray:
    """Generates a simple synthetic 'voice' signal at a given fundamental
    frequency, used as a stand-in for two distinct real speakers in tests."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.2 * np.sin(2 * np.pi * freq * 3 * t)
        + 0.05 * np.random.RandomState(42).randn(len(t))
    )


@pytest.fixture
def two_speaker_audio_path():
    """A, B, A, B — 2 seconds each — written to a temp wav file."""
    segments_audio = [
        _make_voice(120, 2.0),  # speaker A
        _make_voice(220, 2.0),  # speaker B
        _make_voice(120, 2.0),  # speaker A again
        _make_voice(220, 2.0),  # speaker B again
    ]
    full_audio = np.concatenate(segments_audio)
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, full_audio, SR)
    yield path
    os.remove(path)


def test_pause_fallback_without_audio():
    """When no audio_path is given, falls back to pause-based grouping."""
    store = SpeakerStore.load(tempfile.mktemp())
    segments = [
        TranscriptSegment(0.0, 2.0, "Hello."),
        TranscriptSegment(2.1, 4.0, "Still talking."),
        TranscriptSegment(20.0, 22.0, "Someone else now."),
    ]
    turns = diarize_segments(segments, store=store, pause_threshold_s=1.5)
    assert turns[0].speaker_id == turns[1].speaker_id
    assert turns[1].speaker_id != turns[2].speaker_id


def test_audio_based_clustering_separates_two_speakers(two_speaker_audio_path):
    store = SpeakerStore.load(tempfile.mktemp())
    segments = [
        TranscriptSegment(0.0, 2.0, "Speaker A first turn."),
        TranscriptSegment(2.0, 4.0, "Speaker B first turn."),
        TranscriptSegment(4.0, 6.0, "Speaker A second turn."),
        TranscriptSegment(6.0, 8.0, "Speaker B second turn."),
    ]
    turns = diarize_segments(segments, audio_path=two_speaker_audio_path, store=store)

    # A's two turns should match each other, B's two turns should match each
    # other, and A should NOT match B.
    assert turns[0].speaker_id == turns[2].speaker_id
    assert turns[1].speaker_id == turns[3].speaker_id
    assert turns[0].speaker_id != turns[1].speaker_id


def test_speaker_recognized_across_separate_meetings(two_speaker_audio_path):
    """The key differentiator: a speaker labeled in one meeting should be
    auto-recognized (by voice) in a later, separate meeting."""
    store_path = tempfile.mktemp()

    # Meeting 1: label the two speakers
    store1 = SpeakerStore.load(store_path)
    meeting1_segments = [
        TranscriptSegment(0.0, 2.0, "Meeting 1, speaker A."),
        TranscriptSegment(2.0, 4.0, "Meeting 1, speaker B."),
    ]
    turns1 = diarize_segments(meeting1_segments, audio_path=two_speaker_audio_path, store=store1)
    store1.set_display_name(turns1[0].speaker_id, "Priya")
    store1.set_display_name(turns1[1].speaker_id, "Rahul")
    store1.save(store_path)

    # Meeting 2: same two voices, freshly loaded store — should be
    # recognized without re-labeling
    store2 = SpeakerStore.load(store_path)
    meeting2_segments = [
        TranscriptSegment(4.0, 6.0, "Meeting 2, speaker A again."),
        TranscriptSegment(6.0, 8.0, "Meeting 2, speaker B again."),
    ]
    turns2 = diarize_segments(meeting2_segments, audio_path=two_speaker_audio_path, store=store2)

    assert store2.get_display_name(turns2[0].speaker_id) == "Priya"
    assert store2.get_display_name(turns2[1].speaker_id) == "Rahul"

    os.remove(store_path)


def test_store_persists_display_names_to_disk():
    path = tempfile.mktemp()
    store = SpeakerStore.load(path)
    sid = store.match_or_register(np.random.rand(40))
    store.set_display_name(sid, "Priya")
    store.save(path)

    reloaded = SpeakerStore.load(path)
    assert reloaded.get_display_name(sid) == "Priya"
    os.remove(path)
