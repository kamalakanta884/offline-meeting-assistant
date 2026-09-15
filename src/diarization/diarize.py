"""
Speaker diarization + persistent local speaker memory.

Groups transcript segments into speaker turns using real audio-derived
speaker embeddings (see embeddings.py), clustered with agglomerative
clustering. Recognizes returning speakers across meetings by comparing new
embeddings against a persistent local store using cosine similarity.

Deployment target: swap `extract_segment_embedding` for a neural speaker
embedding model (converted/quantized, running on the NPU) for higher
accuracy — the clustering and persistent-memory logic below stays the same
either way. See docs/architecture.md#speaker-diarization.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.asr.transcribe import TranscriptSegment
from src.diarization.embeddings import (
    cosine_similarity,
    extract_segment_embedding,
    load_audio,
)

SPEAKER_STORE_PATH = os.path.expanduser("~/.sahaysar/speakers.json")

# Cosine similarity above this threshold => treat as the same known speaker.
# Must be set higher than the typical similarity between two *different*
# speakers' averaged embeddings (empirically ~0.8-0.85 with MFCC features),
# or distinct speakers will get merged into one stored identity.
MATCH_THRESHOLD = 0.92

# Distance threshold used for clustering segments *within* one meeting
# (1 - cosine similarity, so lower = stricter about grouping as same speaker).
# Tuned empirically: same-speaker segments typically land above ~0.9 cosine
# similarity (distance < 0.1), while different speakers land noticeably
# lower. This may need re-tuning against real meeting audio.
WITHIN_MEETING_CLUSTER_DISTANCE = 0.15


@dataclass
class SpeakerTurn:
    speaker_id: str
    speaker_label: str
    start: float
    end: float
    text: str


@dataclass
class SpeakerStore:
    """Persistent local store mapping speaker_id -> embedding + display name."""

    speakers: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = SPEAKER_STORE_PATH) -> "SpeakerStore":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return cls(speakers=data)
        return cls(speakers={})

    def save(self, path: str = SPEAKER_STORE_PATH) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.speakers, f, indent=2)

    def match_or_register(self, embedding: np.ndarray) -> str:
        """
        Compares `embedding` against every stored speaker's saved embedding.
        Returns the id of the best match if it's above MATCH_THRESHOLD,
        otherwise registers a brand-new speaker and returns its new id.
        """
        best_id = None
        best_score = -1.0

        for speaker_id, data in self.speakers.items():
            stored_embedding = np.array(data.get("embedding") or [])
            if stored_embedding.size == 0:
                continue
            score = cosine_similarity(embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_id is not None and best_score >= MATCH_THRESHOLD:
            return best_id

        new_id = str(uuid.uuid4())[:8]
        self.speakers[new_id] = {
            "display_name": f"Speaker {len(self.speakers) + 1}",
            "embedding": embedding.tolist(),
        }
        return new_id

    def set_display_name(self, speaker_id: str, name: str) -> None:
        if speaker_id in self.speakers:
            self.speakers[speaker_id]["display_name"] = name

    def get_display_name(self, speaker_id: str) -> str:
        return self.speakers.get(speaker_id, {}).get("display_name", speaker_id)


def _cluster_embeddings(embeddings: List[np.ndarray]) -> List[int]:
    """
    Clusters same-meeting segment embeddings into speakers using
    agglomerative clustering on cosine distance. Returns a cluster label
    per input embedding. Falls back to a single cluster if there's only
    one segment (clustering needs at least 2 points).
    """
    if len(embeddings) <= 1:
        return [0] * len(embeddings)

    matrix = np.vstack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=WITHIN_MEETING_CLUSTER_DISTANCE,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(matrix)
    return labels.tolist()


def diarize_segments(
    segments: List[TranscriptSegment],
    audio_path: Optional[str] = None,
    store: Optional[SpeakerStore] = None,
    pause_threshold_s: float = 1.5,
) -> List[SpeakerTurn]:
    """
    Groups transcript segments into speaker turns.

    If `audio_path` is provided, uses real MFCC-based speaker embeddings
    (clustered within the meeting, then matched against the persistent
    store to recognize returning speakers across meetings).

    If no `audio_path` is given (e.g. in tests, or if audio isn't
    available), falls back to a simple pause-based heuristic so the rest of
    the pipeline still runs.
    """
    store = store or SpeakerStore.load()

    if audio_path is None:
        return _diarize_by_pause(segments, store, pause_threshold_s)

    audio, sr = load_audio(audio_path)
    embeddings = [
        extract_segment_embedding(audio, sr, seg.start, seg.end) for seg in segments
    ]

    cluster_labels = _cluster_embeddings(embeddings)

    # Average the embeddings within each local cluster, then match each
    # cluster (not each individual segment) against the persistent store —
    # this is both more accurate and means each speaker is only looked up
    # once per meeting rather than once per utterance.
    cluster_to_speaker_id: Dict[int, str] = {}
    for cluster_id in set(cluster_labels):
        member_embeddings = [
            emb for emb, label in zip(embeddings, cluster_labels) if label == cluster_id
        ]
        cluster_embedding = np.mean(member_embeddings, axis=0)
        cluster_to_speaker_id[cluster_id] = store.match_or_register(cluster_embedding)

    turns = [
        SpeakerTurn(
            speaker_id=cluster_to_speaker_id[label],
            speaker_label=store.get_display_name(cluster_to_speaker_id[label]),
            start=seg.start,
            end=seg.end,
            text=seg.text,
        )
        for seg, label in zip(segments, cluster_labels)
    ]

    store.save()
    return turns


def _diarize_by_pause(
    segments: List[TranscriptSegment],
    store: SpeakerStore,
    pause_threshold_s: float,
) -> List[SpeakerTurn]:
    """Fallback used when no audio is available for embedding extraction."""
    turns: List[SpeakerTurn] = []
    current_speaker_id = str(uuid.uuid4())[:8]
    store.speakers.setdefault(
        current_speaker_id, {"display_name": f"Speaker {len(store.speakers) + 1}", "embedding": []}
    )
    last_end = None

    for seg in segments:
        if last_end is not None and (seg.start - last_end) > pause_threshold_s:
            current_speaker_id = str(uuid.uuid4())[:8]
            store.speakers.setdefault(
                current_speaker_id,
                {"display_name": f"Speaker {len(store.speakers) + 1}", "embedding": []},
            )

        turns.append(
            SpeakerTurn(
                speaker_id=current_speaker_id,
                speaker_label=store.get_display_name(current_speaker_id),
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
        )
        last_end = seg.end

    store.save()
    return turns
