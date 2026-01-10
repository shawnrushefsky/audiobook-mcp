"""Resemblyzer Voice Similarity Engine.

Uses Resemblyzer's voice encoder to create 256-dimensional voice embeddings
and compare voice similarity using cosine similarity.

Reference: https://github.com/resemble-ai/Resemblyzer
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import (
    VoiceEmbeddingResult,
    VoiceSimilarityEngine,
    VoiceSimilarityEngineInfo,
    VoiceSimilarityResult,
)

# Default similarity threshold for same-speaker determination
# Based on typical values from Resemblyzer demos
DEFAULT_THRESHOLD = 0.75

# Lazy-loaded encoder singleton
_encoder = None


def _redirect_stdout_to_stderr():
    """Context manager to redirect stdout to stderr during model loading."""
    import contextlib

    @contextlib.contextmanager
    def redirect():
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            yield
        finally:
            sys.stdout = old_stdout

    return redirect()


def _load_encoder():
    """Lazily load the voice encoder."""
    global _encoder
    if _encoder is not None:
        return _encoder

    with _redirect_stdout_to_stderr():
        from resemblyzer import VoiceEncoder

        print("Loading Resemblyzer voice encoder...", file=sys.stderr, flush=True)
        _encoder = VoiceEncoder()
        print("Resemblyzer encoder loaded successfully", file=sys.stderr, flush=True)

    return _encoder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ResemblyzerEngine(VoiceSimilarityEngine):
    """Voice similarity using Resemblyzer's voice encoder."""

    @property
    def name(self) -> str:
        return "Resemblyzer"

    @property
    def engine_id(self) -> str:
        return "resemblyzer"

    def is_available(self) -> bool:
        """Check if resemblyzer is installed."""
        try:
            import resemblyzer  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> VoiceSimilarityEngineInfo:
        return VoiceSimilarityEngineInfo(
            name=self.name,
            engine_id=self.engine_id,
            description=(
                "Voice similarity using 256-dimensional speaker embeddings. "
                "Fast (~1000x realtime on GPU) and accurate for speaker verification, "
                "voice comparison, and speaker diarization tasks."
            ),
            requirements="resemblyzer (pip install resemblyzer)",
            embedding_dim=256,
            default_threshold=DEFAULT_THRESHOLD,
            extra_info={
                "model": "GE2E Speaker Encoder",
                "paper": "Generalized End-To-End Loss for Speaker Verification",
                "speed": "~1000x realtime on GPU",
                "min_audio_duration": "5-30 seconds recommended",
            },
        )

    def get_setup_instructions(self) -> str:
        return """## Resemblyzer Setup Instructions

### Installation
```bash
pip install resemblyzer
```

### Requirements
- Python 3.5+
- PyTorch
- librosa for audio preprocessing

### Audio Requirements
- WAV format recommended
- 5-30 seconds of clear speech for best results
- Longer audio gives more stable embeddings

### Threshold Guidelines
- 0.75+: Likely same speaker
- 0.60-0.75: Possibly same speaker
- <0.60: Likely different speakers
"""

    def get_embedding(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> VoiceEmbeddingResult:
        """Extract voice embedding from audio.

        Args:
            audio_path: Path to audio file.

        Returns:
            VoiceEmbeddingResult with 256-dimensional embedding.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return VoiceEmbeddingResult(
                status="error",
                error=f"Audio file not found: {audio_path}",
            )

        try:
            start_time = time.time()

            with _redirect_stdout_to_stderr():
                from resemblyzer import preprocess_wav

                encoder = _load_encoder()

                # Preprocess and encode
                wav = preprocess_wav(audio_path)
                embedding = encoder.embed_utterance(wav)

            processing_time_ms = int((time.time() - start_time) * 1000)

            return VoiceEmbeddingResult(
                status="success",
                embedding=embedding.tolist(),
                embedding_dim=len(embedding),
                processing_time_ms=processing_time_ms,
                metadata={
                    "audio_file": str(audio_path),
                    "audio_duration_samples": len(wav),
                },
            )

        except Exception as e:
            return VoiceEmbeddingResult(
                status="error",
                error=str(e),
            )

    def compare_voices(
        self,
        audio_path_1: str | Path,
        audio_path_2: str | Path,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> VoiceSimilarityResult:
        """Compare two audio files for voice similarity.

        Args:
            audio_path_1: Path to first audio file (e.g., reference voice).
            audio_path_2: Path to second audio file (e.g., generated TTS).
            threshold: Similarity threshold for same-speaker (default: 0.75).

        Returns:
            VoiceSimilarityResult with cosine similarity score.
        """
        audio_path_1 = Path(audio_path_1)
        audio_path_2 = Path(audio_path_2)
        threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

        # Check files exist
        if not audio_path_1.exists():
            return VoiceSimilarityResult(
                status="error",
                error=f"Audio file not found: {audio_path_1}",
            )
        if not audio_path_2.exists():
            return VoiceSimilarityResult(
                status="error",
                error=f"Audio file not found: {audio_path_2}",
            )

        try:
            start_time = time.time()

            with _redirect_stdout_to_stderr():
                from resemblyzer import preprocess_wav

                encoder = _load_encoder()

                # Preprocess both audio files
                wav1 = preprocess_wav(audio_path_1)
                wav2 = preprocess_wav(audio_path_2)

                # Get embeddings
                embed1 = encoder.embed_utterance(wav1)
                embed2 = encoder.embed_utterance(wav2)

            # Compute cosine similarity
            similarity = _cosine_similarity(embed1, embed2)
            is_same = similarity >= threshold

            processing_time_ms = int((time.time() - start_time) * 1000)

            return VoiceSimilarityResult(
                status="success",
                similarity_score=similarity,
                is_same_speaker=is_same,
                threshold_used=threshold,
                embedding_dim=256,
                processing_time_ms=processing_time_ms,
                metadata={
                    "audio_file_1": str(audio_path_1),
                    "audio_file_2": str(audio_path_2),
                },
            )

        except Exception as e:
            return VoiceSimilarityResult(
                status="error",
                error=str(e),
            )

    def compare_embedding_to_audio(
        self,
        embedding: list[float],
        audio_path: str | Path,
        threshold: Optional[float] = None,
    ) -> VoiceSimilarityResult:
        """Compare a pre-computed embedding to an audio file.

        Useful when you have a reference embedding stored and want to compare
        against multiple audio files without re-computing the reference.

        Args:
            embedding: Pre-computed 256-dimensional voice embedding.
            audio_path: Path to audio file to compare.
            threshold: Similarity threshold for same-speaker (default: 0.75).

        Returns:
            VoiceSimilarityResult with cosine similarity score.
        """
        audio_path = Path(audio_path)
        threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

        if not audio_path.exists():
            return VoiceSimilarityResult(
                status="error",
                error=f"Audio file not found: {audio_path}",
            )

        if len(embedding) != 256:
            return VoiceSimilarityResult(
                status="error",
                error=f"Invalid embedding dimension: {len(embedding)}, expected 256",
            )

        try:
            start_time = time.time()

            with _redirect_stdout_to_stderr():
                from resemblyzer import preprocess_wav

                encoder = _load_encoder()

                # Preprocess and encode audio
                wav = preprocess_wav(audio_path)
                audio_embed = encoder.embed_utterance(wav)

            # Compute cosine similarity
            ref_embed = np.array(embedding)
            similarity = _cosine_similarity(ref_embed, audio_embed)
            is_same = similarity >= threshold

            processing_time_ms = int((time.time() - start_time) * 1000)

            return VoiceSimilarityResult(
                status="success",
                similarity_score=similarity,
                is_same_speaker=is_same,
                threshold_used=threshold,
                embedding_dim=256,
                processing_time_ms=processing_time_ms,
                metadata={
                    "audio_file": str(audio_path),
                    "reference": "pre-computed embedding",
                },
            )

        except Exception as e:
            return VoiceSimilarityResult(
                status="error",
                error=str(e),
            )
