"""Base classes and interfaces for audio analysis engines.

This module defines the abstract interfaces for audio analysis engines including:
- Emotion detection
- Voice similarity comparison
- Speech quality assessment
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# Emotion Detection
# ============================================================================


@dataclass
class EmotionScore:
    """Score for a single emotion category."""

    emotion: str  # e.g., "happy", "angry", "neutral"
    score: float  # 0.0 to 1.0 probability/confidence


@dataclass
class EmotionResult:
    """Result from emotion detection."""

    status: str  # "success" or "error"
    primary_emotion: Optional[str] = None  # Highest scoring emotion
    primary_score: Optional[float] = None  # Score of primary emotion
    all_emotions: list[EmotionScore] = field(default_factory=list)  # All emotion scores
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EmotionEngineInfo:
    """Information about an emotion detection engine."""

    name: str
    engine_id: str
    description: str
    requirements: str
    supported_emotions: list[str]
    extra_info: dict = field(default_factory=dict)


class EmotionEngine(ABC):
    """Abstract base class for emotion detection engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this engine."""
        pass

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available."""
        pass

    @abstractmethod
    def get_info(self) -> EmotionEngineInfo:
        """Get detailed information about this engine."""
        pass

    @abstractmethod
    def detect_emotion(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> EmotionResult:
        """Detect emotion in audio.

        Args:
            audio_path: Path to the audio file to analyze.
            **kwargs: Engine-specific parameters.

        Returns:
            EmotionResult with detected emotions and scores.
        """
        pass

    def get_setup_instructions(self) -> str:
        """Get setup instructions for this engine."""
        return f"No setup instructions available for {self.name}."


# ============================================================================
# Voice Similarity
# ============================================================================


@dataclass
class VoiceSimilarityResult:
    """Result from voice similarity comparison."""

    status: str  # "success" or "error"
    similarity_score: Optional[float] = None  # 0.0 to 1.0 (1.0 = identical)
    is_same_speaker: Optional[bool] = None  # Above threshold?
    threshold_used: Optional[float] = None  # Threshold for same speaker
    embedding_dim: Optional[int] = None  # Dimension of voice embeddings
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class VoiceEmbeddingResult:
    """Result from voice embedding extraction."""

    status: str  # "success" or "error"
    embedding: Optional[list[float]] = None  # Voice embedding vector
    embedding_dim: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class VoiceSimilarityEngineInfo:
    """Information about a voice similarity engine."""

    name: str
    engine_id: str
    description: str
    requirements: str
    embedding_dim: int
    default_threshold: float  # Default threshold for same-speaker
    extra_info: dict = field(default_factory=dict)


class VoiceSimilarityEngine(ABC):
    """Abstract base class for voice similarity engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this engine."""
        pass

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available."""
        pass

    @abstractmethod
    def get_info(self) -> VoiceSimilarityEngineInfo:
        """Get detailed information about this engine."""
        pass

    @abstractmethod
    def compare_voices(
        self,
        audio_path_1: str | Path,
        audio_path_2: str | Path,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> VoiceSimilarityResult:
        """Compare two audio files for voice similarity.

        Args:
            audio_path_1: Path to first audio file.
            audio_path_2: Path to second audio file.
            threshold: Similarity threshold for same-speaker (default varies by engine).
            **kwargs: Engine-specific parameters.

        Returns:
            VoiceSimilarityResult with similarity score.
        """
        pass

    @abstractmethod
    def get_embedding(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> VoiceEmbeddingResult:
        """Extract voice embedding from audio.

        Args:
            audio_path: Path to audio file.
            **kwargs: Engine-specific parameters.

        Returns:
            VoiceEmbeddingResult with embedding vector.
        """
        pass

    def get_setup_instructions(self) -> str:
        """Get setup instructions for this engine."""
        return f"No setup instructions available for {self.name}."


# ============================================================================
# Speech Quality Assessment
# ============================================================================


@dataclass
class QualityDimension:
    """Score for a single quality dimension."""

    name: str  # e.g., "naturalness", "clarity", "noisiness"
    score: float  # Typically 1-5 MOS scale
    description: Optional[str] = None


@dataclass
class SpeechQualityResult:
    """Result from speech quality assessment."""

    status: str  # "success" or "error"
    overall_quality: Optional[float] = None  # Overall MOS score (1-5)
    dimensions: list[QualityDimension] = field(default_factory=list)
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SpeechQualityEngineInfo:
    """Information about a speech quality engine."""

    name: str
    engine_id: str
    description: str
    requirements: str
    quality_dimensions: list[str]  # Dimensions this engine can assess
    score_range: tuple[float, float]  # Min and max score values
    extra_info: dict = field(default_factory=dict)


class SpeechQualityEngine(ABC):
    """Abstract base class for speech quality assessment engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this engine."""
        pass

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available."""
        pass

    @abstractmethod
    def get_info(self) -> SpeechQualityEngineInfo:
        """Get detailed information about this engine."""
        pass

    @abstractmethod
    def assess_quality(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> SpeechQualityResult:
        """Assess speech quality of audio.

        Args:
            audio_path: Path to audio file to assess.
            **kwargs: Engine-specific parameters.

        Returns:
            SpeechQualityResult with quality scores.
        """
        pass

    def get_setup_instructions(self) -> str:
        """Get setup instructions for this engine."""
        return f"No setup instructions available for {self.name}."
