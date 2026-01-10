"""Audio Analysis Module.

Provides tools for analyzing audio to enable agent self-verification of TTS output:
- Emotion detection: Verify emotional tone matches intent
- Voice similarity: Compare generated voice to reference
- Speech quality: Assess naturalness and technical quality
- SFX analysis: Loudness, clipping, spectrum, silence, format validation

Example usage:
    from talky_talky.tools.analysis import (
        detect_emotion,
        compare_voices,
        assess_quality,
    )

    # Detect emotion in generated audio
    emotion_result = detect_emotion("output.wav", engine="emotion2vec")
    print(f"Primary emotion: {emotion_result.primary_emotion}")

    # Compare voice similarity
    similarity_result = compare_voices(
        "reference.wav",
        "generated.wav",
        engine="resemblyzer",
    )
    print(f"Similarity: {similarity_result.similarity_score:.2%}")

    # Assess speech quality
    quality_result = assess_quality("output.wav", engine="nisqa")
    print(f"Overall MOS: {quality_result.overall_quality:.2f}")
"""

from dataclasses import asdict
from typing import Optional

from .base import (
    # Emotion detection
    EmotionEngine,
    EmotionEngineInfo,
    EmotionResult,
    EmotionScore,
    # Voice similarity
    VoiceEmbeddingResult,
    VoiceSimilarityEngine,
    VoiceSimilarityEngineInfo,
    VoiceSimilarityResult,
    # Speech quality
    QualityDimension,
    SpeechQualityEngine,
    SpeechQualityEngineInfo,
    SpeechQualityResult,
)

# SFX analysis (non-speech audio)
from .sfx import (
    LoudnessResult,
    ClippingResult,
    SpectralResult,
    SilenceResult,
    FormatValidationResult,
    analyze_loudness,
    detect_clipping,
    analyze_spectrum,
    detect_silence,
    validate_format,
    get_sfx_analysis_info,
)

# Engine registries
_emotion_engines: dict[str, EmotionEngine] = {}
_similarity_engines: dict[str, VoiceSimilarityEngine] = {}
_quality_engines: dict[str, SpeechQualityEngine] = {}


def register_emotion_engine(engine_class: type[EmotionEngine]) -> None:
    """Register an emotion detection engine."""
    engine = engine_class()
    _emotion_engines[engine.engine_id] = engine


def register_similarity_engine(engine_class: type[VoiceSimilarityEngine]) -> None:
    """Register a voice similarity engine."""
    engine = engine_class()
    _similarity_engines[engine.engine_id] = engine


def register_quality_engine(engine_class: type[SpeechQualityEngine]) -> None:
    """Register a speech quality engine."""
    engine = engine_class()
    _quality_engines[engine.engine_id] = engine


# Register available engines
try:
    from .emotion2vec import Emotion2vecEngine

    register_emotion_engine(Emotion2vecEngine)
except ImportError:
    pass

try:
    from .resemblyzer import ResemblyzerEngine

    register_similarity_engine(ResemblyzerEngine)
except ImportError:
    pass

try:
    from .nisqa import NISQAEngine

    register_quality_engine(NISQAEngine)
except ImportError:
    pass


# ============================================================================
# Engine Access Functions
# ============================================================================


def list_emotion_engines() -> dict[str, EmotionEngineInfo]:
    """List all registered emotion detection engines."""
    return {eid: engine.get_info() for eid, engine in _emotion_engines.items()}


def list_similarity_engines() -> dict[str, VoiceSimilarityEngineInfo]:
    """List all registered voice similarity engines."""
    return {eid: engine.get_info() for eid, engine in _similarity_engines.items()}


def list_quality_engines() -> dict[str, SpeechQualityEngineInfo]:
    """List all registered speech quality engines."""
    return {eid: engine.get_info() for eid, engine in _quality_engines.items()}


def get_emotion_engine(engine_id: str) -> EmotionEngine:
    """Get an emotion detection engine by ID."""
    if engine_id not in _emotion_engines:
        available = list(_emotion_engines.keys())
        raise ValueError(f"Unknown emotion engine: {engine_id}. Available: {available}")
    return _emotion_engines[engine_id]


def get_similarity_engine(engine_id: str) -> VoiceSimilarityEngine:
    """Get a voice similarity engine by ID."""
    if engine_id not in _similarity_engines:
        available = list(_similarity_engines.keys())
        raise ValueError(f"Unknown similarity engine: {engine_id}. Available: {available}")
    return _similarity_engines[engine_id]


def get_quality_engine(engine_id: str) -> SpeechQualityEngine:
    """Get a speech quality engine by ID."""
    if engine_id not in _quality_engines:
        available = list(_quality_engines.keys())
        raise ValueError(f"Unknown quality engine: {engine_id}. Available: {available}")
    return _quality_engines[engine_id]


# ============================================================================
# High-Level API Functions
# ============================================================================


def detect_emotion(
    audio_path: str,
    engine: str = "emotion2vec",
    **kwargs,
) -> EmotionResult:
    """Detect emotion in audio file.

    Args:
        audio_path: Path to audio file.
        engine: Engine to use (default: "emotion2vec").
        **kwargs: Engine-specific parameters.

    Returns:
        EmotionResult with detected emotions and confidence scores.

    Example:
        result = detect_emotion("speech.wav")
        if result.status == "success":
            print(f"Emotion: {result.primary_emotion} ({result.primary_score:.1%})")
    """
    eng = get_emotion_engine(engine)
    if not eng.is_available():
        return EmotionResult(
            status="error",
            error=f"Engine '{engine}' is not available. {eng.get_setup_instructions()}",
        )
    return eng.detect_emotion(audio_path, **kwargs)


def compare_voices(
    audio_path_1: str,
    audio_path_2: str,
    engine: str = "resemblyzer",
    threshold: Optional[float] = None,
    **kwargs,
) -> VoiceSimilarityResult:
    """Compare two audio files for voice similarity.

    Args:
        audio_path_1: Path to first audio file (e.g., reference voice).
        audio_path_2: Path to second audio file (e.g., generated TTS).
        engine: Engine to use (default: "resemblyzer").
        threshold: Similarity threshold for same-speaker determination.
        **kwargs: Engine-specific parameters.

    Returns:
        VoiceSimilarityResult with similarity score and same-speaker determination.

    Example:
        result = compare_voices("reference.wav", "generated.wav")
        if result.status == "success":
            print(f"Similarity: {result.similarity_score:.1%}")
            print(f"Same speaker: {result.is_same_speaker}")
    """
    eng = get_similarity_engine(engine)
    if not eng.is_available():
        return VoiceSimilarityResult(
            status="error",
            error=f"Engine '{engine}' is not available. {eng.get_setup_instructions()}",
        )
    return eng.compare_voices(audio_path_1, audio_path_2, threshold=threshold, **kwargs)


def get_voice_embedding(
    audio_path: str,
    engine: str = "resemblyzer",
    **kwargs,
) -> VoiceEmbeddingResult:
    """Extract voice embedding from audio.

    Args:
        audio_path: Path to audio file.
        engine: Engine to use (default: "resemblyzer").
        **kwargs: Engine-specific parameters.

    Returns:
        VoiceEmbeddingResult with embedding vector.

    Example:
        result = get_voice_embedding("reference.wav")
        if result.status == "success":
            print(f"Embedding dim: {result.embedding_dim}")
            # Store embedding for later comparison
    """
    eng = get_similarity_engine(engine)
    if not eng.is_available():
        return VoiceEmbeddingResult(
            status="error",
            error=f"Engine '{engine}' is not available. {eng.get_setup_instructions()}",
        )
    return eng.get_embedding(audio_path, **kwargs)


def assess_quality(
    audio_path: str,
    engine: str = "nisqa",
    **kwargs,
) -> SpeechQualityResult:
    """Assess speech quality of audio file.

    Args:
        audio_path: Path to audio file.
        engine: Engine to use (default: "nisqa").
        **kwargs: Engine-specific parameters.

    Returns:
        SpeechQualityResult with MOS score and quality dimensions.

    Example:
        result = assess_quality("generated.wav")
        if result.status == "success":
            print(f"Overall MOS: {result.overall_quality:.2f}/5.0")
            for dim in result.dimensions:
                print(f"  {dim.name}: {dim.score:.2f}")
    """
    eng = get_quality_engine(engine)
    if not eng.is_available():
        return SpeechQualityResult(
            status="error",
            error=f"Engine '{engine}' is not available. {eng.get_setup_instructions()}",
        )
    return eng.assess_quality(audio_path, **kwargs)


def to_dict(result) -> dict:
    """Convert a result dataclass to a dictionary."""
    return asdict(result)


__all__ = [
    # Base classes
    "EmotionEngine",
    "EmotionEngineInfo",
    "EmotionResult",
    "EmotionScore",
    "VoiceSimilarityEngine",
    "VoiceSimilarityEngineInfo",
    "VoiceSimilarityResult",
    "VoiceEmbeddingResult",
    "SpeechQualityEngine",
    "SpeechQualityEngineInfo",
    "SpeechQualityResult",
    "QualityDimension",
    # Engine access
    "list_emotion_engines",
    "list_similarity_engines",
    "list_quality_engines",
    "get_emotion_engine",
    "get_similarity_engine",
    "get_quality_engine",
    # High-level API
    "detect_emotion",
    "compare_voices",
    "get_voice_embedding",
    "assess_quality",
    "to_dict",
    # SFX Analysis
    "LoudnessResult",
    "ClippingResult",
    "SpectralResult",
    "SilenceResult",
    "FormatValidationResult",
    "analyze_loudness",
    "detect_clipping",
    "analyze_spectrum",
    "detect_silence",
    "validate_format",
    "get_sfx_analysis_info",
]
