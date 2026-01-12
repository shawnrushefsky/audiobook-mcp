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


# ============================================================================
# TTS Verification Functions
# ============================================================================


# Tag format mappings for different TTS engines
TTS_TAG_FORMATS = {
    # Maya1 uses angle brackets: <laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited>, <sad>
    "maya1": {
        "format": "angle",  # <tag>
        "tags": ["laugh", "chuckle", "sigh", "gasp", "whisper", "angry", "excited", "sad"],
    },
    # Chatterbox uses square brackets: [laugh], [chuckle], [sigh], [gasp], [cough], etc.
    "chatterbox": {
        "format": "square",  # [tag]
        "tags": [
            "laugh",
            "chuckle",
            "cough",
            "sigh",
            "gasp",
            "groan",
            "yawn",
            "sniff",
            "clearing throat",
            "hmm",
            "uh",
            "um",
            "oh",
            "ah",
            "wow",
            "ooh",
            "eww",
            "huh",
            "mhm",
        ],
    },
    "chatterbox_turbo": {
        "format": "square",
        "tags": [
            "laugh",
            "chuckle",
            "cough",
            "sigh",
            "gasp",
            "groan",
            "yawn",
            "sniff",
            "clearing throat",
            "hmm",
            "uh",
            "um",
            "oh",
            "ah",
            "wow",
            "ooh",
            "eww",
            "huh",
            "mhm",
        ],
    },
    # CosyVoice uses square brackets: [breath]
    "cosyvoice": {
        "format": "square",
        "tags": ["breath"],
    },
}


def convert_tags_for_engine(text: str, target_engine: str) -> dict:
    """Convert paralinguistic tags in text to the correct format for a TTS engine.

    Different TTS engines use different tag formats:
    - Maya1: <tag> (angle brackets)
    - Chatterbox/Chatterbox Turbo: [tag] (square brackets)
    - CosyVoice: [breath] (square brackets)

    This function converts tags between formats automatically.

    Args:
        text: Text containing paralinguistic tags.
        target_engine: Target TTS engine ID ("maya1", "chatterbox", "chatterbox_turbo", "cosyvoice").

    Returns:
        Dict with:
        - converted_text: Text with tags in correct format for the engine
        - changes_made: List of tag conversions performed
        - unsupported_tags: List of tags not supported by the target engine

    Example:
        # Convert Chatterbox-style [laugh] to Maya1-style <laugh>
        result = convert_tags_for_engine("Hello [laugh] world", "maya1")
        # result["converted_text"] = "Hello <laugh> world"

        # Convert Maya1-style <sigh> to Chatterbox-style [sigh]
        result = convert_tags_for_engine("Oh <sigh> fine", "chatterbox")
        # result["converted_text"] = "Oh [sigh] fine"
    """
    import re

    target_engine_lower = target_engine.lower()
    engine_config = TTS_TAG_FORMATS.get(target_engine_lower)

    if not engine_config:
        return {
            "converted_text": text,
            "changes_made": [],
            "unsupported_tags": [],
            "warning": f"Unknown engine '{target_engine}'. No conversion performed.",
        }

    target_format = engine_config["format"]
    supported_tags = set(tag.lower() for tag in engine_config["tags"])

    changes_made = []
    unsupported_tags = []
    converted_text = text

    # Find all tags in both formats
    # Square bracket pattern: [tag]
    square_pattern = r"\[([^\]]+)\]"
    # Angle bracket pattern: <tag>
    angle_pattern = r"<([^>]+)>"

    def convert_tag(match, source_format):
        tag = match.group(1).lower()

        # Check if tag is supported by target engine
        if tag not in supported_tags:
            unsupported_tags.append(tag)
            # Return original unchanged for unsupported tags
            return match.group(0)

        # Convert to target format
        if target_format == "angle" and source_format == "square":
            changes_made.append(f"[{tag}] -> <{tag}>")
            return f"<{tag}>"
        elif target_format == "square" and source_format == "angle":
            changes_made.append(f"<{tag}> -> [{tag}]")
            return f"[{tag}]"
        else:
            # Same format, no change needed
            return match.group(0)

    # Convert square brackets to target format
    if target_format == "angle":
        converted_text = re.sub(square_pattern, lambda m: convert_tag(m, "square"), converted_text)
    # Convert angle brackets to target format
    elif target_format == "square":
        converted_text = re.sub(angle_pattern, lambda m: convert_tag(m, "angle"), converted_text)

    return {
        "converted_text": converted_text,
        "changes_made": changes_made,
        "unsupported_tags": list(set(unsupported_tags)),
    }


def detect_spoken_tags(
    audio_path: str,
    tags: Optional[list[str]] = None,
    tts_engine: Optional[str] = None,
    engine: str = "faster_whisper",
    model_size: str = "base",
) -> dict:
    """Detect if TTS spoke tags as words instead of performing them.

    Checks if TTS engines incorrectly spoke tags like "[chuckle]" or "<laugh>"
    as literal words instead of performing the intended action. This is a
    common issue with some TTS engines.

    Args:
        audio_path: Path to the audio file to analyze.
        tags: List of tags to check for. If None, uses common defaults.
        tts_engine: Optional TTS engine used to generate the audio. If provided,
            returns suggestions for correctly formatted tags for that engine.
        engine: Transcription engine to use (default: "faster_whisper").
        model_size: Whisper model size (default: "base" for speed).

    Returns:
        Dict with:
        - status: "success" or "error"
        - has_spoken_tags: True if tags were spoken as words
        - spoken_tags: List of tags that were spoken as words
        - transcription: Full transcribed text
        - confidence: Overall detection confidence (0-1)
        - suggested_fixes: Dict mapping spoken tags to correct format (if tts_engine provided)
        - error: Error message (if status is "error")

    Example:
        result = detect_spoken_tags("tts_output.wav", tts_engine="maya1")
        if result["has_spoken_tags"]:
            print(f"TTS spoke these tags: {result['spoken_tags']}")
            if result.get("suggested_fixes"):
                print(f"Use these instead: {result['suggested_fixes']}")
    """
    import re
    from ..transcription import transcribe

    # Default tags to detect (common TTS emotion/sound tags)
    if tags is None:
        tags = [
            # Square bracket tags (Chatterbox style)
            "chuckle",
            "laugh",
            "sigh",
            "gasp",
            "cough",
            "groan",
            "yawn",
            "sniff",
            "clearing throat",
            "hmm",
            "um",
            "uh",
            "oh",
            "ah",
            "wow",
            "ooh",
            "eww",
            "huh",
            "mhm",
            # Angle bracket tags (Maya1 style)
            "whisper",
            "angry",
            "excited",
            "sad",
            # CosyVoice style
            "breath",
        ]

    try:
        # Transcribe the audio
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            model_size=model_size,
        )

        if result.status != "success":
            return {
                "status": "error",
                "error": result.error or "Transcription failed",
            }

        transcription = result.text.lower()

        # Check for spoken tags
        spoken_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            # Check for the tag word spoken (not in brackets)
            # Pattern: the tag word appearing as a spoken word
            # e.g., "chuckle" or "laugh" spoken literally
            pattern = r"\b" + re.escape(tag_lower) + r"\b"
            if re.search(pattern, transcription):
                spoken_tags.append(tag)

        # Calculate confidence based on transcription quality
        # (shorter transcriptions with tags are more likely false positives)
        confidence = 1.0
        if len(transcription) < 20 and spoken_tags:
            confidence = 0.7  # Lower confidence for very short audio
        if len(spoken_tags) > 3:
            confidence = 0.9  # Multiple tags increases likelihood of issues

        response = {
            "status": "success",
            "has_spoken_tags": len(spoken_tags) > 0,
            "spoken_tags": spoken_tags,
            "transcription": result.text,
            "confidence": confidence,
        }

        # If TTS engine is specified, suggest correct tag format
        if tts_engine and spoken_tags:
            engine_lower = tts_engine.lower()
            engine_config = TTS_TAG_FORMATS.get(engine_lower)

            if engine_config:
                target_format = engine_config["format"]
                supported_tags = set(tag.lower() for tag in engine_config["tags"])

                suggested_fixes = {}
                for tag in spoken_tags:
                    tag_lower = tag.lower()
                    if tag_lower in supported_tags:
                        if target_format == "angle":
                            suggested_fixes[tag] = f"<{tag}>"
                        else:
                            suggested_fixes[tag] = f"[{tag}]"
                    else:
                        suggested_fixes[tag] = f"(not supported by {tts_engine})"

                response["suggested_fixes"] = suggested_fixes
                response["tts_engine"] = tts_engine
                response["tag_format"] = target_format

        return response

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def compare_audio_to_text(
    audio_path: str,
    expected_text: str,
    engine: str = "faster_whisper",
    model_size: str = "base",
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
) -> dict:
    """Quick comparison of audio content to expected text.

    A lightweight alternative to verify_tts_output that focuses only on
    text matching without quality/emotion checks. Faster for simple
    "does this audio say what it should?" checks.

    Args:
        audio_path: Path to the audio file to check.
        expected_text: The text that should be in the audio.
        engine: Transcription engine to use (default: "faster_whisper").
        model_size: Whisper model size (default: "base" for speed).
        ignore_case: Ignore case differences (default: True).
        ignore_punctuation: Ignore punctuation differences (default: True).

    Returns:
        Dict with:
        - status: "success" or "error"
        - matches: True if transcription matches expected text
        - similarity: Similarity ratio (0.0 to 1.0)
        - transcribed_text: What was actually transcribed
        - expected_text: The normalized expected text
        - differences: List of word-level differences (if any)
        - error: Error message (if status is "error")

    Example:
        result = compare_audio_to_text(
            "greeting.wav",
            "Hello, how are you today?"
        )
        if result["matches"]:
            print("Audio matches expected text!")
        else:
            print(f"Similarity: {result['similarity']:.1%}")
    """
    import re
    from difflib import SequenceMatcher
    from ..transcription import transcribe

    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        if ignore_case:
            text = text.lower()
        if ignore_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text

    try:
        # Transcribe the audio
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            model_size=model_size,
        )

        if result.status != "success":
            return {
                "status": "error",
                "error": result.error or "Transcription failed",
            }

        # Normalize both texts
        norm_transcribed = normalize_text(result.text)
        norm_expected = normalize_text(expected_text)

        # Calculate similarity
        matcher = SequenceMatcher(None, norm_transcribed, norm_expected)
        similarity = matcher.ratio()

        # Find word-level differences
        transcribed_words = norm_transcribed.split()
        expected_words = norm_expected.split()

        differences = []
        word_matcher = SequenceMatcher(None, transcribed_words, expected_words)
        for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
            if tag != "equal":
                differences.append(
                    {
                        "type": tag,
                        "transcribed": " ".join(transcribed_words[i1:i2]) if i1 < i2 else None,
                        "expected": " ".join(expected_words[j1:j2]) if j1 < j2 else None,
                    }
                )

        # Consider a match if similarity is above threshold
        matches = similarity >= 0.9

        return {
            "status": "success",
            "matches": matches,
            "similarity": similarity,
            "transcribed_text": result.text,
            "expected_text": expected_text,
            "differences": differences if not matches else [],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ============================================================================
# Speech Boundary Detection Functions
# ============================================================================


def detect_speech_onset(
    audio_path: str,
    approximate_ms: float,
    search_window_ms: float = 150,
    energy_threshold: float = 0.1,
) -> dict:
    """Given a rough timestamp, find the precise speech onset using energy detection.

    Analyzes waveform energy to find where voiced audio actually begins within
    a search window around the approximate timestamp. This is useful for aligning
    audio to precise timestamps.

    Args:
        audio_path: Path to the audio file.
        approximate_ms: Approximate timestamp where speech should start.
        search_window_ms: Search window size in ms (default: 150ms).
            Searches ±search_window_ms around approximate_ms.
        energy_threshold: Energy rise threshold (0-1, default: 0.1).
            Lower values detect quieter onsets.

    Returns:
        Dict with:
        - status: "success" or "error"
        - onset_ms: Precise millisecond where voiced audio begins
        - confidence: Detection confidence (0-1)
        - search_start_ms: Start of search window
        - search_end_ms: End of search window
        - approximate_ms: Original approximate timestamp

    Example:
        # Find precise onset near 1000ms
        result = detect_speech_onset("speech.wav", approximate_ms=1000, search_window_ms=100)
        print(f"Actual onset at {result['onset_ms']}ms")
    """
    try:
        import numpy as np

        try:
            import librosa
        except ImportError:
            return {
                "status": "error",
                "error": "librosa not installed. Install with: pip install librosa",
            }

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate search window in samples
        search_start_ms = max(0, approximate_ms - search_window_ms)
        search_end_ms = approximate_ms + search_window_ms

        start_sample = int((search_start_ms / 1000) * sr)
        end_sample = int((search_end_ms / 1000) * sr)
        end_sample = min(end_sample, len(y))

        # Extract window
        window = y[start_sample:end_sample]

        if len(window) == 0:
            return {
                "status": "error",
                "error": "Search window is outside audio bounds",
            }

        # Calculate short-term energy (RMS in small frames)
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = int(0.005 * sr)  # 5ms hop

        rms = librosa.feature.rms(y=window, frame_length=frame_length, hop_length=hop_length)[0]

        # Normalize RMS
        rms_max = np.max(rms) if np.max(rms) > 0 else 1
        rms_norm = rms / rms_max

        # Find first frame where energy exceeds threshold
        onset_frame = None
        for i, energy in enumerate(rms_norm):
            if energy >= energy_threshold:
                onset_frame = i
                break

        if onset_frame is None:
            # No speech detected, return approximate
            return {
                "status": "success",
                "onset_ms": approximate_ms,
                "confidence": 0.3,
                "search_start_ms": search_start_ms,
                "search_end_ms": search_end_ms,
                "approximate_ms": approximate_ms,
                "note": "No clear onset detected, returning approximate",
            }

        # Convert frame to milliseconds
        onset_sample = start_sample + (onset_frame * hop_length)
        onset_ms = (onset_sample / sr) * 1000

        # Calculate confidence based on energy rise clarity
        if onset_frame < len(rms_norm) - 1:
            rise = rms_norm[onset_frame + 1] - rms_norm[max(0, onset_frame - 1)]
            confidence = min(1.0, 0.5 + rise)
        else:
            confidence = 0.7

        return {
            "status": "success",
            "onset_ms": round(onset_ms, 2),
            "confidence": round(confidence, 3),
            "search_start_ms": search_start_ms,
            "search_end_ms": search_end_ms,
            "approximate_ms": approximate_ms,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def detect_truncated_audio(
    audio_path: str,
    attack_threshold_ms: float = 10,
    decay_threshold_ms: float = 50,
) -> dict:
    """Analyze if audio beginning/end sounds clipped or truncated.

    Detects if audio is missing attack transients at the start or has
    abrupt cutoffs at the end. This can catch issues like "oday" vs "Today"
    where the initial consonant was clipped.

    Args:
        audio_path: Path to the audio file.
        attack_threshold_ms: Time to reach peak energy from start (default: 10ms).
            If peak is reached too quickly, may indicate clipped attack.
        decay_threshold_ms: Minimum decay time at end (default: 50ms).
            If audio ends too abruptly, may indicate clipped ending.

    Returns:
        Dict with:
        - status: "success" or "error"
        - is_truncated: True if either start or end appears truncated
        - start_clipped: True if beginning appears clipped
        - end_clipped: True if ending appears abruptly cut
        - attack_time_ms: Time from start to first energy peak
        - decay_time_ms: Time from last peak to end
        - start_confidence: Confidence in start clipping detection (0-1)
        - end_confidence: Confidence in end clipping detection (0-1)
        - suggestions: List of suggested fixes

    Example:
        result = detect_truncated_audio("segment.wav")
        if result["is_truncated"]:
            print("Audio may be clipped!")
            if result["start_clipped"]:
                print("- Beginning may be cut off")
    """
    try:
        import numpy as np

        try:
            import librosa
        except ImportError:
            return {
                "status": "error",
                "error": "librosa not installed. Install with: pip install librosa",
            }

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration_ms = (len(y) / sr) * 1000

        # Calculate RMS energy
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = int(0.005 * sr)  # 5ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if len(rms) < 3:
            return {
                "status": "error",
                "error": "Audio too short for analysis",
            }

        # Normalize
        rms_max = np.max(rms) if np.max(rms) > 0 else 1
        rms_norm = rms / rms_max

        # Analyze start - check how quickly energy rises
        peak_threshold = 0.5  # Consider it a peak if above 50% of max
        attack_frame = None
        for i, energy in enumerate(rms_norm):
            if energy >= peak_threshold:
                attack_frame = i
                break

        attack_time_ms = (attack_frame * hop_length / sr) * 1000 if attack_frame else 0

        # Check if first frame already has significant energy (suggests clipped start)
        start_energy = rms_norm[0] if len(rms_norm) > 0 else 0
        start_clipped = start_energy > 0.3 and attack_time_ms < attack_threshold_ms
        start_confidence = min(1.0, start_energy * 2) if start_clipped else 1.0 - start_energy

        # Analyze end - check decay time
        decay_frame = None
        for i in range(len(rms_norm) - 1, -1, -1):
            if rms_norm[i] >= peak_threshold:
                decay_frame = i
                break

        if decay_frame is not None:
            decay_time_ms = ((len(rms_norm) - 1 - decay_frame) * hop_length / sr) * 1000
        else:
            decay_time_ms = duration_ms

        # Check if audio ends abruptly
        end_energy = rms_norm[-1] if len(rms_norm) > 0 else 0
        end_clipped = end_energy > 0.2 and decay_time_ms < decay_threshold_ms
        end_confidence = min(1.0, end_energy * 2) if end_clipped else 1.0 - end_energy

        suggestions = []
        if start_clipped:
            suggestions.append(
                "Add padding before audio start or check if initial sounds are missing"
            )
        if end_clipped:
            suggestions.append("Add padding after audio end or check if final sounds are cut off")

        return {
            "status": "success",
            "is_truncated": start_clipped or end_clipped,
            "start_clipped": start_clipped,
            "end_clipped": end_clipped,
            "attack_time_ms": round(attack_time_ms, 2),
            "decay_time_ms": round(decay_time_ms, 2),
            "start_confidence": round(start_confidence, 3),
            "end_confidence": round(end_confidence, 3),
            "start_energy": round(start_energy, 3),
            "end_energy": round(end_energy, 3),
            "suggestions": suggestions,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def verify_segment_boundaries(
    audio_path: str,
    expected_text: str,
    engine: str = "faster_whisper",
    model_size: str = "base",
) -> dict:
    """Transcribe audio and verify it starts/ends cleanly with expected words.

    Checks if the transcription matches the expected text at the boundaries,
    which can indicate whether the audio was clipped. For example, if expected
    text is "Today is Monday" but transcription is "oday is Monday", the start
    is likely clipped.

    Args:
        audio_path: Path to the audio file.
        expected_text: The text the audio should contain.
        engine: Transcription engine (default: "faster_whisper").
        model_size: Whisper model size (default: "base").

    Returns:
        Dict with:
        - status: "success" or "error"
        - boundaries_clean: True if both start and end match expected
        - start_matches: True if first word matches
        - end_matches: True if last word matches
        - expected_first_word: Expected first word
        - transcribed_first_word: Actual first word
        - expected_last_word: Expected last word
        - transcribed_last_word: Actual last word
        - full_transcription: Full transcribed text
        - suggestions: List of issues found

    Example:
        result = verify_segment_boundaries("segment.wav", "Today is a beautiful day")
        if not result["start_matches"]:
            print(f"Expected '{result['expected_first_word']}' but got '{result['transcribed_first_word']}'")
    """
    import re
    from ..transcription import transcribe as do_transcribe

    try:
        # Transcribe the audio
        result = do_transcribe(
            audio_path=audio_path,
            engine=engine,
            model_size=model_size,
        )

        if result.status != "success":
            return {
                "status": "error",
                "error": result.error or "Transcription failed",
            }

        # Extract words (normalize to lowercase, keep only word characters)
        def extract_words(text: str) -> list[str]:
            words = re.findall(r"\b\w+\b", text.lower())
            return words

        expected_words = extract_words(expected_text)
        transcribed_words = extract_words(result.text)

        if not expected_words or not transcribed_words:
            return {
                "status": "error",
                "error": "Could not extract words from text",
            }

        # Check first and last words
        expected_first = expected_words[0]
        transcribed_first = transcribed_words[0] if transcribed_words else ""
        expected_last = expected_words[-1]
        transcribed_last = transcribed_words[-1] if transcribed_words else ""

        # Check for partial matches (e.g., "oday" is suffix of "today")
        start_matches = expected_first == transcribed_first
        end_matches = expected_last == transcribed_last

        # Check for truncation patterns
        start_possibly_truncated = (
            not start_matches
            and len(transcribed_first) > 1
            and expected_first.endswith(transcribed_first)
        )
        end_possibly_truncated = (
            not end_matches
            and len(transcribed_last) > 1
            and expected_last.startswith(transcribed_last)
        )

        suggestions = []
        if start_possibly_truncated:
            suggestions.append(
                f"Start appears clipped: expected '{expected_first}' but got '{transcribed_first}'"
            )
        elif not start_matches:
            suggestions.append(
                f"First word mismatch: expected '{expected_first}', got '{transcribed_first}'"
            )

        if end_possibly_truncated:
            suggestions.append(
                f"End appears clipped: expected '{expected_last}' but got '{transcribed_last}'"
            )
        elif not end_matches:
            suggestions.append(
                f"Last word mismatch: expected '{expected_last}', got '{transcribed_last}'"
            )

        return {
            "status": "success",
            "boundaries_clean": start_matches and end_matches,
            "start_matches": start_matches,
            "end_matches": end_matches,
            "start_possibly_truncated": start_possibly_truncated,
            "end_possibly_truncated": end_possibly_truncated,
            "expected_first_word": expected_first,
            "transcribed_first_word": transcribed_first,
            "expected_last_word": expected_last,
            "transcribed_last_word": transcribed_last,
            "full_transcription": result.text,
            "suggestions": suggestions,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def trim_to_speech_with_padding(
    audio_path: str,
    output_path: Optional[str] = None,
    target_start_ms: Optional[float] = None,
    padding_before_ms: float = 75,
    padding_after_ms: float = 75,
    search_window_ms: float = 100,
) -> dict:
    """Trim audio to speech boundaries with intelligent padding.

    Instead of trusting timestamps exactly, finds the nearest silence→speech
    transition and adds configurable padding. This ensures clean cuts that
    don't clip speech.

    Args:
        audio_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_smart_trimmed' suffix.
        target_start_ms: Optional approximate start time. If provided, searches
            for speech onset near this point. If None, trims from audio start.
        padding_before_ms: Padding to add before detected speech start (default: 75ms).
        padding_after_ms: Padding to add after detected speech end (default: 75ms).
        search_window_ms: Window to search for speech transitions (default: 100ms).

    Returns:
        Dict with:
        - status: "success" or "error"
        - input_path: Original file path
        - output_path: Trimmed file path
        - detected_start_ms: Where speech was detected to start
        - detected_end_ms: Where speech was detected to end
        - actual_start_ms: Trim start point (with padding)
        - actual_end_ms: Trim end point (with padding)
        - original_duration_ms: Original file duration
        - trimmed_duration_ms: New file duration

    Example:
        # Smart trim with 75ms padding
        result = trim_to_speech_with_padding("speech.wav", padding_before_ms=75)
    """
    from pathlib import Path as PathLib

    try:
        import numpy as np

        try:
            import librosa
            import soundfile as sf
        except ImportError:
            return {
                "status": "error",
                "error": "librosa and soundfile required. Install with: pip install librosa soundfile",
            }

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        original_duration_ms = (len(y) / sr) * 1000

        # Calculate RMS energy
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = int(0.005 * sr)  # 5ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if len(rms) < 3:
            return {
                "status": "error",
                "error": "Audio too short for analysis",
            }

        # Normalize and find energy threshold
        rms_max = np.max(rms) if np.max(rms) > 0 else 1
        rms_norm = rms / rms_max

        # Adaptive threshold based on noise floor
        noise_floor = np.percentile(rms_norm, 10)
        threshold = max(0.1, noise_floor * 3)

        # Find speech start
        if target_start_ms is not None:
            # Search around target
            search_start_sample = max(0, int(((target_start_ms - search_window_ms) / 1000) * sr))
            search_end_sample = min(len(y), int(((target_start_ms + search_window_ms) / 1000) * sr))
            search_start_frame = search_start_sample // hop_length
            search_end_frame = min(len(rms), search_end_sample // hop_length)

            speech_start_frame = search_start_frame
            for i in range(search_start_frame, search_end_frame):
                if rms_norm[i] >= threshold:
                    speech_start_frame = i
                    break
        else:
            # Search from beginning
            speech_start_frame = 0
            for i, energy in enumerate(rms_norm):
                if energy >= threshold:
                    speech_start_frame = i
                    break

        # Find speech end (search from the end)
        speech_end_frame = len(rms) - 1
        for i in range(len(rms) - 1, -1, -1):
            if rms_norm[i] >= threshold:
                speech_end_frame = i
                break

        # Convert frames to milliseconds
        detected_start_ms = (speech_start_frame * hop_length / sr) * 1000
        detected_end_ms = (speech_end_frame * hop_length / sr) * 1000

        # Apply padding
        actual_start_ms = max(0, detected_start_ms - padding_before_ms)
        actual_end_ms = min(original_duration_ms, detected_end_ms + padding_after_ms)

        # Determine output path
        if output_path is None:
            input_path_obj = PathLib(audio_path)
            stem = input_path_obj.stem
            suffix = input_path_obj.suffix
            output_path = str(input_path_obj.with_name(f"{stem}_smart_trimmed{suffix}"))

        # Create output directory if needed
        output_path_obj = PathLib(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Trim audio
        start_sample = int((actual_start_ms / 1000) * sr)
        end_sample = int((actual_end_ms / 1000) * sr)
        trimmed = y[start_sample:end_sample]

        # Save
        sf.write(output_path, trimmed, sr)

        trimmed_duration_ms = (len(trimmed) / sr) * 1000

        return {
            "status": "success",
            "input_path": audio_path,
            "output_path": output_path,
            "detected_start_ms": round(detected_start_ms, 2),
            "detected_end_ms": round(detected_end_ms, 2),
            "actual_start_ms": round(actual_start_ms, 2),
            "actual_end_ms": round(actual_end_ms, 2),
            "original_duration_ms": round(original_duration_ms, 2),
            "trimmed_duration_ms": round(trimmed_duration_ms, 2),
            "padding_before_ms": padding_before_ms,
            "padding_after_ms": padding_after_ms,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


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
    # TTS Verification
    "TTS_TAG_FORMATS",
    "convert_tags_for_engine",
    "detect_spoken_tags",
    "compare_audio_to_text",
    # Speech Boundary Detection
    "detect_speech_onset",
    "detect_truncated_audio",
    "verify_segment_boundaries",
    "trim_to_speech_with_padding",
]
