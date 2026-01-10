"""AI-powered auto-tagging for audio assets.

Uses transcription, emotion detection, and audio analysis to automatically
generate semantic tags for audio assets.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Tag sources for tracking where tags came from
TAG_SOURCE_AI = "ai"
TAG_SOURCE_TRANSCRIPTION = "ai:transcription"
TAG_SOURCE_EMOTION = "ai:emotion"
TAG_SOURCE_QUALITY = "ai:quality"
TAG_SOURCE_CLASSIFICATION = "ai:classification"


@dataclass
class AutoTagResult:
    """Result of auto-tagging an audio file."""

    status: str  # "success" or "error"
    tags: list[str] = field(default_factory=list)
    tag_sources: dict[str, str] = field(default_factory=dict)  # tag -> source
    transcription: str | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None
    quality_score: float | None = None
    error: str | None = None
    processing_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "tags": self.tags,
            "tag_sources": self.tag_sources,
            "transcription": self.transcription,
            "emotion": self.emotion,
            "emotion_confidence": self.emotion_confidence,
            "quality_score": self.quality_score,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
        }


# Common stop words to filter out from transcription
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then", "once",
    "if", "because", "until", "while", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "any", "um", "uh", "like", "yeah", "okay", "ok", "well",
    "right", "know", "think", "going", "got", "get", "go", "come", "came",
    "said", "say", "says", "tell", "told", "ask", "asked", "let", "make",
    "made", "take", "took", "see", "saw", "look", "looked", "want", "wanted",
    "give", "gave", "put", "seem", "seemed", "try", "tried", "leave", "left",
    "call", "called", "keep", "kept", "still", "even", "back", "way", "being",
}

# Quality level thresholds (MOS scale 1-5)
QUALITY_THRESHOLDS = {
    "excellent-quality": 4.5,
    "good-quality": 3.5,
    "fair-quality": 2.5,
    # Below 2.5 = poor quality (no tag added to avoid negative tagging)
}

# Emotion confidence threshold for tagging
EMOTION_CONFIDENCE_THRESHOLD = 0.4


def extract_keywords_from_text(
    text: str,
    min_word_length: int = 3,
    max_keywords: int = 10,
) -> list[str]:
    """Extract meaningful keywords from transcribed text.

    Args:
        text: The transcribed text.
        min_word_length: Minimum word length to consider.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of keyword tags.
    """
    if not text:
        return []

    # Normalize text
    text = text.lower()

    # Extract words (alphanumeric only)
    words = re.findall(r"\b[a-z]+\b", text)

    # Filter stop words and short words
    words = [
        w for w in words
        if w not in STOP_WORDS and len(w) >= min_word_length
    ]

    # Count word frequency
    word_counts: dict[str, int] = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def _check_transcription_available() -> bool:
    """Check if transcription engines are available."""
    try:
        from ..transcription import get_engine
        # Try faster_whisper first (preferred), then whisper
        for engine_id in ["faster_whisper", "whisper"]:
            try:
                engine = get_engine(engine_id)
                if engine and engine.is_available():
                    return True
            except Exception:
                pass
        return False
    except ImportError:
        return False


def _check_emotion_available() -> bool:
    """Check if emotion detection is available."""
    try:
        from ..analysis.emotion2vec import Emotion2vecEngine
        engine = Emotion2vecEngine()
        return engine.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def _check_quality_available() -> bool:
    """Check if quality assessment is available."""
    try:
        from ..analysis.nisqa import NISQAEngine
        engine = NISQAEngine()
        return engine.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def auto_tag_audio(
    audio_path: str,
    use_transcription: bool = True,
    use_emotion: bool = True,
    use_quality: bool = True,
    transcription_model: str = "base",
    max_keywords: int = 8,
    progress_callback: Callable[[str], None] | None = None,
) -> AutoTagResult:
    """Auto-tag an audio file using AI analysis.

    Combines multiple analysis methods:
    1. Transcription-based: Extracts keywords from speech content
    2. Emotion detection: Tags detected emotions (happy, sad, angry, etc.)
    3. Quality assessment: Tags quality level (excellent, good, fair)

    Args:
        audio_path: Path to the audio file.
        use_transcription: Whether to use transcription for keyword extraction.
        use_emotion: Whether to detect and tag emotions.
        use_quality: Whether to assess and tag quality.
        transcription_model: Whisper model size for transcription.
        max_keywords: Maximum keywords to extract from transcription.
        progress_callback: Optional callback for progress updates.

    Returns:
        AutoTagResult with generated tags and metadata.
    """
    import time
    start_time = time.time()

    audio_path = str(Path(audio_path).resolve())
    if not Path(audio_path).exists():
        return AutoTagResult(
            status="error",
            error=f"Audio file not found: {audio_path}",
        )

    result = AutoTagResult(status="success")
    tags: list[str] = []
    tag_sources: dict[str, str] = {}

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

    # 1. Transcription-based tagging
    if use_transcription and _check_transcription_available():
        log("Transcribing audio for keyword extraction...")
        try:
            from ..transcription import transcribe

            transcription_result = transcribe(
                audio_path,
                engine="faster_whisper",
                model_size=transcription_model,
            )

            if transcription_result.status == "success" and transcription_result.text:
                result.transcription = transcription_result.text
                keywords = extract_keywords_from_text(
                    transcription_result.text,
                    max_keywords=max_keywords,
                )
                for kw in keywords:
                    if kw not in tags:
                        tags.append(kw)
                        tag_sources[kw] = TAG_SOURCE_TRANSCRIPTION
                log(f"Extracted {len(keywords)} keywords from transcription")

                # Add "speech" tag if transcription succeeded
                if "speech" not in tags:
                    tags.append("speech")
                    tag_sources["speech"] = TAG_SOURCE_CLASSIFICATION

        except Exception as e:
            log(f"Transcription failed: {e}")

    # 2. Emotion detection
    if use_emotion and _check_emotion_available():
        log("Detecting emotion...")
        try:
            from ..analysis import detect_emotion

            emotion_result = detect_emotion(audio_path)

            if emotion_result.status == "success":
                result.emotion = emotion_result.primary_emotion
                result.emotion_confidence = emotion_result.primary_score

                # Only tag if confidence is above threshold
                if (
                    emotion_result.primary_score
                    and emotion_result.primary_score >= EMOTION_CONFIDENCE_THRESHOLD
                    and emotion_result.primary_emotion
                    and emotion_result.primary_emotion.lower() not in ("neutral", "unknown", "other")
                ):
                    emotion_tag = emotion_result.primary_emotion.lower()
                    if emotion_tag not in tags:
                        tags.append(emotion_tag)
                        tag_sources[emotion_tag] = TAG_SOURCE_EMOTION
                    log(f"Detected emotion: {emotion_tag} ({emotion_result.primary_score:.1%})")

        except Exception as e:
            log(f"Emotion detection failed: {e}")

    # 3. Quality assessment
    if use_quality and _check_quality_available():
        log("Assessing audio quality...")
        try:
            from ..analysis import assess_quality

            quality_result = assess_quality(audio_path)

            if quality_result.status == "success":
                result.quality_score = quality_result.overall_quality

                # Add quality tag based on MOS score
                for tag, threshold in QUALITY_THRESHOLDS.items():
                    if quality_result.overall_quality >= threshold:
                        if tag not in tags:
                            tags.append(tag)
                            tag_sources[tag] = TAG_SOURCE_QUALITY
                        log(f"Quality: {quality_result.overall_quality:.2f} -> {tag}")
                        break

        except Exception as e:
            log(f"Quality assessment failed: {e}")

    result.tags = tags
    result.tag_sources = tag_sources
    result.processing_time_ms = int((time.time() - start_time) * 1000)

    return result


def get_autotag_capabilities() -> dict:
    """Get information about available auto-tagging capabilities.

    Returns:
        Dict with available engines and their status.
    """
    return {
        "transcription": {
            "available": _check_transcription_available(),
            "description": "Extract keywords from speech content",
            "tags_generated": ["speech", "content keywords"],
        },
        "emotion": {
            "available": _check_emotion_available(),
            "description": "Detect emotional tone",
            "tags_generated": ["happy", "sad", "angry", "fearful", "surprised", "disgusted"],
        },
        "quality": {
            "available": _check_quality_available(),
            "description": "Assess audio quality",
            "tags_generated": ["excellent-quality", "good-quality", "fair-quality"],
        },
    }
