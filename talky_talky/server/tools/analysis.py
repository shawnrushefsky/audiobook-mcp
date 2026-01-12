"""Audio analysis tools - emotion, voice similarity, quality, and SFX analysis."""

from typing import Optional

from ...tools.analysis import (
    detect_emotion,
    compare_voices,
    get_voice_embedding,
    assess_quality,
    to_dict as analysis_to_dict,
    analyze_loudness,
    detect_clipping,
    analyze_spectrum,
    detect_silence,
    validate_format,
    detect_spoken_tags,
    convert_tags_for_engine,
    detect_speech_onset,
    detect_truncated_audio,
    verify_segment_boundaries,
    trim_to_speech_with_padding,
)
from ...tools.transcription import transcribe


def register_analysis_tools(mcp):
    """Register analysis tools with the MCP server."""

    # ========== Emotion & Voice Analysis ==========

    @mcp.tool()
    def analyze_emotion(
        audio_path: str,
        engine: str = "emotion2vec",
    ) -> dict:
        """Detect emotion in speech.

        Returns primary_emotion, confidence, and all emotion scores.
        Emotions: angry, disgusted, fearful, happy, neutral, sad, surprised.
        """
        result = detect_emotion(audio_path, engine)
        return analysis_to_dict(result)

    @mcp.tool()
    def analyze_voice_similarity(
        audio_path_1: str,
        audio_path_2: str,
        threshold: float = 0.75,
        engine: str = "resemblyzer",
    ) -> dict:
        """Compare two audio files for voice similarity.

        Returns similarity_score (0-1) and is_same_speaker (bool).
        Threshold: 0.75+ likely same, 0.60-0.75 possibly same.
        """
        result = compare_voices(audio_path_1, audio_path_2, threshold, engine)
        return analysis_to_dict(result)

    @mcp.tool()
    def extract_voice_embedding(
        audio_path: str,
        engine: str = "resemblyzer",
    ) -> dict:
        """Extract 256-dim voice embedding vector for comparison."""
        result = get_voice_embedding(audio_path, engine)
        return analysis_to_dict(result)

    @mcp.tool()
    def analyze_speech_quality(
        audio_path: str,
        engine: str = "nisqa",
    ) -> dict:
        """Assess speech quality (MOS 1-5 scale).

        Returns overall_quality plus dimensions: noisiness, discontinuity,
        coloration, loudness.
        """
        result = assess_quality(audio_path, engine)
        return analysis_to_dict(result)

    # ========== TTS Verification ==========

    @mcp.tool()
    def verify_tts_comprehensive(
        generated_audio_path: str,
        expected_text: str,
        reference_audio_path: Optional[str] = None,
        expected_emotion: Optional[str] = None,
        min_quality_score: float = 3.0,
        min_similarity_score: float = 0.70,
        min_text_similarity: float = 0.8,
    ) -> dict:
        """Comprehensive TTS verification: text, voice, emotion, quality.

        Returns overall_pass and detailed check results with recommendations.
        """
        checks = {}
        recommendations = []
        overall_pass = True

        # Text accuracy check
        transcription = transcribe(generated_audio_path, engine="faster_whisper")
        if transcription.status == "success":
            from difflib import SequenceMatcher

            expected_norm = expected_text.lower().strip()
            transcribed_norm = transcription.text.lower().strip()
            similarity = SequenceMatcher(None, expected_norm, transcribed_norm).ratio()
            text_pass = similarity >= min_text_similarity
            checks["text_accuracy"] = {
                "pass": text_pass,
                "similarity": similarity,
                "expected": expected_norm,
                "transcribed": transcribed_norm,
            }
            if not text_pass:
                overall_pass = False
                recommendations.append("Text mismatch - check TTS output")

        # Voice similarity check
        if reference_audio_path:
            voice_result = compare_voices(
                reference_audio_path, generated_audio_path, min_similarity_score
            )
            voice_pass = voice_result.similarity_score >= min_similarity_score
            checks["voice_similarity"] = {
                "pass": voice_pass,
                "score": voice_result.similarity_score,
            }
            if not voice_pass:
                overall_pass = False
                recommendations.append("Voice mismatch - try different reference audio")

        # Emotion check
        if expected_emotion:
            emotion_result = detect_emotion(generated_audio_path)
            emotion_pass = emotion_result.primary_emotion.lower() == expected_emotion.lower()
            checks["emotion_match"] = {
                "pass": emotion_pass,
                "detected": emotion_result.primary_emotion,
                "expected": expected_emotion,
            }
            if not emotion_pass:
                overall_pass = False
                recommendations.append("Emotion mismatch - adjust emotion tags")

        # Quality check
        quality_result = assess_quality(generated_audio_path)
        quality_pass = quality_result.overall_quality >= min_quality_score
        checks["speech_quality"] = {
            "pass": quality_pass,
            "score": quality_result.overall_quality,
        }
        if not quality_pass:
            overall_pass = False
            recommendations.append("Low quality - try different engine or settings")

        return {
            "status": "success",
            "overall_pass": overall_pass,
            "checks": checks,
            "recommendations": recommendations,
        }

    @mcp.tool()
    def detect_spoken_tts_tags(
        audio_path: str,
        model_size: str = "base",
    ) -> dict:
        """Detect if TTS verbalized emotion tags instead of expressing them.

        Returns detected spoken tags and whether audio is clean.
        """
        return detect_spoken_tags(audio_path, model_size)

    @mcp.tool()
    def convert_emotion_tags(
        text: str,
        source_engine: str,
        target_engine: str,
    ) -> dict:
        """Convert emotion tags between TTS engine formats.

        E.g., Maya1 <laugh> to Chatterbox [laugh].
        """
        return convert_tags_for_engine(text, source_engine, target_engine)

    # ========== Speech Boundary Detection ==========

    @mcp.tool()
    def find_speech_onset(
        audio_path: str,
        silence_threshold_db: float = -40.0,
        min_speech_ms: float = 50.0,
    ) -> dict:
        """Find where speech actually begins in audio."""
        return detect_speech_onset(audio_path, silence_threshold_db, min_speech_ms)

    @mcp.tool()
    def detect_audio_truncation(
        audio_path: str,
        expected_text: str,
        min_expected_duration_ms: Optional[float] = None,
    ) -> dict:
        """Check if TTS output was truncated mid-sentence."""
        return detect_truncated_audio(audio_path, expected_text, min_expected_duration_ms)

    @mcp.tool()
    def verify_audio_segment_boundaries(
        audio_path: str,
        expected_text: str,
        max_leading_silence_ms: float = 200.0,
        max_trailing_silence_ms: float = 500.0,
    ) -> dict:
        """Verify audio has proper speech boundaries."""
        return verify_segment_boundaries(
            audio_path, expected_text, max_leading_silence_ms, max_trailing_silence_ms
        )

    @mcp.tool()
    def smart_trim_to_speech(
        input_path: str,
        output_path: Optional[str] = None,
        padding_before_ms: float = 50.0,
        padding_after_ms: float = 100.0,
    ) -> dict:
        """Trim audio to speech boundaries with natural padding."""
        return trim_to_speech_with_padding(
            input_path, output_path, padding_before_ms, padding_after_ms
        )

    # ========== SFX Analysis ==========

    @mcp.tool()
    def analyze_audio_loudness(audio_path: str) -> dict:
        """Analyze loudness: peak, RMS, LUFS, dynamic range, true peak."""
        return analyze_loudness(audio_path)

    @mcp.tool()
    def detect_audio_clipping(
        audio_path: str,
        threshold: float = 0.99,
        min_consecutive: int = 2,
    ) -> dict:
        """Detect digital clipping (distortion)."""
        return detect_clipping(audio_path, threshold, min_consecutive)

    @mcp.tool()
    def analyze_audio_spectrum(audio_path: str) -> dict:
        """Analyze frequency content: dominant freq, centroid, bandwidth, energy."""
        return analyze_spectrum(audio_path)

    @mcp.tool()
    def detect_audio_silence(
        audio_path: str,
        threshold_db: float = -40.0,
        min_silence_ms: float = 100.0,
    ) -> dict:
        """Detect silence regions in audio."""
        return detect_silence(audio_path, threshold_db, min_silence_ms)

    @mcp.tool()
    def validate_audio_format(
        audio_path: str,
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None,
        target_bit_depth: Optional[int] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        max_file_size_bytes: Optional[int] = None,
    ) -> dict:
        """Validate audio format against requirements."""
        return validate_format(
            audio_path,
            target_sample_rate,
            target_channels,
            target_bit_depth,
            min_duration_ms,
            max_duration_ms,
            max_file_size_bytes,
        )
