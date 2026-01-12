"""Transcription tools - speech-to-text."""

from typing import Optional

from ...tools.transcription import transcribe
from ..config import to_dict


def register_transcription_tools(mcp):
    """Register transcription tools with the MCP server."""

    @mcp.tool()
    def transcribe_audio(
        audio_path: str,
        engine: str = "faster_whisper",
        language: Optional[str] = None,
        model_size: str = "large-v3",
    ) -> dict:
        """Transcribe audio to text.

        Args:
            audio_path: Audio file to transcribe.
            engine: whisper or faster_whisper (4x faster).
            language: Language code (auto-detected if None).
            model_size: tiny, base, small, medium, large-v3, large-v3-turbo.

        Returns text, segments with timing, detected language.
        """
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
        )
        return to_dict(result)

    @mcp.tool()
    def transcribe_with_timestamps(
        audio_path: str,
        engine: str = "faster_whisper",
        language: Optional[str] = None,
        model_size: str = "large-v3",
        word_level: bool = False,
    ) -> dict:
        """Transcribe with detailed timing info.

        Args:
            word_level: Include word-level timestamps (slower but precise).
        """
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
            word_timestamps=word_level,
        )
        return to_dict(result)

    @mcp.tool()
    def verify_tts_output(
        audio_path: str,
        expected_text: str,
        engine: str = "faster_whisper",
        model_size: str = "large-v3",
        similarity_threshold: float = 0.8,
    ) -> dict:
        """Verify TTS audio contains expected text.

        Returns verified (bool), similarity score, and transcribed text.
        """
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            model_size=model_size,
        )

        if result.status != "success":
            return to_dict(result)

        # Normalize and compare
        expected_norm = expected_text.lower().strip()
        transcribed_norm = result.text.lower().strip()

        # Simple similarity (could be improved with difflib)
        from difflib import SequenceMatcher

        similarity = SequenceMatcher(None, expected_norm, transcribed_norm).ratio()

        return {
            "status": "success",
            "verified": similarity >= similarity_threshold,
            "similarity": similarity,
            "expected_text": expected_norm,
            "transcribed_text": transcribed_norm,
        }
