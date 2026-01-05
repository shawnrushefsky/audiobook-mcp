"""Tool implementations for Talky Talky TTS MCP server."""

from .audio import (
    get_audio_info,
    convert_audio,
    concatenate_audio,
    normalize_audio,
    is_ffmpeg_available,
)
from .tts import (
    check_tts,
    get_tts_info,
    generate,
    get_engine,
    get_available_engines,
    list_engines,
)

__all__ = [
    # Audio tools
    "get_audio_info",
    "convert_audio",
    "concatenate_audio",
    "normalize_audio",
    "is_ffmpeg_available",
    # TTS tools
    "check_tts",
    "get_tts_info",
    "generate",
    "get_engine",
    "get_available_engines",
    "list_engines",
]
