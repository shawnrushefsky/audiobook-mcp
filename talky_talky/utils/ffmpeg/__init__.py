"""FFmpeg wrapper functions for audio processing.

This package provides a clean interface to ffmpeg for common audio operations:
- Format conversion and validation
- Audio concatenation with gaps and crossfades
- Audio manipulation (trimming, effects, mixing)
"""

from .core import (
    AudioProperties,
    AudioValidation,
    ChapterMarker,
    check_ffmpeg,
    get_audio_duration,
    get_audio_properties,
    resample_audio,
    validate_audio_file,
)
from .format import (
    convert_audio_format,
    normalize_audio,
)
from .concat import (
    concatenate_audio_files,
    concatenate_with_gaps,
    concatenate_with_variable_gaps,
    create_audiobook_with_chapters,
    crossfade_audio_files,
    generate_silence,
)
from .manipulation import (
    adjust_audio_volume,
    apply_audio_effects,
    apply_audio_fade,
    insert_silence_file,
    mix_audio_tracks,
    overlay_audio_at_position,
    trim_audio_file,
)

__all__ = [
    # Data classes
    "AudioProperties",
    "AudioValidation",
    "ChapterMarker",
    # Core utilities
    "check_ffmpeg",
    "get_audio_duration",
    "get_audio_properties",
    "resample_audio",
    "validate_audio_file",
    # Format conversion
    "convert_audio_format",
    "normalize_audio",
    # Concatenation
    "concatenate_audio_files",
    "concatenate_with_gaps",
    "concatenate_with_variable_gaps",
    "create_audiobook_with_chapters",
    "crossfade_audio_files",
    "generate_silence",
    # Manipulation
    "adjust_audio_volume",
    "apply_audio_effects",
    "apply_audio_fade",
    "insert_silence_file",
    "mix_audio_tracks",
    "overlay_audio_at_position",
    "trim_audio_file",
]
