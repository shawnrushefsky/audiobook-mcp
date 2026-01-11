"""Audio utility tools for format conversion, concatenation, and info.

These are standalone utilities with no project/database dependencies.
Agents can use these to manipulate audio files directly by path.
"""

from .types import (
    AudioInfo,
    ConvertResult,
    ConcatenateResult,
    NormalizeResult,
    TrimResult,
    InsertSilenceResult,
    CrossfadeResult,
    MixResult,
    VolumeResult,
    FadeResult,
    EffectsResult,
    OverlayResult,
    PitchShiftResult,
    TimeStretchResult,
    VoiceEffectResult,
    FormantShiftResult,
    ResampleResult,
    AudioCompatibilityResult,
)
from .utilities import (
    get_audio_info,
    convert_audio,
    concatenate_audio,
    normalize_audio,
    resample_audio,
    validate_audio_compatibility,
    is_ffmpeg_available,
)
from .trimming import (
    trim_audio,
    batch_detect_silence,
    insert_silence,
    crossfade_join,
)
from .design import (
    mix_audio,
    adjust_volume,
    apply_fade,
    apply_effects,
    overlay_audio,
)
from .modulation import (
    VOICE_EFFECTS,
    shift_pitch,
    stretch_time,
    apply_voice_effect,
    shift_formant,
)

__all__ = [
    # Data classes
    "AudioInfo",
    "ConvertResult",
    "ConcatenateResult",
    "NormalizeResult",
    "TrimResult",
    "InsertSilenceResult",
    "CrossfadeResult",
    "MixResult",
    "VolumeResult",
    "FadeResult",
    "EffectsResult",
    "OverlayResult",
    "PitchShiftResult",
    "TimeStretchResult",
    "VoiceEffectResult",
    "FormantShiftResult",
    "ResampleResult",
    "AudioCompatibilityResult",
    # Audio utilities
    "get_audio_info",
    "convert_audio",
    "concatenate_audio",
    "normalize_audio",
    "resample_audio",
    "validate_audio_compatibility",
    "is_ffmpeg_available",
    # Trimming utilities
    "trim_audio",
    "batch_detect_silence",
    "insert_silence",
    "crossfade_join",
    # Audio design tools
    "mix_audio",
    "adjust_volume",
    "apply_fade",
    "apply_effects",
    "overlay_audio",
    # Voice modulation
    "VOICE_EFFECTS",
    "shift_pitch",
    "stretch_time",
    "apply_voice_effect",
    "shift_formant",
]
