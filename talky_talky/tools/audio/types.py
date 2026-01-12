"""Data classes for audio tool results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioInfo:
    """Information about an audio file."""

    path: str
    exists: bool
    format: Optional[str] = None
    duration_ms: Optional[int] = None
    size_bytes: Optional[int] = None
    valid: bool = False
    error: Optional[str] = None


@dataclass
class ConvertResult:
    """Result of an audio conversion operation."""

    input_path: str
    output_path: str
    input_format: str
    output_format: str
    input_size_bytes: int
    output_size_bytes: int
    compression_ratio: float
    duration_ms: int


@dataclass
class ConcatenateResult:
    """Result of audio concatenation."""

    output_path: str
    input_count: int
    total_duration_ms: int
    output_format: str


@dataclass
class NormalizeResult:
    """Result of audio normalization."""

    input_path: str
    output_path: str
    duration_ms: int


@dataclass
class TrimResult:
    """Result of audio trimming operation."""

    input_path: str
    output_path: str
    original_duration_ms: int
    trimmed_duration_ms: int
    start_ms: float
    end_ms: float
    silence_removed_ms: float
    auto_detected: bool = False


@dataclass
class InsertSilenceResult:
    """Result of inserting silence into audio."""

    input_path: str
    output_path: str
    original_duration_ms: int
    new_duration_ms: int
    silence_before_ms: float
    silence_after_ms: float


@dataclass
class CrossfadeResult:
    """Result of crossfade concatenation."""

    output_path: str
    input_count: int
    total_duration_ms: int
    crossfade_ms: float
    output_format: str


@dataclass
class MixResult:
    """Result of mixing/layering audio tracks."""

    output_path: str
    input_count: int
    duration_ms: int
    normalized: bool


@dataclass
class VolumeResult:
    """Result of volume adjustment."""

    input_path: str
    output_path: str
    duration_ms: int
    volume_change: str  # e.g., "2.0x" or "+6dB"


@dataclass
class FadeResult:
    """Result of applying fade in/out."""

    input_path: str
    output_path: str
    duration_ms: int
    fade_in_ms: float
    fade_out_ms: float


@dataclass
class EffectsResult:
    """Result of applying audio effects."""

    input_path: str
    output_path: str
    duration_ms: int
    effects_applied: list[str]


@dataclass
class OverlayResult:
    """Result of overlaying audio."""

    base_path: str
    overlay_path: str
    output_path: str
    duration_ms: int
    overlay_position_ms: float
    overlay_volume: float


@dataclass
class PitchShiftResult:
    """Result of pitch shifting operation."""

    input_path: str
    output_path: str
    duration_ms: int
    semitones: float
    sample_rate: int


@dataclass
class TimeStretchResult:
    """Result of time stretching operation."""

    input_path: str
    output_path: str
    original_duration_ms: int
    new_duration_ms: int
    rate: float
    sample_rate: int


@dataclass
class VoiceEffectResult:
    """Result of applying a voice effect."""

    input_path: str
    output_path: str
    duration_ms: int
    effect: str
    intensity: float


@dataclass
class FormantShiftResult:
    """Result of formant shifting operation."""

    input_path: str
    output_path: str
    duration_ms: int
    shift_ratio: float
    sample_rate: int


@dataclass
class ResampleResult:
    """Result of audio resampling operation."""

    input_path: str
    output_path: str
    original_sample_rate: int
    new_sample_rate: int
    duration_ms: int


@dataclass
class AudioCompatibilityResult:
    """Result of audio compatibility validation."""

    compatible: bool
    files_checked: int
    issues: list[str]
    sample_rates: dict[str, int]
    channels: dict[str, int]
    recommendation: Optional[str] = None


@dataclass
class LufsNormalizeResult:
    """Result of LUFS normalization."""

    input_path: str
    output_path: str
    duration_ms: int
    target_lufs: float
    input_lufs: Optional[float] = None
    output_lufs: Optional[float] = None


@dataclass
class MeanLevelResult:
    """Result of mean level analysis."""

    path: str
    mean_db: float
    peak_db: float
    rms_db: float
    duration_ms: int


@dataclass
class MultiOverlayResult:
    """Result of overlaying multiple audio tracks."""

    base_path: str
    output_path: str
    duration_ms: int
    overlay_count: int
    overlays_applied: list[dict]


@dataclass
class LevelComparisonResult:
    """Result of comparing audio levels between two files."""

    path1: str
    path2: str
    path1_mean_db: float
    path2_mean_db: float
    difference_db: float
    louder_file: str
    audibility_prediction: str
