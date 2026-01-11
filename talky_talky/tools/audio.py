"""Audio utility tools for format conversion, concatenation, and info.

These are standalone utilities with no project/database dependencies.
Agents can use these to manipulate audio files directly by path.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..utils.ffmpeg import (
    check_ffmpeg,
    validate_audio_file,
    get_audio_duration,
    get_audio_properties,
    resample_audio as _resample_audio,
    concatenate_audio_files as _concat_files,
    concatenate_with_gaps as _concat_with_gaps,
    concatenate_with_variable_gaps as _concat_with_variable_gaps,
    convert_audio_format as _convert_format,
    normalize_audio as _normalize,
    trim_audio_file as _trim_audio,
    insert_silence_file as _insert_silence,
    crossfade_audio_files as _crossfade,
    mix_audio_tracks as _mix_tracks,
    adjust_audio_volume as _adjust_volume,
    apply_audio_fade as _apply_fade,
    apply_audio_effects as _apply_effects,
    overlay_audio_at_position as _overlay_audio,
)


# ============================================================================
# Data Classes
# ============================================================================


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


# ============================================================================
# Audio Utilities
# ============================================================================


def get_audio_info(audio_path: str) -> AudioInfo:
    """Get information about an audio file.

    Args:
        audio_path: Path to the audio file.

    Returns:
        AudioInfo with duration, format, size, and validity.
    """
    path = Path(audio_path)

    if not path.exists():
        return AudioInfo(
            path=audio_path,
            exists=False,
            error="File not found",
        )

    # Check if ffprobe is available
    if not check_ffmpeg():
        return AudioInfo(
            path=audio_path,
            exists=True,
            size_bytes=path.stat().st_size,
            valid=False,
            error="ffprobe not available - install ffmpeg to get audio metadata",
        )

    # Validate and get info
    validation = validate_audio_file(audio_path)

    if not validation.valid:
        return AudioInfo(
            path=audio_path,
            exists=True,
            valid=False,
            error=validation.error,
        )

    return AudioInfo(
        path=audio_path,
        exists=True,
        format=validation.format,
        duration_ms=validation.duration_ms,
        size_bytes=path.stat().st_size,
        valid=True,
    )


def convert_audio(
    input_path: str,
    output_format: str = "mp3",
    output_path: Optional[str] = None,
) -> ConvertResult:
    """Convert an audio file to a different format.

    Args:
        input_path: Path to the input audio file.
        output_format: Target format ('mp3', 'wav', 'm4a'). Default: 'mp3'.
        output_path: Optional output path. If not provided, creates a file
            with the same name but new extension in the same directory.

    Returns:
        ConvertResult with paths and size comparison.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
        ValueError: If output format is unsupported.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if output_format not in ("mp3", "wav", "m4a"):
        raise ValueError(f"Unsupported format: {output_format}. Use 'mp3', 'wav', or 'm4a'.")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        output_path = str(input_path_obj.with_suffix(f".{output_format}"))

    output_path_obj = Path(output_path)

    # Get input info
    input_format = input_path_obj.suffix.lstrip(".")
    input_size = input_path_obj.stat().st_size

    # Convert
    _convert_format(input_path, output_path, output_format)

    # Get output info
    output_size = output_path_obj.stat().st_size
    duration = get_audio_duration(output_path)

    return ConvertResult(
        input_path=input_path,
        output_path=output_path,
        input_format=input_format,
        output_format=output_format,
        input_size_bytes=input_size,
        output_size_bytes=output_size,
        compression_ratio=round(input_size / output_size, 2) if output_size else 0,
        duration_ms=duration,
    )


def concatenate_audio(
    audio_paths: list[str],
    output_path: str,
    output_format: str = "wav",
    gap_ms: float | list[float] = 0,
    resample: bool = False,
    target_sample_rate: Optional[int] = None,
) -> ConcatenateResult:
    """Concatenate multiple audio files into one.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'wav'.
        gap_ms: Silence between segments. Can be:
            - A single float: uniform gap between all segments (e.g., 300)
            - A list of floats: variable gaps (e.g., [300, 300, 800, 300])
              Length must be len(audio_paths) - 1
            Default: 0 (no gaps).
            Typical values: 300-400ms between dialogue, 800ms+ for scene breaks.
        resample: If True, resample all files to a common sample rate before joining.
            Uses target_sample_rate if provided, otherwise uses the first file's rate.
        target_sample_rate: Target sample rate when resample=True. If None, uses
            the sample rate of the first file.

    Returns:
        ConcatenateResult with output info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If no audio paths provided, format unsupported, or sample rate mismatch
            detected without resample=True.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if not audio_paths:
        raise ValueError("No audio paths provided")

    if output_format not in ("mp3", "wav", "m4a"):
        raise ValueError(f"Unsupported format: {output_format}. Use 'mp3', 'wav', or 'm4a'.")

    # Validate all input files exist
    for path in audio_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    # Check for sample rate mismatches
    sample_rates = {}
    for path in audio_paths:
        props = get_audio_properties(path)
        sample_rates[path] = props.sample_rate

    unique_rates = set(sample_rates.values())
    if len(unique_rates) > 1:
        if resample:
            # Resample to target rate (or first file's rate)
            if target_sample_rate is None:
                target_sample_rate = sample_rates[audio_paths[0]]

            import tempfile

            temp_dir = tempfile.mkdtemp()
            resampled_paths = []

            for i, path in enumerate(audio_paths):
                if sample_rates[path] != target_sample_rate:
                    temp_path = os.path.join(temp_dir, f"resampled_{i}.wav")
                    _resample_audio(path, temp_path, target_sample_rate)
                    resampled_paths.append(temp_path)
                else:
                    resampled_paths.append(path)

            # Continue with resampled files
            audio_paths = resampled_paths
        else:
            # Report mismatch error
            rate_info = ", ".join(f"{os.path.basename(p)}: {r}Hz" for p, r in sample_rates.items())
            raise ValueError(
                f"Sample rate mismatch detected: {rate_info}. "
                f"Use resample=True to auto-convert, or resample files manually."
            )

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Handle variable gaps
    if isinstance(gap_ms, list):
        if len(gap_ms) != len(audio_paths) - 1:
            raise ValueError(
                f"Gap list length ({len(gap_ms)}) must be one less than "
                f"audio paths count ({len(audio_paths)})"
            )
        # Use variable gaps - need to concatenate manually with different gaps
        _concat_with_variable_gaps(audio_paths, output_path, gap_ms, output_format)
    elif gap_ms > 0:
        _concat_with_gaps(audio_paths, output_path, gap_ms, output_format)
    else:
        _concat_files(audio_paths, output_path, output_format)

    # Get output info
    duration = get_audio_duration(output_path)

    return ConcatenateResult(
        output_path=output_path,
        input_count=len(audio_paths),
        total_duration_ms=duration,
        output_format=output_format,
    )


def normalize_audio(
    input_path: str,
    output_path: Optional[str] = None,
) -> NormalizeResult:
    """Normalize audio levels to broadcast standard (-16 LUFS).

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_normalized' suffix.

    Returns:
        NormalizeResult with input_path, output_path, and duration_ms.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_normalized{suffix}"))

    # Normalize
    _normalize(input_path, output_path)

    duration = get_audio_duration(output_path)

    return NormalizeResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
    )


def resample_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_sample_rate: int = 44100,
) -> ResampleResult:
    """Resample audio to a target sample rate.

    Use this to convert between sample rates without other processing.
    Essential for ensuring compatibility when joining audio files from
    different sources (e.g., TTS output at 24kHz with effects at 48kHz).

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_resampled' suffix.
        target_sample_rate: Target sample rate in Hz. Common values:
            - 24000: Common TTS output rate (Chatterbox, Maya1)
            - 44100: CD quality, general audio
            - 48000: Professional video/broadcast

    Returns:
        ResampleResult with paths and sample rate info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
        ValueError: If target_sample_rate is invalid.

    Example:
        # Resample voice effect output to match TTS output
        resample_audio("effect_192k.wav", "effect_24k.wav", target_sample_rate=24000)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get original sample rate
    props = get_audio_properties(input_path)
    original_rate = props.sample_rate

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_resampled{suffix}"))

    # Resample
    _resample_audio(input_path, output_path, target_sample_rate)

    duration = get_audio_duration(output_path)

    return ResampleResult(
        input_path=input_path,
        output_path=output_path,
        original_sample_rate=original_rate,
        new_sample_rate=target_sample_rate,
        duration_ms=duration,
    )


def validate_audio_compatibility(
    audio_paths: list[str],
) -> AudioCompatibilityResult:
    """Check if multiple audio files are compatible for joining.

    Validates sample rates and channel counts to ensure files can be
    concatenated without artifacts or corruption.

    Args:
        audio_paths: List of audio file paths to check.

    Returns:
        AudioCompatibilityResult with:
        - compatible: True if all files are compatible
        - issues: List of compatibility issues found
        - sample_rates: Dict mapping each file to its sample rate
        - channels: Dict mapping each file to its channel count
        - recommendation: Suggested fix if incompatible

    Example:
        result = validate_audio_compatibility(["a.wav", "b.wav", "c.wav"])
        if not result.compatible:
            print("Issues:", result.issues)
            print("Fix:", result.recommendation)
    """
    if not audio_paths:
        return AudioCompatibilityResult(
            compatible=True,
            files_checked=0,
            issues=[],
            sample_rates={},
            channels={},
        )

    issues = []
    sample_rates = {}
    channels = {}

    for path in audio_paths:
        if not os.path.exists(path):
            issues.append(f"File not found: {path}")
            continue

        try:
            props = get_audio_properties(path)
            sample_rates[path] = props.sample_rate
            channels[path] = props.channels
        except Exception as e:
            issues.append(f"Could not read {path}: {e}")

    # Check for sample rate mismatches
    unique_rates = set(sample_rates.values())
    if len(unique_rates) > 1:
        rate_info = ", ".join(f"{os.path.basename(p)}: {r}Hz" for p, r in sample_rates.items())
        issues.append(f"Sample rate mismatch: {rate_info}")

    # Check for channel count mismatches
    unique_channels = set(channels.values())
    if len(unique_channels) > 1:
        ch_info = ", ".join(f"{os.path.basename(p)}: {c}ch" for p, c in channels.items())
        issues.append(f"Channel count mismatch: {ch_info}")

    # Build recommendation
    recommendation = None
    if issues:
        if len(unique_rates) > 1:
            # Find the most common sample rate
            rate_counts = {}
            for rate in sample_rates.values():
                rate_counts[rate] = rate_counts.get(rate, 0) + 1
            most_common_rate = max(rate_counts, key=rate_counts.get)

            mismatched = [
                os.path.basename(p) for p, r in sample_rates.items() if r != most_common_rate
            ]
            recommendation = (
                f"Resample the following files to {most_common_rate}Hz: {', '.join(mismatched)}. "
                f"Use resample_audio() or set resample=True in join_audio_files()."
            )

    return AudioCompatibilityResult(
        compatible=len(issues) == 0,
        files_checked=len(audio_paths),
        issues=issues,
        sample_rates=sample_rates,
        channels=channels,
        recommendation=recommendation,
    )


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and available.

    Returns:
        True if ffmpeg is available, False otherwise.
    """
    return check_ffmpeg()


def trim_audio(
    input_path: str,
    output_path: Optional[str] = None,
    start_ms: Optional[float] = None,
    end_ms: Optional[float] = None,
    padding_ms: float = 50,
) -> TrimResult:
    """Trim audio to specified boundaries or auto-detect content boundaries.

    If start_ms and end_ms are not provided, uses silence detection to
    automatically find content boundaries and trims to those.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_trimmed' suffix.
        start_ms: Start time in milliseconds (None = auto-detect from silence).
        end_ms: End time in milliseconds (None = auto-detect from silence).
        padding_ms: Milliseconds of silence to keep as buffer when auto-detecting.
            Default: 50ms. Set to 0 for tight trim.

    Returns:
        TrimResult with input/output paths and duration info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_trimmed{suffix}"))

    # Get original duration
    original_duration = get_audio_duration(input_path)

    # Auto-detect mode if start_ms or end_ms not provided
    auto_detected = start_ms is None or end_ms is None

    if auto_detected:
        # Import silence detection from analysis module
        from .analysis import detect_silence as _detect_silence

        silence_result = _detect_silence(input_path)

        if silence_result.status != "success":
            raise RuntimeError(f"Silence detection failed: {silence_result.error}")

        # Use content boundaries with padding
        if start_ms is None:
            start_ms = max(0, silence_result.content_start_ms - padding_ms)
        if end_ms is None:
            end_ms = min(original_duration, silence_result.content_end_ms + padding_ms)

    # Ensure start and end are valid
    start_ms = max(0, start_ms)
    end_ms = min(original_duration, end_ms)

    if start_ms >= end_ms:
        raise ValueError(f"Invalid trim range: start_ms ({start_ms}) >= end_ms ({end_ms})")

    # Perform trim
    _trim_audio(input_path, output_path, start_ms, end_ms)

    # Get new duration
    trimmed_duration = get_audio_duration(output_path)

    return TrimResult(
        input_path=input_path,
        output_path=output_path,
        original_duration_ms=original_duration,
        trimmed_duration_ms=trimmed_duration,
        start_ms=start_ms,
        end_ms=end_ms,
        silence_removed_ms=original_duration - trimmed_duration,
        auto_detected=auto_detected,
    )


def batch_detect_silence(
    audio_paths: list[str],
    threshold_db: float = -40.0,
    min_silence_ms: float = 100.0,
) -> list[dict]:
    """Detect silence in multiple audio files at once.

    More efficient than calling detect_audio_silence individually for
    large batches of files.

    Args:
        audio_paths: List of paths to audio files to analyze.
        threshold_db: dB threshold below which audio is silent (default -40).
        min_silence_ms: Minimum duration to count as silence (default 100ms).

    Returns:
        List of dicts with silence analysis for each file. Each dict has:
        - path: The input file path
        - status: "success" or "error"
        - leading_silence_ms: Silence at start
        - trailing_silence_ms: Silence at end
        - content_start_ms: Where actual content begins
        - content_end_ms: Where actual content ends
        - content_duration_ms: Duration of non-silent content
        - error: Error message if status is "error"
    """
    from .analysis import detect_silence as _detect_silence

    results = []

    for path in audio_paths:
        result = _detect_silence(path, threshold_db=threshold_db, min_silence_ms=min_silence_ms)

        result_dict = {
            "path": path,
            "status": result.status,
        }

        if result.status == "success":
            result_dict.update(
                {
                    "leading_silence_ms": result.leading_silence_ms,
                    "trailing_silence_ms": result.trailing_silence_ms,
                    "total_silence_ms": result.total_silence_ms,
                    "silence_percentage": result.silence_percentage,
                    "content_start_ms": result.content_start_ms,
                    "content_end_ms": result.content_end_ms,
                    "content_duration_ms": result.content_duration_ms,
                }
            )
        else:
            result_dict["error"] = result.error

        results.append(result_dict)

    return results


def insert_silence(
    input_path: str,
    output_path: Optional[str] = None,
    before_ms: float = 0,
    after_ms: float = 0,
) -> InsertSilenceResult:
    """Add silence before and/or after an audio file.

    Useful for adding consistent gaps between segments in audiobook production.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_padded' suffix.
        before_ms: Milliseconds of silence to add before audio. Default: 0.
        after_ms: Milliseconds of silence to add after audio. Default: 0.

    Returns:
        InsertSilenceResult with input/output paths and duration info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_padded{suffix}"))

    # Get original duration
    original_duration = get_audio_duration(input_path)

    # Insert silence
    _insert_silence(input_path, output_path, before_ms, after_ms)

    # Get new duration
    new_duration = get_audio_duration(output_path)

    return InsertSilenceResult(
        input_path=input_path,
        output_path=output_path,
        original_duration_ms=original_duration,
        new_duration_ms=new_duration,
        silence_before_ms=before_ms,
        silence_after_ms=after_ms,
    )


def crossfade_join(
    audio_paths: list[str],
    output_path: str,
    crossfade_ms: float = 50,
    output_format: str = "wav",
) -> CrossfadeResult:
    """Concatenate audio files with smooth crossfade transitions.

    Creates seamless transitions between audio segments by overlapping
    and fading between them.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        crossfade_ms: Duration of crossfade overlap in milliseconds. Default: 50ms.
            Typical values: 20-50ms for dialogue, 100-200ms for music.
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'wav'.

    Returns:
        CrossfadeResult with output info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If no audio paths provided or format unsupported.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if not audio_paths:
        raise ValueError("No audio paths provided")

    if output_format not in ("mp3", "wav", "m4a"):
        raise ValueError(f"Unsupported format: {output_format}. Use 'mp3', 'wav', or 'm4a'.")

    # Validate all input files exist
    for path in audio_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Perform crossfade concatenation
    _crossfade(audio_paths, output_path, crossfade_ms, output_format)

    # Get output info
    duration = get_audio_duration(output_path)

    return CrossfadeResult(
        output_path=output_path,
        input_count=len(audio_paths),
        total_duration_ms=duration,
        crossfade_ms=crossfade_ms,
        output_format=output_format,
    )


# ============================================================================
# Audio Design Tools
# ============================================================================


def mix_audio(
    audio_paths: list[str],
    output_path: str,
    volumes: Optional[list[float]] = None,
    normalize: bool = True,
) -> MixResult:
    """Mix multiple audio tracks together (layer them).

    All input files play simultaneously, mixed into a single output.
    Perfect for layering voice + background music + ambient sounds.

    Args:
        audio_paths: List of audio file paths to mix together.
        output_path: Path for the output mixed file.
        volumes: Optional list of volume multipliers for each track (1.0 = original).
            If not provided, all tracks are mixed at original volume.
            Example: [1.0, 0.3, 0.5] - full voice, 30% music, 50% ambience.
        normalize: If True, normalize the output to prevent clipping (default True).

    Returns:
        MixResult with output info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If no audio paths provided.

    Example:
        # Layer voice over background music
        mix_audio(
            ["narration.wav", "music.wav"],
            "scene.wav",
            volumes=[1.0, 0.2]  # Full narration, 20% music
        )
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if not audio_paths:
        raise ValueError("No audio paths provided")

    # Validate all input files exist
    for path in audio_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Mix tracks
    _mix_tracks(audio_paths, output_path, volumes, normalize)

    # Get output info
    duration = get_audio_duration(output_path)

    return MixResult(
        output_path=output_path,
        input_count=len(audio_paths),
        duration_ms=duration,
        normalized=normalize,
    )


def adjust_volume(
    input_path: str,
    output_path: Optional[str] = None,
    volume: float = 1.0,
    volume_db: Optional[float] = None,
) -> VolumeResult:
    """Adjust the volume of an audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_vol' suffix.
        volume: Volume multiplier (1.0 = original, 2.0 = double, 0.5 = half).
            Ignored if volume_db is specified.
        volume_db: Volume adjustment in dB (overrides volume if specified).
            Positive = louder, negative = quieter.
            +6dB = roughly 2x louder, -6dB = roughly half.

    Returns:
        VolumeResult with input/output paths and adjustment info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_vol{suffix}"))

    # Adjust volume
    _adjust_volume(input_path, output_path, volume, volume_db)

    # Get duration
    duration = get_audio_duration(output_path)

    # Describe the change
    if volume_db is not None:
        volume_change = f"{volume_db:+.1f}dB"
    else:
        volume_change = f"{volume}x"

    return VolumeResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
        volume_change=volume_change,
    )


def apply_fade(
    input_path: str,
    output_path: Optional[str] = None,
    fade_in_ms: float = 0,
    fade_out_ms: float = 0,
) -> FadeResult:
    """Apply fade in and/or fade out to audio.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_faded' suffix.
        fade_in_ms: Duration of fade in at start in ms (0 = no fade in).
        fade_out_ms: Duration of fade out at end in ms (0 = no fade out).

    Returns:
        FadeResult with input/output paths and fade durations.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_faded{suffix}"))

    # Apply fade
    _apply_fade(input_path, output_path, fade_in_ms, fade_out_ms)

    # Get duration
    duration = get_audio_duration(output_path)

    return FadeResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
    )


def apply_effects(
    input_path: str,
    output_path: Optional[str] = None,
    lowpass_hz: Optional[float] = None,
    highpass_hz: Optional[float] = None,
    bass_gain_db: Optional[float] = None,
    treble_gain_db: Optional[float] = None,
    speed: Optional[float] = None,
    reverb: bool = False,
    echo_delay_ms: Optional[float] = None,
    echo_decay: float = 0.5,
) -> EffectsResult:
    """Apply audio effects to a file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_fx' suffix.
        lowpass_hz: Low-pass filter cutoff (removes high frequencies above this).
            Example: 3000 for "telephone" effect, 8000 for "muffled" sound.
        highpass_hz: High-pass filter cutoff (removes low frequencies below this).
            Example: 300 to remove rumble, 80 for subtle bass cut.
        bass_gain_db: Bass boost/cut in dB. Positive = more bass, negative = less.
            Example: +6 for bass boost, -3 for subtle reduction.
        treble_gain_db: Treble boost/cut in dB. Positive = brighter, negative = darker.
            Example: +3 for clarity, -6 for warm sound.
        speed: Playback speed multiplier (0.5 = half speed, 2.0 = double).
            Note: This also affects pitch.
        reverb: Add reverb effect (room-like ambience).
        echo_delay_ms: Echo delay in milliseconds (None = no echo).
            Example: 200 for subtle echo, 500 for dramatic effect.
        echo_decay: Echo decay factor 0-1 (how quickly echo fades).

    Returns:
        EffectsResult with input/output paths and effects applied.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.

    Example:
        # Create a "phone call" effect
        apply_effects("voice.wav", lowpass_hz=3000, highpass_hz=300)

        # Add dramatic reverb
        apply_effects("voice.wav", reverb=True)

        # Speed up with echo
        apply_effects("voice.wav", speed=1.2, echo_delay_ms=200)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_fx{suffix}"))

    # Apply effects
    _apply_effects(
        input_path,
        output_path,
        lowpass_hz=lowpass_hz,
        highpass_hz=highpass_hz,
        bass_gain_db=bass_gain_db,
        treble_gain_db=treble_gain_db,
        speed=speed,
        reverb=reverb,
        echo_delay_ms=echo_delay_ms,
        echo_decay=echo_decay,
    )

    # Get duration
    duration = get_audio_duration(output_path)

    # Build list of effects applied
    effects_applied = []
    if lowpass_hz is not None:
        effects_applied.append(f"lowpass:{lowpass_hz}Hz")
    if highpass_hz is not None:
        effects_applied.append(f"highpass:{highpass_hz}Hz")
    if bass_gain_db is not None:
        effects_applied.append(f"bass:{bass_gain_db:+.1f}dB")
    if treble_gain_db is not None:
        effects_applied.append(f"treble:{treble_gain_db:+.1f}dB")
    if speed is not None and speed != 1.0:
        effects_applied.append(f"speed:{speed}x")
    if reverb:
        effects_applied.append("reverb")
    if echo_delay_ms is not None:
        effects_applied.append(f"echo:{echo_delay_ms}ms")

    return EffectsResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
        effects_applied=effects_applied,
    )


def overlay_audio(
    base_path: str,
    overlay_path: str,
    output_path: str,
    position_ms: float = 0,
    overlay_volume: float = 1.0,
) -> OverlayResult:
    """Overlay one audio track on top of another at a specific position.

    Useful for adding sound effects, background music, or ambience at specific times.

    Args:
        base_path: Path to the base audio file.
        overlay_path: Path to the audio file to overlay.
        output_path: Path for the output file.
        position_ms: Position in base audio where overlay starts (in milliseconds).
            Default: 0 (overlay starts at beginning).
        overlay_volume: Volume multiplier for the overlay (1.0 = original).
            Use lower values (0.2-0.5) for background music under narration.

    Returns:
        OverlayResult with paths and overlay info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If any input file doesn't exist.

    Example:
        # Add a sound effect at 5 seconds
        overlay_audio("narration.wav", "explosion.wav", "scene.wav", position_ms=5000)

        # Add background music at lower volume
        overlay_audio("narration.wav", "music.wav", "scene.wav", overlay_volume=0.2)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base file not found: {base_path}")
    if not os.path.exists(overlay_path):
        raise FileNotFoundError(f"Overlay file not found: {overlay_path}")

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Overlay audio
    _overlay_audio(base_path, overlay_path, output_path, position_ms, overlay_volume)

    # Get duration
    duration = get_audio_duration(output_path)

    return OverlayResult(
        base_path=base_path,
        overlay_path=overlay_path,
        output_path=output_path,
        duration_ms=duration,
        overlay_position_ms=position_ms,
        overlay_volume=overlay_volume,
    )


# ============================================================================
# Voice Modulation Tools
# ============================================================================

# Available voice effects and their descriptions
VOICE_EFFECTS = {
    "robot": "Robotic/synthetic voice using ring modulation",
    "chorus": "Choir/ensemble effect with multiple voices",
    "vibrato": "Pitch wobble effect",
    "flanger": "Sweeping phaser effect",
    "telephone": "Lo-fi telephone quality",
    "megaphone": "PA/bullhorn sound",
    "deep": "Deeper voice with bass boost",
    "chipmunk": "Higher pitched, faster voice",
    "whisper": "Soft whisper effect",
    "cave": "Cavernous echo effect",
}


def shift_pitch(
    input_path: str,
    output_path: Optional[str] = None,
    semitones: float = 0,
) -> PitchShiftResult:
    """Shift the pitch of audio without changing its speed.

    Uses librosa's high-quality pitch shifting algorithm.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_pitched' suffix.
        semitones: Pitch shift in semitones.
            Positive = higher pitch, negative = lower pitch.
            12 semitones = 1 octave.
            Typical range: -12 to +12.

    Returns:
        PitchShiftResult with input/output paths and shift info.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If librosa is not installed.

    Example:
        # Raise pitch by 4 semitones (major third)
        shift_pitch("voice.wav", "higher.wav", semitones=4)

        # Lower pitch by 5 semitones (perfect fourth)
        shift_pitch("voice.wav", "lower.wav", semitones=-5)
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "librosa is required for pitch shifting. Install with: pip install librosa"
        ) from e

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_pitched{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Shift pitch
    if semitones != 0:
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    else:
        y_shifted = y

    # Save output
    sf.write(str(output_path), y_shifted, sr)

    # Get duration
    duration_ms = int(len(y_shifted) / sr * 1000)

    return PitchShiftResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        semitones=semitones,
        sample_rate=sr,
    )


def stretch_time(
    input_path: str,
    output_path: Optional[str] = None,
    rate: float = 1.0,
) -> TimeStretchResult:
    """Stretch or compress the duration of audio without changing its pitch.

    Uses librosa's phase vocoder for high-quality time stretching.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_stretched' suffix.
        rate: Time stretch factor.
            > 1.0 = faster (shorter duration)
            < 1.0 = slower (longer duration)
            0.5 = half speed (double duration)
            2.0 = double speed (half duration)
            Typical range: 0.5 to 2.0.

    Returns:
        TimeStretchResult with input/output paths and stretch info.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If librosa is not installed.
        ValueError: If rate is invalid.

    Example:
        # Slow down to 75% speed
        stretch_time("voice.wav", "slow.wav", rate=0.75)

        # Speed up to 150% speed
        stretch_time("voice.wav", "fast.wav", rate=1.5)
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "librosa is required for time stretching. Install with: pip install librosa"
        ) from e

    if rate <= 0:
        raise ValueError(f"Rate must be positive, got: {rate}")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_stretched{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    original_duration_ms = int(len(y) / sr * 1000)

    # Stretch time
    if rate != 1.0:
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
    else:
        y_stretched = y

    # Save output
    sf.write(str(output_path), y_stretched, sr)

    # Get new duration
    new_duration_ms = int(len(y_stretched) / sr * 1000)

    return TimeStretchResult(
        input_path=input_path,
        output_path=output_path,
        original_duration_ms=original_duration_ms,
        new_duration_ms=new_duration_ms,
        rate=rate,
        sample_rate=sr,
    )


def apply_voice_effect(
    input_path: str,
    output_path: Optional[str] = None,
    effect: str = "robot",
    intensity: float = 0.5,
) -> VoiceEffectResult:
    """Apply a voice effect to audio using FFmpeg filters.

    Preserves the original sample rate of the input file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_effect' suffix.
        effect: Voice effect to apply. One of:
            - "robot": Robotic/synthetic voice
            - "chorus": Choir/ensemble effect
            - "vibrato": Pitch wobble
            - "flanger": Sweeping phaser effect
            - "telephone": Lo-fi telephone quality
            - "megaphone": PA/bullhorn sound (good for PA/intercom at 0.4-0.5)
            - "deep": Deeper voice with bass boost
            - "chipmunk": Higher pitched, faster voice
            - "whisper": Soft whisper effect
            - "cave": Cavernous echo (use 0.1-0.15 for subtle room ambience)
        intensity: Effect strength from 0.0 to 1.0. Default: 0.5.
            Higher values = more pronounced effect.
            Recommended intensities:
            - megaphone: 0.4-0.5 for PA/announcement systems
            - cave: 0.1-0.15 for subtle room ambience, higher causes extreme echo

    Returns:
        VoiceEffectResult with input/output paths and effect info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
        ValueError: If effect is not recognized.

    Example:
        # Apply robot voice effect
        apply_voice_effect("voice.wav", "robot.wav", effect="robot")

        # Apply subtle chorus effect
        apply_voice_effect("voice.wav", "chorus.wav", effect="chorus", intensity=0.3)

        # PA/intercom voice (recommended for announcements)
        apply_voice_effect("voice.wav", "pa.wav", effect="megaphone", intensity=0.4)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if effect not in VOICE_EFFECTS:
        available = ", ".join(VOICE_EFFECTS.keys())
        raise ValueError(f"Unknown effect: {effect}. Available: {available}")

    intensity = max(0.0, min(1.0, intensity))

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get input sample rate to preserve it
    props = get_audio_properties(input_path)
    sample_rate = props.sample_rate

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_{effect}{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg filter based on effect, intensity, and sample rate
    filter_chain = _build_voice_effect_filter(effect, intensity, sample_rate)

    # Apply effect using FFmpeg, preserving sample rate
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        filter_chain,
        "-ar",
        str(sample_rate),
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

    # Get duration
    duration = get_audio_duration(output_path)

    return VoiceEffectResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
        effect=effect,
        intensity=intensity,
    )


def _build_voice_effect_filter(effect: str, intensity: float, sample_rate: int = 44100) -> str:
    """Build FFmpeg filter chain for a voice effect.

    Args:
        effect: The effect name.
        intensity: Effect strength (0.0 to 1.0).
        sample_rate: Input sample rate (used for asetrate/aresample effects).

    Returns:
        FFmpeg audio filter string.
    """
    # Scale intensity parameters
    i = intensity  # shorthand

    if effect == "robot":
        # Ring modulation + flanger for robotic sound
        mod_freq = 30 + int(i * 70)  # 30-100 Hz modulation
        return f"afftfilt=real='hypot(re,im)*cos(2*PI*{mod_freq}*t)':imag='hypot(re,im)*sin(2*PI*{mod_freq}*t)',flanger=delay={int(1 + i * 4)}:depth={i * 2}"

    elif effect == "chorus":
        # Multi-voice chorus effect
        delays = f"{int(20 + i * 30)}|{int(30 + i * 40)}|{int(40 + i * 50)}"
        decays = f"{0.3 + i * 0.2}|{0.25 + i * 0.15}|{0.2 + i * 0.1}"
        speeds = f"{0.3 + i * 0.5}|{0.4 + i * 0.6}|{0.5 + i * 0.7}"
        depths = f"{0.2 + i * 0.3}|{0.3 + i * 0.4}|{0.4 + i * 0.5}"
        return f"chorus={0.5 + i * 0.3}:{0.7 + i * 0.2}:{delays}:{decays}:{speeds}:{depths}"

    elif effect == "vibrato":
        # Pitch wobble
        freq = 3 + i * 7  # 3-10 Hz wobble
        depth = 0.2 + i * 0.6  # 0.2-0.8 depth
        return f"vibrato=f={freq}:d={depth}"

    elif effect == "flanger":
        # Sweeping phaser
        delay = 2 + int(i * 8)  # 2-10 ms
        depth = 2 + int(i * 8)  # 2-10 ms
        speed = 0.2 + i * 0.6  # 0.2-0.8 Hz
        return f"flanger=delay={delay}:depth={depth}:speed={speed}"

    elif effect == "telephone":
        # Lo-fi telephone quality
        low_cut = 300 + int(i * 200)  # 300-500 Hz highpass
        high_cut = 3400 - int(i * 400)  # 3000-3400 Hz lowpass
        return f"highpass=f={low_cut},lowpass=f={high_cut},volume=1.2"

    elif effect == "megaphone":
        # PA/bullhorn sound - good for announcements at 0.4-0.5 intensity
        low_cut = 400 + int(i * 300)  # 400-700 Hz
        high_cut = 3000 - int(i * 1000)  # 2000-3000 Hz
        return f"highpass=f={low_cut},lowpass=f={high_cut},volume=1.5,aecho=0.6:0.3:10:0.3"

    elif effect == "deep":
        # Deeper voice with bass boost and slight pitch shift
        # Use actual sample rate instead of hardcoded 44100
        bass_boost = 6 + int(i * 10)  # 6-16 dB
        rate_factor = 0.95 - i * 0.1  # 0.85-0.95 of original
        return f"asetrate={sample_rate}*{rate_factor},aresample={sample_rate},bass=g={bass_boost}"

    elif effect == "chipmunk":
        # Higher pitched, faster voice
        # Use actual sample rate instead of hardcoded 44100
        rate_factor = 1.2 + i * 0.4  # 1.2-1.6 of original
        return f"asetrate={sample_rate}*{rate_factor},aresample={sample_rate}"

    elif effect == "whisper":
        # Soft whisper effect - emphasize high frequencies, add noise
        noise_amount = 0.01 + i * 0.03
        return f"highpass=f=500,treble=g={int(3 + i * 6)},anlmdn=s={noise_amount}"

    elif effect == "cave":
        # Cavernous echo effect
        # Warning: Use low intensity (0.1-0.15) for subtle room ambience
        # Higher values create extreme echo unsuitable for PA/intercom
        delay = int(200 + i * 400)  # 200-600 ms
        decay = 0.4 + i * 0.3  # 0.4-0.7
        return f"aecho=0.8:0.8:{delay}:{decay}"

    else:
        # Default: no effect
        return "anull"


def shift_formant(
    input_path: str,
    output_path: Optional[str] = None,
    shift_ratio: float = 1.0,
) -> FormantShiftResult:
    """Shift the formants of a voice to change its character.

    Formants determine the "character" of a voice. Shifting formants can make
    a voice sound more masculine or feminine without changing the pitch.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_formant' suffix.
        shift_ratio: Formant shift ratio.
            < 1.0 = more masculine (deeper resonance), e.g., 0.8
            > 1.0 = more feminine (higher resonance), e.g., 1.2
            1.0 = no change
            Typical range: 0.7 to 1.4.

    Returns:
        FormantShiftResult with input/output paths and shift info.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If required libraries are not installed.

    Example:
        # Make voice sound more feminine
        shift_formant("male.wav", "feminine.wav", shift_ratio=1.2)

        # Make voice sound more masculine
        shift_formant("female.wav", "masculine.wav", shift_ratio=0.85)

    Note:
        For best quality, install pyworld: pip install pyworld
        Falls back to librosa-based approximation if pyworld is not available.
    """
    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_formant{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Try pyworld first (best quality), fall back to librosa approximation
    try:
        return _shift_formant_pyworld(input_path, output_path, shift_ratio)
    except ImportError:
        return _shift_formant_librosa(input_path, output_path, shift_ratio)


def _shift_formant_pyworld(
    input_path: str,
    output_path: str,
    shift_ratio: float,
) -> FormantShiftResult:
    """Shift formants using pyworld WORLD vocoder (best quality)."""
    import numpy as np
    import pyworld as pw
    import soundfile as sf

    # Load audio
    y, sr = sf.read(input_path)

    # Ensure mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    # Convert to float64 for pyworld
    y = y.astype(np.float64)

    # Extract WORLD features
    f0, sp, ap = pw.wav2world(y, sr)

    # Shift formants by resampling spectral envelope
    if shift_ratio != 1.0:
        sp_shifted = np.zeros_like(sp)
        freq_axis = np.arange(sp.shape[1])
        new_freq_axis = freq_axis * shift_ratio

        for i in range(sp.shape[0]):
            # Interpolate spectral envelope to shifted frequencies
            sp_shifted[i] = np.interp(
                freq_axis,
                new_freq_axis,
                sp[i],
                left=sp[i, 0],
                right=sp[i, -1],
            )
    else:
        sp_shifted = sp

    # Synthesize with shifted formants
    y_out = pw.synthesize(f0, sp_shifted, ap, sr)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val * 0.95

    # Save output
    sf.write(output_path, y_out, sr)

    duration_ms = int(len(y_out) / sr * 1000)

    return FormantShiftResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        shift_ratio=shift_ratio,
        sample_rate=sr,
    )


def _shift_formant_librosa(
    input_path: str,
    output_path: str,
    shift_ratio: float,
) -> FormantShiftResult:
    """Shift formants using librosa (approximation when pyworld not available).

    This uses a combination of pitch shifting and time stretching to approximate
    formant shifting. Not as accurate as pyworld but works without extra dependencies.
    """
    import librosa
    import soundfile as sf

    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    if shift_ratio != 1.0:
        # Approximate formant shift by:
        # 1. Pitch shift in opposite direction of formant shift
        # 2. Time stretch to compensate
        # This creates a similar perceptual effect to formant shifting

        # Calculate semitones for inverse pitch shift
        # shift_ratio > 1 means higher formants, so we pitch down and speed up
        import math

        semitones = -12 * math.log2(shift_ratio)

        # Pitch shift
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

        # Time stretch to compensate for perceived speed change
        y_out = librosa.effects.time_stretch(y_shifted, rate=shift_ratio)
    else:
        y_out = y

    # Save output
    sf.write(output_path, y_out, sr)

    duration_ms = int(len(y_out) / sr * 1000)

    return FormantShiftResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        shift_ratio=shift_ratio,
        sample_rate=sr,
    )
