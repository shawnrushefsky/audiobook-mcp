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
    concatenate_audio_files as _concat_files,
    concatenate_with_gaps as _concat_with_gaps,
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
    output_format: str = "mp3",
    gap_ms: float = 0,
) -> ConcatenateResult:
    """Concatenate multiple audio files into one.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'mp3'.
        gap_ms: Milliseconds of silence to insert between segments. Default: 0.
            Use 300ms for pauses between dialogue lines, 800ms for scene breaks.

    Returns:
        ConcatenateResult with output info.

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

    # Concatenate with or without gaps
    if gap_ms > 0:
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
