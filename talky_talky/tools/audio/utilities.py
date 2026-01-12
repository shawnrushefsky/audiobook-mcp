"""Core audio utility functions."""

import os
from pathlib import Path
from typing import Optional

from ...utils.ffmpeg import (
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
)

from .types import (
    AudioInfo,
    ConvertResult,
    ConcatenateResult,
    NormalizeResult,
    ResampleResult,
    AudioCompatibilityResult,
)


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
    resample: bool = True,
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


def batch_normalize_audio(
    audio_paths: list[str],
    output_dir: Optional[str] = None,
    suffix: str = "_normalized",
) -> list[dict]:
    """Normalize multiple audio files to broadcast standard (-16 LUFS) at once.

    More efficient than calling normalize_audio individually for large batches.

    Args:
        audio_paths: List of paths to audio files to normalize.
        output_dir: Directory for output files. If None, outputs are created
            in the same directory as inputs with the suffix appended.
        suffix: Suffix to append to output filenames. Default: "_normalized".

    Returns:
        List of dicts with results for each file:
        - path: The input file path
        - status: "success" or "error"
        - output_path: Path to normalized file (if success)
        - duration_ms: Duration of the output
        - error: Error message (if status is "error")

    Example:
        results = batch_normalize_audio([
            "chapter_01.wav",
            "chapter_02.wav",
            "chapter_03.wav",
        ], output_dir="/path/to/normalized/")
    """
    results = []

    for input_path in audio_paths:
        result_dict = {"path": input_path}

        try:
            input_path_obj = Path(input_path)

            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(out_dir / f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}")
            else:
                output_path = str(
                    input_path_obj.with_name(
                        f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"
                    )
                )

            norm_result = normalize_audio(input_path=input_path, output_path=output_path)

            result_dict.update(
                {
                    "status": "success",
                    "output_path": norm_result.output_path,
                    "duration_ms": norm_result.duration_ms,
                }
            )

        except Exception as e:
            result_dict.update({"status": "error", "error": str(e)})

        results.append(result_dict)

    return results


def generate_silence(
    output_path: str,
    duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
) -> dict:
    """Generate a silent audio file of specified duration.

    Creates a pure silence audio file without requiring any input audio.
    Useful for creating gaps, pauses, or placeholder audio.

    Args:
        output_path: Path for the output audio file.
        duration_ms: Duration of silence in milliseconds.
        sample_rate: Sample rate in Hz. Default: 44100.
        channels: Number of audio channels (1=mono, 2=stereo). Default: 2.

    Returns:
        Dict with:
        - status: "success" or "error"
        - output_path: Path to the generated file
        - duration_ms: Duration of the output
        - sample_rate: Sample rate used
        - channels: Number of channels

    Raises:
        RuntimeError: If ffmpeg is not installed.
        ValueError: If duration_ms is not positive.

    Example:
        # Create 2 seconds of stereo silence
        result = generate_silence("pause.wav", duration_ms=2000)

        # Create 500ms mono silence at 24kHz (to match TTS output)
        result = generate_silence("gap.wav", duration_ms=500, sample_rate=24000, channels=1)
    """
    import subprocess

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    duration_secs = duration_ms / 1000

    # Use ffmpeg anullsrc to generate silence
    channel_layout = "stereo" if channels == 2 else "mono"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl={channel_layout}",
        "-t",
        str(duration_secs),
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    return {
        "status": "success",
        "output_path": output_path,
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def loop_audio_to_duration(
    input_path: str,
    output_path: Optional[str] = None,
    target_duration_ms: float = 0,
    crossfade_ms: float = 0,
) -> dict:
    """Loop a short audio file to reach a target duration.

    Repeats the audio as many times as needed to reach or exceed the target duration,
    then trims to exactly the target length. Useful for looping ambient sounds or
    background music to match chapter/scene length.

    Args:
        input_path: Path to the input audio file to loop.
        output_path: Optional output path. If not provided, creates a file
            with '_looped' suffix.
        target_duration_ms: Target duration in milliseconds.
        crossfade_ms: Optional crossfade between loops in milliseconds. Default: 0.
            Use 50-200ms for smoother loop transitions.

    Returns:
        Dict with:
        - status: "success" or "error"
        - input_path: Original file path
        - output_path: Looped file path
        - original_duration_ms: Original file duration
        - target_duration_ms: Requested target duration
        - actual_duration_ms: Actual output duration
        - loop_count: Number of times the audio was looped

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
        ValueError: If target_duration_ms is not positive.

    Example:
        # Loop 30-second ambient to match 10-minute chapter
        result = loop_audio_to_duration(
            "forest_ambience.wav",
            target_duration_ms=600000,  # 10 minutes
            crossfade_ms=100
        )
    """
    import subprocess
    import math

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if target_duration_ms <= 0:
        raise ValueError("target_duration_ms must be positive")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_looped{suffix}"))

    # Get original duration
    original_duration_ms = get_audio_duration(input_path)

    # Calculate how many loops we need
    loop_count = math.ceil(target_duration_ms / original_duration_ms)

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    target_duration_secs = target_duration_ms / 1000

    if crossfade_ms > 0 and loop_count > 1:
        # Use acrossfade filter for smoother transitions
        # First, create looped audio, then apply crossfade
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Loop the audio
            cmd = [
                "ffmpeg",
                "-y",
                "-stream_loop",
                str(loop_count - 1),  # -1 because original counts as first
                "-i",
                input_path,
                "-t",
                str(target_duration_secs),
                "-c",
                "copy",
                tmp_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Apply a subtle fade at loop points
            crossfade_secs = crossfade_ms / 1000
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-af",
                f"afade=t=in:st=0:d={crossfade_secs},afade=t=out:st={target_duration_secs - crossfade_secs}:d={crossfade_secs}",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Simple loop without crossfade
        cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop",
            str(loop_count - 1),
            "-i",
            input_path,
            "-t",
            str(target_duration_secs),
            "-c",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    # Get actual output duration
    actual_duration_ms = get_audio_duration(output_path)

    return {
        "status": "success",
        "input_path": input_path,
        "output_path": output_path,
        "original_duration_ms": original_duration_ms,
        "target_duration_ms": target_duration_ms,
        "actual_duration_ms": actual_duration_ms,
        "loop_count": loop_count,
    }
