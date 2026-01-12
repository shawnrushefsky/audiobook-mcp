"""Audio trimming, silence detection, and crossfade utilities."""

import os
from pathlib import Path
from typing import Optional

from ...utils.ffmpeg import (
    check_ffmpeg,
    get_audio_duration,
    get_audio_properties,
    resample_audio as _resample_audio,
    trim_audio_file as _trim_audio,
    insert_silence_file as _insert_silence,
    crossfade_audio_files as _crossfade,
)

from .types import TrimResult, InsertSilenceResult, CrossfadeResult


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
        from ..analysis import detect_silence as _detect_silence

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


def batch_trim_audio(
    audio_paths: list[str],
    output_dir: Optional[str] = None,
    padding_ms: float = 50,
    suffix: str = "_trimmed",
) -> list[dict]:
    """Trim multiple audio files to content boundaries at once.

    Auto-detects silence at the start and end of each file and trims to content.
    More efficient than calling trim_audio individually for large batches.

    Args:
        audio_paths: List of paths to audio files to trim.
        output_dir: Directory for output files. If None, outputs are created
            in the same directory as inputs with the suffix appended.
        padding_ms: Milliseconds of silence to keep as buffer. Default: 50ms.
        suffix: Suffix to append to output filenames. Default: "_trimmed".

    Returns:
        List of dicts with results for each file:
        - path: The input file path
        - status: "success" or "error"
        - output_path: Path to trimmed file (if success)
        - original_duration_ms: Original duration
        - trimmed_duration_ms: New duration
        - silence_removed_ms: Amount of silence removed
        - error: Error message (if status is "error")

    Example:
        results = batch_trim_audio([
            "segment_001.wav",
            "segment_002.wav",
            "segment_003.wav",
        ], output_dir="/path/to/trimmed/", padding_ms=30)
    """
    results = []

    for input_path in audio_paths:
        result_dict = {"path": input_path}

        try:
            # Determine output path
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

            # Trim the audio (auto-detect mode)
            trim_result = trim_audio(
                input_path=input_path,
                output_path=output_path,
                start_ms=None,  # Auto-detect
                end_ms=None,  # Auto-detect
                padding_ms=padding_ms,
            )

            result_dict.update(
                {
                    "status": "success",
                    "output_path": trim_result.output_path,
                    "original_duration_ms": trim_result.original_duration_ms,
                    "trimmed_duration_ms": trim_result.trimmed_duration_ms,
                    "silence_removed_ms": trim_result.silence_removed_ms,
                }
            )

        except Exception as e:
            result_dict.update({"status": "error", "error": str(e)})

        results.append(result_dict)

    return results


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
    from ..analysis import detect_silence as _detect_silence

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
    resample: bool = True,
    target_sample_rate: Optional[int] = None,
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
        resample: If True, resample all files to a common sample rate before joining.
            This prevents static/noise artifacts from sample rate mismatches.
        target_sample_rate: Target sample rate when resample=True. If None, uses
            the sample rate of the first file.

    Returns:
        CrossfadeResult with output info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If no audio paths provided, format unsupported, or sample rate
            mismatch detected without resample=True.
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
    paths_to_use = audio_paths
    temp_dir = None

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

            paths_to_use = resampled_paths
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

    try:
        # Perform crossfade concatenation
        _crossfade(paths_to_use, output_path, crossfade_ms, output_format)
    finally:
        # Clean up temp files
        if temp_dir:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    # Get output info
    duration = get_audio_duration(output_path)

    return CrossfadeResult(
        output_path=output_path,
        input_count=len(audio_paths),
        total_duration_ms=duration,
        crossfade_ms=crossfade_ms,
        output_format=output_format,
    )
