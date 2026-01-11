"""Audio concatenation utilities."""

import subprocess
import os
import tempfile
from typing import Optional

from .core import ChapterMarker, get_audio_duration
from .format import convert_audio_format


def concatenate_audio_files(input_files: list[str], output_path: str, format: str = "mp3") -> None:
    """Concatenate multiple audio files into one."""
    if not input_files:
        raise ValueError("No input files provided")

    # Validate all input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file not found: {file}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a file list for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = f.name
        for file in input_files:
            # Escape single quotes in file paths
            escaped = file.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        # Set codec based on format
        if format == "mp3":
            codec = ["-c:a", "libmp3lame", "-q:a", "2"]
        elif format == "wav":
            codec = ["-c:a", "pcm_s16le"]
        elif format == "m4a":
            codec = ["-c:a", "aac", "-b:a", "192k"]
        else:
            codec = []

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, *codec, output_path],
            capture_output=True,
            check=True,
        )
    finally:
        os.unlink(list_path)


def create_audiobook_with_chapters(
    input_files: list[str],
    output_path: str,
    chapters: list[ChapterMarker],
    metadata: Optional[dict] = None,
) -> None:
    """Create an MP3 with chapter markers (ID3v2 chapters)."""
    if not input_files:
        raise ValueError("No input files provided")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # First concatenate all files to a temp file
    temp_path = os.path.join(output_dir or ".", ".temp_concat.mp3")

    try:
        concatenate_audio_files(input_files, temp_path, "mp3")

        # Get total duration for the last chapter end
        total_duration = get_audio_duration(temp_path)

        # Create FFMETADATA file for chapters
        metadata_content = ";FFMETADATA1\n"

        if metadata:
            if metadata.get("title"):
                metadata_content += f"title={metadata['title']}\n"
            if metadata.get("artist"):
                metadata_content += f"artist={metadata['artist']}\n"
            if metadata.get("album"):
                metadata_content += f"album={metadata['album']}\n"

        # Add chapter markers
        for i, chapter in enumerate(chapters):
            start_ms = chapter.start_ms
            # End is either the start of the next chapter or total duration
            end_ms = chapters[i + 1].start_ms if i + 1 < len(chapters) else total_duration

            metadata_content += f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\ntitle={chapter.title}\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            metadata_path = f.name
            f.write(metadata_content)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_path,
                    "-i",
                    metadata_path,
                    "-map_metadata",
                    "1",
                    "-c",
                    "copy",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
        finally:
            os.unlink(metadata_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def generate_silence(output_path: str, duration_ms: float, sample_rate: int = 44100) -> None:
    """Generate a silent audio file.

    Args:
        output_path: Path for output audio file.
        duration_ms: Duration of silence in milliseconds.
        sample_rate: Sample rate for the silence (default 44100 Hz).
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=stereo",
            "-t",
            str(duration_ms / 1000),
            output_path,
        ],
        capture_output=True,
        check=True,
    )


def concatenate_with_gaps(
    input_files: list[str],
    output_path: str,
    gap_ms: float = 0,
    format: str = "mp3",
) -> None:
    """Concatenate multiple audio files with silence gaps between them.

    Args:
        input_files: List of input audio file paths.
        output_path: Path for output audio file.
        gap_ms: Milliseconds of silence between each file.
        format: Output format (mp3, wav, m4a).
    """
    if not input_files:
        raise ValueError("No input files provided")

    if gap_ms <= 0:
        # No gaps, use standard concatenation
        concatenate_audio_files(input_files, output_path, format)
        return

    # Validate all input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file not found: {file}")

    # Create temp directory for silence files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get sample rate from first file
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_files[0],
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        sample_rate = int(result.stdout.strip().split("\n")[0])

        # Generate silence file
        silence_path = os.path.join(temp_dir, "silence.wav")
        generate_silence(silence_path, gap_ms, sample_rate)

        # Build file list with silence interleaved
        files_with_gaps = []
        for i, file in enumerate(input_files):
            files_with_gaps.append(file)
            if i < len(input_files) - 1:
                files_with_gaps.append(silence_path)

        # Concatenate with gaps
        concatenate_audio_files(files_with_gaps, output_path, format)


def concatenate_with_variable_gaps(
    input_files: list[str],
    output_path: str,
    gaps_ms: list[float],
    format: str = "wav",
) -> None:
    """Concatenate multiple audio files with variable silence gaps between them.

    Args:
        input_files: List of input audio file paths.
        output_path: Path for output audio file.
        gaps_ms: List of gap durations in milliseconds. Length must be len(input_files) - 1.
        format: Output format (mp3, wav, m4a).
    """
    if not input_files:
        raise ValueError("No input files provided")

    if len(gaps_ms) != len(input_files) - 1:
        raise ValueError(
            f"gaps_ms length ({len(gaps_ms)}) must be len(input_files) - 1 ({len(input_files) - 1})"
        )

    # Validate all input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file not found: {file}")

    # Create temp directory for silence files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get sample rate from first file
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_files[0],
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        sample_rate = int(result.stdout.strip().split("\n")[0])

        # Build file list with silence interleaved
        files_with_gaps = []
        for i, file in enumerate(input_files):
            files_with_gaps.append(file)
            if i < len(input_files) - 1:
                gap_ms = gaps_ms[i]
                if gap_ms > 0:
                    silence_path = os.path.join(temp_dir, f"silence_{i}.wav")
                    generate_silence(silence_path, gap_ms, sample_rate)
                    files_with_gaps.append(silence_path)

        # Concatenate with variable gaps
        concatenate_audio_files(files_with_gaps, output_path, format)


def crossfade_audio_files(
    input_files: list[str],
    output_path: str,
    crossfade_ms: float = 50,
    format: str = "wav",
) -> None:
    """Concatenate audio files with crossfade transitions.

    Args:
        input_files: List of input audio file paths.
        output_path: Path for output audio file.
        crossfade_ms: Duration of crossfade overlap in milliseconds.
        format: Output format (mp3, wav, m4a).
    """
    if not input_files:
        raise ValueError("No input files provided")

    if len(input_files) == 1:
        convert_audio_format(input_files[0], output_path)
        return

    # Validate all input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file not found: {file}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build complex filter for crossfade
    # Using acrossfade filter for each pair
    crossfade_secs = crossfade_ms / 1000

    # Build inputs
    inputs = []
    for file in input_files:
        inputs.extend(["-i", file])

    # Build filter chain for crossfading
    # For n files, we need n-1 crossfades
    filter_parts = []
    n = len(input_files)

    if n == 2:
        # Simple case: just crossfade two files
        filter_str = f"[0:a][1:a]acrossfade=d={crossfade_secs}:c1=tri:c2=tri[out]"
    else:
        # Chain multiple crossfades
        # First crossfade: [0][1] -> [cf1]
        filter_parts.append(f"[0:a][1:a]acrossfade=d={crossfade_secs}:c1=tri:c2=tri[cf1]")

        # Middle crossfades
        for i in range(2, n):
            prev_label = f"cf{i - 1}"
            next_label = f"cf{i}" if i < n - 1 else "out"
            filter_parts.append(
                f"[{prev_label}][{i}:a]acrossfade=d={crossfade_secs}:c1=tri:c2=tri[{next_label}]"
            )

        filter_str = ";".join(filter_parts)

    # Set codec based on format
    if format == "mp3":
        codec = ["-c:a", "libmp3lame", "-q:a", "2"]
    elif format == "wav":
        codec = ["-c:a", "pcm_s16le"]
    elif format == "m4a":
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = []

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_str,
        "-map",
        "[out]",
        *codec,
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=True)
