"""FFmpeg wrapper functions for audio processing."""

import subprocess
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChapterMarker:
    title: str
    start_ms: int


@dataclass
class AudioValidation:
    valid: bool
    duration_ms: Optional[int] = None
    format: Optional[str] = None
    error: Optional[str] = None


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_audio_duration(file_path: str) -> int:
    """Get the duration of an audio file in milliseconds."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        seconds = float(result.stdout.strip())
        return round(seconds * 1000)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get audio duration: {e.stderr}")


@dataclass
class AudioProperties:
    """Audio file properties."""

    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    duration_ms: Optional[int] = None
    format: Optional[str] = None


def get_audio_properties(file_path: str) -> AudioProperties:
    """Get audio file properties including sample rate, channels, and duration.

    Args:
        file_path: Path to the audio file.

    Returns:
        AudioProperties with sample_rate, channels, bit_depth, duration_ms, format.

    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If ffprobe fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels,bits_per_sample:format=duration,format_name",
                "-of",
                "json",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        probe = json.loads(result.stdout)

        stream = probe.get("streams", [{}])[0]
        fmt = probe.get("format", {})

        sample_rate = int(stream.get("sample_rate", 44100))
        channels = int(stream.get("channels", 2))
        bits = stream.get("bits_per_sample")
        bit_depth = int(bits) if bits and int(bits) > 0 else None
        duration = fmt.get("duration")
        duration_ms = round(float(duration) * 1000) if duration else None
        format_name = fmt.get("format_name")

        return AudioProperties(
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            duration_ms=duration_ms,
            format=format_name,
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to get audio properties: {e}")


def resample_audio(
    input_path: str,
    output_path: str,
    target_sample_rate: int,
) -> None:
    """Resample audio to a target sample rate.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        target_sample_rate: Target sample rate in Hz (e.g., 24000, 44100, 48000).

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If target_sample_rate is invalid.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if target_sample_rate < 8000 or target_sample_rate > 192000:
        raise ValueError(
            f"Invalid sample rate: {target_sample_rate}. Must be between 8000 and 192000 Hz."
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(target_sample_rate),
            output_path,
        ],
        capture_output=True,
        check=True,
    )


def validate_audio_file(file_path: str) -> AudioValidation:
    """Validate that an audio file exists and is readable."""
    if not os.path.exists(file_path):
        return AudioValidation(valid=False, error="File not found")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "json",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        probe = json.loads(result.stdout)

        return AudioValidation(
            valid=True,
            duration_ms=round(float(probe["format"]["duration"]) * 1000),
            format=probe["format"]["format_name"],
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        return AudioValidation(valid=False, error=f"Invalid audio file: {e}")


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


def convert_audio_format(input_path: str, output_path: str, format: str = "mp3") -> None:
    """Convert audio file to a specific format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if format == "mp3":
        codec = ["-c:a", "libmp3lame", "-q:a", "2"]
    elif format == "wav":
        codec = ["-c:a", "pcm_s16le"]
    elif format == "m4a":
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = []

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, *codec, output_path], capture_output=True, check=True
    )


def normalize_audio(input_path: str, output_path: str) -> None:
    """Normalize audio levels (to -16 LUFS for podcast/audiobook standard).

    Preserves the original sample rate of the input file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get input sample rate to preserve it
    props = get_audio_properties(input_path)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            str(props.sample_rate),
            output_path,
        ],
        capture_output=True,
        check=True,
    )


def trim_audio_file(
    input_path: str,
    output_path: str,
    start_ms: Optional[float] = None,
    end_ms: Optional[float] = None,
) -> None:
    """Trim an audio file to the specified start and end times.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        start_ms: Start time in milliseconds (None = start from beginning).
        end_ms: End time in milliseconds (None = go to end of file).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", input_path]

    if start_ms is not None:
        cmd.extend(["-ss", str(start_ms / 1000)])

    if end_ms is not None:
        if start_ms is not None:
            # Duration from start point
            duration_ms = end_ms - start_ms
            cmd.extend(["-t", str(duration_ms / 1000)])
        else:
            cmd.extend(["-to", str(end_ms / 1000)])

    # Copy codec for lossless trim when possible
    cmd.extend(["-c", "copy", output_path])

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        # Fallback to re-encoding if copy fails (e.g., for format mismatches)
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start_ms is not None:
            cmd.extend(["-ss", str(start_ms / 1000)])
        if end_ms is not None:
            if start_ms is not None:
                duration_ms = end_ms - start_ms
                cmd.extend(["-t", str(duration_ms / 1000)])
            else:
                cmd.extend(["-to", str(end_ms / 1000)])
        cmd.append(output_path)
        subprocess.run(cmd, capture_output=True, check=True)


def insert_silence_file(
    input_path: str,
    output_path: str,
    before_ms: float = 0,
    after_ms: float = 0,
) -> None:
    """Add silence before and/or after an audio file.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        before_ms: Milliseconds of silence to add before audio.
        after_ms: Milliseconds of silence to add after audio.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if before_ms <= 0 and after_ms <= 0:
        # No silence to add, just copy
        convert_audio_format(input_path, output_path)
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get audio info for sample rate
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    sample_rate = int(result.stdout.strip().split("\n")[0])

    # Build filter for adding silence
    filters = []

    if before_ms > 0:
        # Add silence at the beginning using adelay
        filters.append(f"adelay={int(before_ms)}|{int(before_ms)}")

    if after_ms > 0:
        # Pad silence at the end
        after_samples = int((after_ms / 1000) * sample_rate)
        filters.append(f"apad=pad_len={after_samples}")

    filter_str = ",".join(filters)

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-af", filter_str, output_path],
        capture_output=True,
        check=True,
    )


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


def mix_audio_tracks(
    input_files: list[str],
    output_path: str,
    volumes: list[float] | None = None,
    normalize: bool = True,
) -> None:
    """Mix multiple audio tracks together (layer them).

    All input files play simultaneously, mixed into a single output.

    Args:
        input_files: List of audio file paths to mix together.
        output_path: Path for the output mixed file.
        volumes: Optional list of volume multipliers for each track (1.0 = original).
            If not provided, all tracks are mixed at original volume.
        normalize: If True, normalize the output to prevent clipping (default True).
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

    # Build inputs
    inputs = []
    for file in input_files:
        inputs.extend(["-i", file])

    # Build amix filter
    n = len(input_files)

    # If volumes specified, apply them before mixing
    if volumes and len(volumes) == n:
        # Create volume filters for each input
        vol_filters = []
        for i, vol in enumerate(volumes):
            vol_filters.append(f"[{i}:a]volume={vol}[v{i}]")

        # Mix the volume-adjusted streams
        mix_inputs = "".join(f"[v{i}]" for i in range(n))
        vol_filter_str = ";".join(vol_filters)
        filter_str = f"{vol_filter_str};{mix_inputs}amix=inputs={n}:duration=longest"
    else:
        # Direct mix without volume adjustment
        mix_inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_str = f"{mix_inputs}amix=inputs={n}:duration=longest"

    # Add normalization if requested
    if normalize:
        filter_str += ":normalize=1"
    else:
        filter_str += ":normalize=0"

    filter_str += "[out]"

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_str, "-map", "[out]", output_path]

    subprocess.run(cmd, capture_output=True, check=True)


def adjust_audio_volume(
    input_path: str,
    output_path: str,
    volume: float = 1.0,
    volume_db: float | None = None,
) -> None:
    """Adjust the volume of an audio file.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        volume: Volume multiplier (1.0 = original, 2.0 = double, 0.5 = half).
        volume_db: Volume adjustment in dB (overrides volume if specified).
            Positive = louder, negative = quieter.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if volume_db is not None:
        filter_str = f"volume={volume_db}dB"
    else:
        filter_str = f"volume={volume}"

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-af", filter_str, output_path],
        capture_output=True,
        check=True,
    )


def apply_audio_fade(
    input_path: str,
    output_path: str,
    fade_in_ms: float = 0,
    fade_out_ms: float = 0,
) -> None:
    """Apply fade in and/or fade out to audio.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        fade_in_ms: Duration of fade in at start (0 = no fade in).
        fade_out_ms: Duration of fade out at end (0 = no fade out).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if fade_in_ms <= 0 and fade_out_ms <= 0:
        convert_audio_format(input_path, output_path)
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get duration for fade out
    duration_ms = get_audio_duration(input_path)
    duration_secs = duration_ms / 1000

    filters = []

    if fade_in_ms > 0:
        fade_in_secs = fade_in_ms / 1000
        filters.append(f"afade=t=in:st=0:d={fade_in_secs}")

    if fade_out_ms > 0:
        fade_out_secs = fade_out_ms / 1000
        fade_out_start = max(0, duration_secs - fade_out_secs)
        filters.append(f"afade=t=out:st={fade_out_start}:d={fade_out_secs}")

    filter_str = ",".join(filters)

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-af", filter_str, output_path],
        capture_output=True,
        check=True,
    )


def apply_audio_effects(
    input_path: str,
    output_path: str,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    bass_gain_db: float | None = None,
    treble_gain_db: float | None = None,
    speed: float | None = None,
    reverb: bool = False,
    echo_delay_ms: float | None = None,
    echo_decay: float = 0.5,
) -> None:
    """Apply audio effects to a file.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output audio file.
        lowpass_hz: Low-pass filter cutoff frequency (removes high frequencies).
        highpass_hz: High-pass filter cutoff frequency (removes low frequencies).
        bass_gain_db: Bass boost/cut in dB (positive = boost, negative = cut).
        treble_gain_db: Treble boost/cut in dB (positive = boost, negative = cut).
        speed: Playback speed multiplier (0.5 = half speed, 2.0 = double).
            Note: This also affects pitch.
        reverb: Add reverb effect (room-like ambience).
        echo_delay_ms: Echo delay in milliseconds (None = no echo).
        echo_decay: Echo decay factor 0-1 (how quickly echo fades).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    filters = []

    if highpass_hz is not None:
        filters.append(f"highpass=f={highpass_hz}")

    if lowpass_hz is not None:
        filters.append(f"lowpass=f={lowpass_hz}")

    if bass_gain_db is not None:
        filters.append(f"bass=g={bass_gain_db}")

    if treble_gain_db is not None:
        filters.append(f"treble=g={treble_gain_db}")

    if speed is not None and speed != 1.0:
        # atempo only supports 0.5-2.0, chain for larger ranges
        if 0.5 <= speed <= 2.0:
            filters.append(f"atempo={speed}")
        elif speed < 0.5:
            # Chain multiple atempo filters
            filters.append("atempo=0.5")
            remaining = speed / 0.5
            while remaining < 0.5:
                filters.append("atempo=0.5")
                remaining *= 2
            if remaining != 1.0:
                filters.append(f"atempo={remaining}")
        else:  # speed > 2.0
            filters.append("atempo=2.0")
            remaining = speed / 2.0
            while remaining > 2.0:
                filters.append("atempo=2.0")
                remaining /= 2
            if remaining != 1.0:
                filters.append(f"atempo={remaining}")

    if reverb:
        # Simulate reverb with multiple delays
        filters.append("aecho=0.8:0.88:60:0.4")

    if echo_delay_ms is not None:
        # aecho format: in_gain:out_gain:delays:decays
        filters.append(f"aecho=0.8:0.9:{int(echo_delay_ms)}:{echo_decay}")

    if not filters:
        convert_audio_format(input_path, output_path)
        return

    filter_str = ",".join(filters)

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-af", filter_str, output_path],
        capture_output=True,
        check=True,
    )


def overlay_audio_at_position(
    base_path: str,
    overlay_path: str,
    output_path: str,
    position_ms: float = 0,
    overlay_volume: float = 1.0,
) -> None:
    """Overlay one audio track on top of another at a specific position.

    Useful for adding sound effects, background music, or ambience at specific times.

    Args:
        base_path: Path to the base audio file.
        overlay_path: Path to the audio file to overlay.
        output_path: Path for the output file.
        position_ms: Position in base audio where overlay starts (in milliseconds).
        overlay_volume: Volume multiplier for the overlay (1.0 = original).
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base file not found: {base_path}")
    if not os.path.exists(overlay_path):
        raise FileNotFoundError(f"Overlay file not found: {overlay_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build filter: delay overlay, adjust volume, mix
    if overlay_volume != 1.0:
        filter_str = (
            f"[1:a]adelay={int(position_ms)}|{int(position_ms)},volume={overlay_volume}[ov];"
            f"[0:a][ov]amix=inputs=2:duration=first:normalize=0"
        )
    else:
        filter_str = (
            f"[1:a]adelay={int(position_ms)}|{int(position_ms)}[ov];"
            f"[0:a][ov]amix=inputs=2:duration=first:normalize=0"
        )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            base_path,
            "-i",
            overlay_path,
            "-filter_complex",
            filter_str,
            output_path,
        ],
        capture_output=True,
        check=True,
    )
