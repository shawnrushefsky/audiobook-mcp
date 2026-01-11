"""Core FFmpeg utilities and data classes."""

import subprocess
import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChapterMarker:
    """Chapter marker for audiobook creation."""

    title: str
    start_ms: int


@dataclass
class AudioValidation:
    """Result of audio file validation."""

    valid: bool
    duration_ms: Optional[int] = None
    format: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AudioProperties:
    """Audio file properties."""

    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    duration_ms: Optional[int] = None
    format: Optional[str] = None


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
