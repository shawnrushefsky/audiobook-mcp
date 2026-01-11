"""Audio format conversion utilities."""

import subprocess
import os

from .core import get_audio_properties


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
