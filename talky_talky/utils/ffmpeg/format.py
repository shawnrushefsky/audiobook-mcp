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


def normalize_to_lufs(
    input_path: str,
    output_path: str,
    target_lufs: float = -20.0,
    true_peak: float = -1.5,
    lra: float = 11.0,
) -> dict:
    """Normalize audio to a specific LUFS target.

    Uses ffmpeg's loudnorm filter for broadcast-standard normalization.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output file.
        target_lufs: Target integrated loudness in LUFS (default: -20).
            Common values: -16 (broadcast), -14 (streaming), -20 (SFX mixing).
        true_peak: Maximum true peak in dBTP (default: -1.5).
        lra: Target loudness range (default: 11).

    Returns:
        Dict with input_lufs, output_lufs (if two-pass), and other metadata.
    """
    import json
    import re

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get input sample rate to preserve it
    props = get_audio_properties(input_path)

    # First pass: measure input levels
    measure_cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    measure_result = subprocess.run(measure_cmd, capture_output=True, text=True)

    # Parse measured values from stderr
    input_lufs = None
    stderr_output = measure_result.stderr

    # Try to extract JSON from output
    json_match = re.search(r"\{[^{}]*input_i[^{}]*\}", stderr_output, re.DOTALL)
    if json_match:
        try:
            measured = json.loads(json_match.group())
            input_lufs = float(measured.get("input_i", 0))
        except (json.JSONDecodeError, ValueError):
            pass

    # Second pass: apply normalization
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}",
            "-ar",
            str(props.sample_rate),
            output_path,
        ],
        capture_output=True,
        check=True,
    )

    return {
        "input_lufs": input_lufs,
        "target_lufs": target_lufs,
        "true_peak": true_peak,
    }


def measure_audio_levels(audio_path: str) -> dict:
    """Measure audio levels (mean, peak, RMS) using ffmpeg.

    Args:
        audio_path: Path to audio file.

    Returns:
        Dict with mean_volume, max_volume (peak), and other stats.
    """
    import re

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use volumedetect filter
    cmd = [
        "ffmpeg",
        "-i",
        audio_path,
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse output
    mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
    max_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)

    mean_db = float(mean_match.group(1)) if mean_match else -999.0
    peak_db = float(max_match.group(1)) if max_match else -999.0

    # RMS is approximately mean_volume for audio
    rms_db = mean_db

    return {
        "mean_db": mean_db,
        "peak_db": peak_db,
        "rms_db": rms_db,
    }


def overlay_multiple_tracks(
    base_path: str,
    overlays: list[dict],
    output_path: str,
) -> None:
    """Overlay multiple audio tracks onto a base track in one ffmpeg call.

    Args:
        base_path: Path to the base audio file.
        overlays: List of overlay dicts, each with:
            - path: Path to overlay audio file
            - position_ms: Position in base where overlay starts (default: 0)
            - volume: Volume multiplier for overlay (default: 1.0)
        output_path: Path for output file.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base file not found: {base_path}")

    for i, overlay in enumerate(overlays):
        if not os.path.exists(overlay["path"]):
            raise FileNotFoundError(f"Overlay file not found: {overlay['path']}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build ffmpeg command with complex filter
    # Input files: base + all overlays
    inputs = ["-i", base_path]
    for overlay in overlays:
        inputs.extend(["-i", overlay["path"]])

    # Build filter complex
    # Format: [0:a] is base, [1:a], [2:a], etc. are overlays
    filter_parts = []
    current_output = "[0:a]"

    for i, overlay in enumerate(overlays):
        position_ms = overlay.get("position_ms", 0)
        volume = overlay.get("volume", 1.0)
        position_sec = position_ms / 1000.0

        overlay_input = f"[{i + 1}:a]"

        # Apply volume if not 1.0
        if volume != 1.0:
            vol_label = f"[ov{i}]"
            filter_parts.append(f"{overlay_input}volume={volume}{vol_label}")
            overlay_input = vol_label

        # Apply delay and mix
        if position_sec > 0:
            delay_label = f"[del{i}]"
            delay_ms = int(position_ms)
            filter_parts.append(f"{overlay_input}adelay={delay_ms}|{delay_ms}{delay_label}")
            overlay_input = delay_label

        # Mix with current output
        mix_output = f"[mix{i}]" if i < len(overlays) - 1 else "[out]"
        filter_parts.append(
            f"{current_output}{overlay_input}amix=inputs=2:duration=longest{mix_output}"
        )
        current_output = mix_output

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=True)
