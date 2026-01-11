"""Audio manipulation utilities (trimming, effects, mixing)."""

import subprocess
import os
from typing import Optional

from .core import get_audio_duration
from .format import convert_audio_format


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
