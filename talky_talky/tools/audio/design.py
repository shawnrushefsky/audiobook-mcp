"""Audio design tools for mixing, volume, effects, and overlay."""

import os
from pathlib import Path
from typing import Optional

from ...utils.ffmpeg import (
    check_ffmpeg,
    get_audio_duration,
    mix_audio_tracks as _mix_tracks,
    adjust_audio_volume as _adjust_volume,
    apply_audio_fade as _apply_fade,
    apply_audio_effects as _apply_effects,
    overlay_audio_at_position as _overlay_audio,
)

from .types import MixResult, VolumeResult, FadeResult, EffectsResult, OverlayResult


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
