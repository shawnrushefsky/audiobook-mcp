"""Autotune and pitch correction tools for vocals.

Provides scale-aware pitch correction (autotune) for audio processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .music_theory import (
    KEY_OFFSETS,
    SCALES,
    list_keys,
    list_scales,
    snap_frequency_to_scale,
)
from .pitch_detection import PitchDetectionResult, detect_pitch

# Re-export for backwards compatibility
__all__ = [
    "AutotuneResult",
    "PitchDetectionResult",
    "autotune",
    "detect_pitch",
    "list_keys",
    "list_scales",
]


@dataclass
class AutotuneResult:
    """Result of autotune processing."""

    input_path: str
    output_path: str
    duration_ms: int
    sample_rate: int
    key: str
    scale: str
    correction_strength: float
    speed: float
    frames_corrected: int
    total_voiced_frames: int
    average_correction_cents: float
    max_correction_cents: float


def autotune(
    input_path: str,
    output_path: Optional[str] = None,
    key: str = "C",
    scale: str = "major",
    correction_strength: float = 1.0,
    speed: float = 1.0,
) -> AutotuneResult:
    """Apply autotune (pitch correction) to vocals.

    Uses pyworld WORLD vocoder for high-quality pitch correction.
    Detects pitch frame-by-frame and corrects to the nearest note
    in the specified key and scale.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_autotuned' suffix.
        key: Musical key/root note (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
            Also supports flats: Db, Eb, Gb, Ab, Bb.
        scale: Scale type. Options:
            - "major" / "ionian": Major scale (default)
            - "minor" / "aeolian": Natural minor scale
            - "harmonic_minor": Harmonic minor scale
            - "melodic_minor": Melodic minor scale
            - "dorian", "phrygian", "lydian", "mixolydian", "locrian": Modes
            - "major_pentatonic", "minor_pentatonic": Pentatonic scales
            - "blues": Blues scale
            - "chromatic": All 12 notes (subtle correction only)
        correction_strength: How strongly to correct pitch (0.0-1.0).
            0.0 = no correction (bypass)
            0.5 = subtle, natural correction
            1.0 = full "T-Pain" style hard correction (default)
        speed: How quickly to snap to the correct pitch.
            1.0 = instant snap (robotic effect)
            0.5 = medium speed (more natural)
            0.1 = slow glide (very natural, subtle)
            Note: Lower values apply smoothing across frames.

    Returns:
        AutotuneResult with processing details.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If pyworld is not installed.
        ValueError: If key or scale is invalid.

    Example:
        # Classic "T-Pain" hard autotune in A minor
        autotune("vocals.wav", "autotuned.wav", key="A", scale="minor")

        # Subtle pitch correction in C major
        autotune("vocals.wav", key="C", scale="major", correction_strength=0.5)

        # Natural-sounding correction with slow glide
        autotune("vocals.wav", key="G", scale="major",
                 correction_strength=0.8, speed=0.3)
    """
    try:
        import pyworld as pw
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "pyworld and soundfile are required for autotune. "
            "Install with: pip install pyworld soundfile"
        ) from e

    # Validate inputs
    if key not in KEY_OFFSETS:
        raise ValueError(f"Unknown key: {key}. Valid keys: {list(KEY_OFFSETS.keys())}")
    if scale not in SCALES:
        raise ValueError(f"Unknown scale: {scale}. Valid scales: {list(SCALES.keys())}")

    correction_strength = max(0.0, min(1.0, correction_strength))
    speed = max(0.01, min(1.0, speed))

    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_autotuned{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    y, sr = sf.read(input_path)

    # Ensure mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    # Convert to float64 for pyworld
    y = y.astype(np.float64)

    # Extract WORLD features
    # Use harvest for better quality pitch detection
    f0, t = pw.harvest(y, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)

    # Correct pitch frame by frame
    f0_corrected = np.copy(f0)
    corrections_cents = []
    frames_corrected = 0

    for i in range(len(f0)):
        if f0[i] > 0:  # Only correct voiced frames
            corrected_freq, correction_cents = snap_frequency_to_scale(
                f0[i], key, scale, correction_strength
            )

            # Apply speed smoothing
            if speed < 1.0 and i > 0 and f0_corrected[i - 1] > 0:
                # Smooth transition from previous frame
                f0_corrected[i] = f0_corrected[i - 1] + speed * (
                    corrected_freq - f0_corrected[i - 1]
                )
            else:
                f0_corrected[i] = corrected_freq

            corrections_cents.append(abs(correction_cents))
            if abs(correction_cents) > 1:  # More than 1 cent correction
                frames_corrected += 1

    # Synthesize with corrected pitch
    y_out = pw.synthesize(f0_corrected, sp, ap, sr)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val * 0.95

    # Save output
    sf.write(output_path, y_out, sr)

    # Calculate statistics
    voiced_frames = int(np.sum(f0 > 0))
    avg_correction = float(np.mean(corrections_cents)) if corrections_cents else 0.0
    max_correction = float(np.max(corrections_cents)) if corrections_cents else 0.0
    duration_ms = int(len(y_out) / sr * 1000)

    return AutotuneResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        sample_rate=sr,
        key=key,
        scale=scale,
        correction_strength=correction_strength,
        speed=speed,
        frames_corrected=frames_corrected,
        total_voiced_frames=voiced_frames,
        average_correction_cents=avg_correction,
        max_correction_cents=max_correction,
    )
