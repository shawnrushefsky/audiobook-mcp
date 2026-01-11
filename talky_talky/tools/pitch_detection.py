"""Pitch detection for audio files.

Provides high-quality pitch detection using pyworld's HARVEST and DIO algorithms.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .music_theory import NOTE_NAMES, freq_to_midi, freq_to_note, midi_to_freq


@dataclass
class PitchDetectionResult:
    """Result of pitch detection analysis."""

    input_path: str
    duration_ms: int
    sample_rate: int
    frame_count: int
    frame_period_ms: float
    voiced_frames: int
    unvoiced_frames: int
    pitch_range_hz: tuple  # (min, max)
    average_pitch_hz: float
    median_pitch_hz: float
    detected_notes: list  # List of (note_name, frequency, count)


def detect_pitch(
    input_path: str,
    method: str = "harvest",
    frame_period_ms: float = 5.0,
) -> PitchDetectionResult:
    """Detect pitch (fundamental frequency) in an audio file.

    Uses pyworld for high-quality pitch detection. Provides detailed
    analysis including pitch range, average pitch, and note distribution.

    Args:
        input_path: Path to the audio file.
        method: Pitch detection method.
            "harvest" - High quality, slower (default)
            "dio" - Faster, slightly less accurate
        frame_period_ms: Frame period in milliseconds (default 5.0).

    Returns:
        PitchDetectionResult with detailed pitch analysis.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If pyworld is not installed.
    """
    try:
        import pyworld as pw
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "pyworld and soundfile are required for pitch detection. "
            "Install with: pip install pyworld soundfile"
        ) from e

    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load audio
    y, sr = sf.read(input_path)

    # Ensure mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    # Convert to float64 for pyworld
    y = y.astype(np.float64)

    # Extract pitch
    if method == "dio":
        f0, t = pw.dio(y, sr, frame_period=frame_period_ms)
        f0 = pw.stonemask(y, f0, t, sr)  # Refine pitch
    else:  # harvest (default)
        f0, t = pw.harvest(y, sr, frame_period=frame_period_ms)

    # Analyze results
    voiced_mask = f0 > 0
    voiced_f0 = f0[voiced_mask]

    voiced_frames = int(np.sum(voiced_mask))
    unvoiced_frames = len(f0) - voiced_frames

    if len(voiced_f0) > 0:
        pitch_min = float(np.min(voiced_f0))
        pitch_max = float(np.max(voiced_f0))
        pitch_avg = float(np.mean(voiced_f0))
        pitch_median = float(np.median(voiced_f0))
    else:
        pitch_min = pitch_max = pitch_avg = pitch_median = 0.0

    # Count notes
    note_counts = {}
    for freq in voiced_f0:
        note_info = freq_to_note(freq)
        if note_info.name:
            note_counts[note_info.name] = note_counts.get(note_info.name, 0) + 1

    # Format detected notes with proper frequencies
    detected_notes = []
    for name, count in sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        # Parse note name to get frequency
        note_letter = name[:-1]  # e.g., "A" from "A4"
        octave = int(name[-1])
        semitone = NOTE_NAMES.index(note_letter)
        midi = 12 + (octave * 12) + semitone
        detected_notes.append((name, midi_to_freq(midi), count))

    duration_ms = int(len(y) / sr * 1000)

    return PitchDetectionResult(
        input_path=input_path,
        duration_ms=duration_ms,
        sample_rate=sr,
        frame_count=len(f0),
        frame_period_ms=frame_period_ms,
        voiced_frames=voiced_frames,
        unvoiced_frames=unvoiced_frames,
        pitch_range_hz=(pitch_min, pitch_max),
        average_pitch_hz=pitch_avg,
        median_pitch_hz=pitch_median,
        detected_notes=detected_notes,
    )
