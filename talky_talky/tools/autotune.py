"""Autotune and pitch correction tools for vocals.

Provides pitch detection, scale-aware pitch correction (autotune),
and music theory utilities for audio processing.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================================
# Music Theory Constants
# ============================================================================

# Standard A4 reference frequency (ISO 16)
A4_FREQ = 440.0
A4_MIDI = 69

# Note names for display
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale definitions as semitone intervals from root
SCALES = {
    # Major modes
    "major": [0, 2, 4, 5, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],  # Same as major
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    # Minor scales
    "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    # Pentatonic
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    # Blues
    "blues": [0, 3, 5, 6, 7, 10],
    # Chromatic (all notes - useful for subtle correction)
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# Key name to semitone offset from C
KEY_OFFSETS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


# ============================================================================
# Data Classes
# ============================================================================


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


@dataclass
class NoteInfo:
    """Information about a musical note."""

    name: str  # e.g., "A4"
    frequency: float
    midi_number: int
    octave: int
    semitone: int  # 0-11, where 0=C


# ============================================================================
# Music Theory Utilities
# ============================================================================


def freq_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI note number (can be fractional).

    Args:
        freq: Frequency in Hz.

    Returns:
        MIDI note number (69 = A4 = 440Hz).
    """
    if freq <= 0:
        return 0
    return 12 * math.log2(freq / A4_FREQ) + A4_MIDI


def midi_to_freq(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz.

    Args:
        midi: MIDI note number (can be fractional).

    Returns:
        Frequency in Hz.
    """
    return A4_FREQ * (2 ** ((midi - A4_MIDI) / 12))


def freq_to_note(freq: float) -> NoteInfo:
    """Convert frequency to nearest note information.

    Args:
        freq: Frequency in Hz.

    Returns:
        NoteInfo with note name, frequency, MIDI number, etc.
    """
    if freq <= 0:
        return NoteInfo(name="", frequency=0, midi_number=0, octave=0, semitone=0)

    midi = freq_to_midi(freq)
    midi_rounded = round(midi)
    semitone = midi_rounded % 12
    octave = (midi_rounded // 12) - 1  # MIDI octave convention
    note_name = NOTE_NAMES[semitone]

    return NoteInfo(
        name=f"{note_name}{octave}",
        frequency=midi_to_freq(midi_rounded),
        midi_number=midi_rounded,
        octave=octave,
        semitone=semitone,
    )


def get_scale_frequencies(
    key: str = "C",
    scale: str = "major",
    octave_range: tuple = (1, 7),
) -> list:
    """Get all frequencies in a given key and scale across octave range.

    Args:
        key: Root note (C, C#, D, etc.).
        scale: Scale name (major, minor, pentatonic, etc.).
        octave_range: Tuple of (min_octave, max_octave).

    Returns:
        List of frequencies in Hz for all scale degrees in range.
    """
    if key not in KEY_OFFSETS:
        raise ValueError(f"Unknown key: {key}. Valid keys: {list(KEY_OFFSETS.keys())}")
    if scale not in SCALES:
        raise ValueError(f"Unknown scale: {scale}. Valid scales: {list(SCALES.keys())}")

    key_offset = KEY_OFFSETS[key]
    scale_intervals = SCALES[scale]
    frequencies = []

    for octave in range(octave_range[0], octave_range[1] + 1):
        for interval in scale_intervals:
            # MIDI note: C4 = 60, so C0 = 12
            midi_note = 12 + (octave * 12) + key_offset + interval
            freq = midi_to_freq(midi_note)
            frequencies.append(freq)

    return sorted(frequencies)


def snap_frequency_to_scale(
    freq: float,
    key: str = "C",
    scale: str = "major",
    correction_strength: float = 1.0,
) -> tuple:
    """Snap a frequency to the nearest note in a scale.

    Args:
        freq: Input frequency in Hz.
        key: Root note of the scale.
        scale: Scale name.
        correction_strength: How strongly to correct (0.0-1.0).
            0.0 = no correction, 1.0 = snap exactly to note.

    Returns:
        Tuple of (corrected_freq, correction_cents).
    """
    if freq <= 0:
        return freq, 0.0

    # Get MIDI note (fractional)
    midi = freq_to_midi(freq)

    # Get key offset and scale intervals
    key_offset = KEY_OFFSETS.get(key, 0)
    scale_intervals = SCALES.get(scale, SCALES["chromatic"])

    # Find which scale degree we're closest to
    # Normalize MIDI to semitone within octave, relative to key
    semitone_in_key = (round(midi) - key_offset) % 12

    # Find nearest scale degree
    min_distance = float("inf")
    nearest_interval = 0

    for interval in scale_intervals:
        # Distance considering octave wraparound
        distance = min(
            abs(semitone_in_key - interval),
            abs(semitone_in_key - interval - 12),
            abs(semitone_in_key - interval + 12),
        )
        if distance < min_distance:
            min_distance = distance
            nearest_interval = interval

    # Calculate target MIDI note
    octave = round(midi) // 12
    target_midi = (octave * 12) + key_offset + nearest_interval

    # Adjust if we wrapped around octaves
    if abs(midi - target_midi) > 6:
        if midi > target_midi:
            target_midi += 12
        else:
            target_midi -= 12

    # Calculate correction
    correction_cents = (target_midi - midi) * 100  # 100 cents per semitone

    # Apply correction strength
    corrected_midi = midi + (target_midi - midi) * correction_strength
    corrected_freq = midi_to_freq(corrected_midi)

    return corrected_freq, correction_cents


# ============================================================================
# Pitch Detection
# ============================================================================


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

    # Sort by frequency and format
    detected_notes = [
        (name, midi_to_freq(freq_to_midi(midi_to_freq(69))), count)
        for name, count in sorted(
            note_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]  # Top 10 notes
    ]

    # Fix the note frequencies
    detected_notes = []
    for name, count in sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
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


# ============================================================================
# Autotune
# ============================================================================


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


def list_scales() -> dict:
    """List all available scales with their intervals.

    Returns:
        Dict mapping scale names to their semitone intervals.
    """
    return {name: intervals.copy() for name, intervals in SCALES.items()}


def list_keys() -> list:
    """List all available keys/root notes.

    Returns:
        List of key names (C, C#, Db, D, etc.).
    """
    return list(KEY_OFFSETS.keys())
