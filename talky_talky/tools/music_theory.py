"""Music theory constants and utilities.

Provides note names, scales, key offsets, and frequency/MIDI conversion utilities
for audio processing and pitch correction.
"""

import math
from dataclasses import dataclass


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
class NoteInfo:
    """Information about a musical note."""

    name: str  # e.g., "A4"
    frequency: float
    midi_number: int
    octave: int
    semitone: int  # 0-11, where 0=C


# ============================================================================
# Frequency/MIDI Conversion
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


# ============================================================================
# Scale Utilities
# ============================================================================


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
