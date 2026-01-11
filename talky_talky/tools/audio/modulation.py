"""Voice modulation tools for pitch shifting, time stretching, and voice effects."""

import subprocess
from pathlib import Path
from typing import Optional

from ...utils.ffmpeg import check_ffmpeg, get_audio_duration, get_audio_properties

from .types import PitchShiftResult, TimeStretchResult, VoiceEffectResult, FormantShiftResult


# Available voice effects and their descriptions
VOICE_EFFECTS = {
    "robot": "Robotic/synthetic voice using ring modulation",
    "chorus": "Choir/ensemble effect with multiple voices",
    "vibrato": "Pitch wobble effect",
    "flanger": "Sweeping phaser effect",
    "telephone": "Lo-fi telephone quality",
    "megaphone": "PA/bullhorn sound",
    "deep": "Deeper voice with bass boost",
    "chipmunk": "Higher pitched, faster voice",
    "whisper": "Soft whisper effect",
    "cave": "Cavernous echo effect",
}


def shift_pitch(
    input_path: str,
    output_path: Optional[str] = None,
    semitones: float = 0,
) -> PitchShiftResult:
    """Shift the pitch of audio without changing its speed.

    Uses librosa's high-quality pitch shifting algorithm.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_pitched' suffix.
        semitones: Pitch shift in semitones.
            Positive = higher pitch, negative = lower pitch.
            12 semitones = 1 octave.
            Typical range: -12 to +12.

    Returns:
        PitchShiftResult with input/output paths and shift info.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If librosa is not installed.

    Example:
        # Raise pitch by 4 semitones (major third)
        shift_pitch("voice.wav", "higher.wav", semitones=4)

        # Lower pitch by 5 semitones (perfect fourth)
        shift_pitch("voice.wav", "lower.wav", semitones=-5)
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "librosa is required for pitch shifting. Install with: pip install librosa"
        ) from e

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_pitched{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    # Shift pitch
    if semitones != 0:
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    else:
        y_shifted = y

    # Save output
    sf.write(str(output_path), y_shifted, sr)

    # Get duration
    duration_ms = int(len(y_shifted) / sr * 1000)

    return PitchShiftResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        semitones=semitones,
        sample_rate=sr,
    )


def stretch_time(
    input_path: str,
    output_path: Optional[str] = None,
    rate: float = 1.0,
) -> TimeStretchResult:
    """Stretch or compress the duration of audio without changing its pitch.

    Uses librosa's phase vocoder for high-quality time stretching.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_stretched' suffix.
        rate: Time stretch factor.
            > 1.0 = faster (shorter duration)
            < 1.0 = slower (longer duration)
            0.5 = half speed (double duration)
            2.0 = double speed (half duration)
            Typical range: 0.5 to 2.0.

    Returns:
        TimeStretchResult with input/output paths and stretch info.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If librosa is not installed.
        ValueError: If rate is invalid.

    Example:
        # Slow down to 75% speed
        stretch_time("voice.wav", "slow.wav", rate=0.75)

        # Speed up to 150% speed
        stretch_time("voice.wav", "fast.wav", rate=1.5)
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "librosa is required for time stretching. Install with: pip install librosa"
        ) from e

    if rate <= 0:
        raise ValueError(f"Rate must be positive, got: {rate}")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_stretched{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    original_duration_ms = int(len(y) / sr * 1000)

    # Stretch time
    if rate != 1.0:
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
    else:
        y_stretched = y

    # Save output
    sf.write(str(output_path), y_stretched, sr)

    # Get new duration
    new_duration_ms = int(len(y_stretched) / sr * 1000)

    return TimeStretchResult(
        input_path=input_path,
        output_path=output_path,
        original_duration_ms=original_duration_ms,
        new_duration_ms=new_duration_ms,
        rate=rate,
        sample_rate=sr,
    )


def apply_voice_effect(
    input_path: str,
    output_path: Optional[str] = None,
    effect: str = "robot",
    intensity: float = 0.5,
) -> VoiceEffectResult:
    """Apply a voice effect to audio using FFmpeg filters.

    Preserves the original sample rate of the input file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_effect' suffix.
        effect: Voice effect to apply. One of:
            - "robot": Robotic/synthetic voice
            - "chorus": Choir/ensemble effect
            - "vibrato": Pitch wobble
            - "flanger": Sweeping phaser effect
            - "telephone": Lo-fi telephone quality
            - "megaphone": PA/bullhorn sound (good for PA/intercom at 0.4-0.5)
            - "deep": Deeper voice with bass boost
            - "chipmunk": Higher pitched, faster voice
            - "whisper": Soft whisper effect
            - "cave": Cavernous echo (use 0.1-0.15 for subtle room ambience)
        intensity: Effect strength from 0.0 to 1.0. Default: 0.5.
            Higher values = more pronounced effect.
            Recommended intensities:
            - megaphone: 0.4-0.5 for PA/announcement systems
            - cave: 0.1-0.15 for subtle room ambience, higher causes extreme echo

    Returns:
        VoiceEffectResult with input/output paths and effect info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
        ValueError: If effect is not recognized.

    Example:
        # Apply robot voice effect
        apply_voice_effect("voice.wav", "robot.wav", effect="robot")

        # Apply subtle chorus effect
        apply_voice_effect("voice.wav", "chorus.wav", effect="chorus", intensity=0.3)

        # PA/intercom voice (recommended for announcements)
        apply_voice_effect("voice.wav", "pa.wav", effect="megaphone", intensity=0.4)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if effect not in VOICE_EFFECTS:
        available = ", ".join(VOICE_EFFECTS.keys())
        raise ValueError(f"Unknown effect: {effect}. Available: {available}")

    intensity = max(0.0, min(1.0, intensity))

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get input sample rate to preserve it
    props = get_audio_properties(input_path)
    sample_rate = props.sample_rate

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_{effect}{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg filter based on effect, intensity, and sample rate
    filter_chain = _build_voice_effect_filter(effect, intensity, sample_rate)

    # Apply effect using FFmpeg, preserving sample rate
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        filter_chain,
        "-ar",
        str(sample_rate),
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

    # Get duration
    duration = get_audio_duration(output_path)

    return VoiceEffectResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
        effect=effect,
        intensity=intensity,
    )


def _build_voice_effect_filter(effect: str, intensity: float, sample_rate: int = 44100) -> str:
    """Build FFmpeg filter chain for a voice effect.

    Args:
        effect: The effect name.
        intensity: Effect strength (0.0 to 1.0).
        sample_rate: Input sample rate (used for asetrate/aresample effects).

    Returns:
        FFmpeg audio filter string.
    """
    # Scale intensity parameters
    i = intensity  # shorthand

    if effect == "robot":
        # Ring modulation + flanger for robotic sound
        mod_freq = 30 + int(i * 70)  # 30-100 Hz modulation
        return f"afftfilt=real='hypot(re,im)*cos(2*PI*{mod_freq}*t)':imag='hypot(re,im)*sin(2*PI*{mod_freq}*t)',flanger=delay={int(1 + i * 4)}:depth={i * 2}"

    elif effect == "chorus":
        # Multi-voice chorus effect
        delays = f"{int(20 + i * 30)}|{int(30 + i * 40)}|{int(40 + i * 50)}"
        decays = f"{0.3 + i * 0.2}|{0.25 + i * 0.15}|{0.2 + i * 0.1}"
        speeds = f"{0.3 + i * 0.5}|{0.4 + i * 0.6}|{0.5 + i * 0.7}"
        depths = f"{0.2 + i * 0.3}|{0.3 + i * 0.4}|{0.4 + i * 0.5}"
        return f"chorus={0.5 + i * 0.3}:{0.7 + i * 0.2}:{delays}:{decays}:{speeds}:{depths}"

    elif effect == "vibrato":
        # Pitch wobble
        freq = 3 + i * 7  # 3-10 Hz wobble
        depth = 0.2 + i * 0.6  # 0.2-0.8 depth
        return f"vibrato=f={freq}:d={depth}"

    elif effect == "flanger":
        # Sweeping phaser
        delay = 2 + int(i * 8)  # 2-10 ms
        depth = 2 + int(i * 8)  # 2-10 ms
        speed = 0.2 + i * 0.6  # 0.2-0.8 Hz
        return f"flanger=delay={delay}:depth={depth}:speed={speed}"

    elif effect == "telephone":
        # Lo-fi telephone quality
        low_cut = 300 + int(i * 200)  # 300-500 Hz highpass
        high_cut = 3400 - int(i * 400)  # 3000-3400 Hz lowpass
        return f"highpass=f={low_cut},lowpass=f={high_cut},volume=1.2"

    elif effect == "megaphone":
        # PA/bullhorn sound - good for announcements at 0.4-0.5 intensity
        low_cut = 400 + int(i * 300)  # 400-700 Hz
        high_cut = 3000 - int(i * 1000)  # 2000-3000 Hz
        return f"highpass=f={low_cut},lowpass=f={high_cut},volume=1.5,aecho=0.6:0.3:10:0.3"

    elif effect == "deep":
        # Deeper voice with bass boost and slight pitch shift
        # Use actual sample rate instead of hardcoded 44100
        bass_boost = 6 + int(i * 10)  # 6-16 dB
        rate_factor = 0.95 - i * 0.1  # 0.85-0.95 of original
        return f"asetrate={sample_rate}*{rate_factor},aresample={sample_rate},bass=g={bass_boost}"

    elif effect == "chipmunk":
        # Higher pitched, faster voice
        # Use actual sample rate instead of hardcoded 44100
        rate_factor = 1.2 + i * 0.4  # 1.2-1.6 of original
        return f"asetrate={sample_rate}*{rate_factor},aresample={sample_rate}"

    elif effect == "whisper":
        # Soft whisper effect - emphasize high frequencies, add noise
        noise_amount = 0.01 + i * 0.03
        return f"highpass=f=500,treble=g={int(3 + i * 6)},anlmdn=s={noise_amount}"

    elif effect == "cave":
        # Cavernous echo effect
        # Warning: Use low intensity (0.1-0.15) for subtle room ambience
        # Higher values create extreme echo unsuitable for PA/intercom
        delay = int(200 + i * 400)  # 200-600 ms
        decay = 0.4 + i * 0.3  # 0.4-0.7
        return f"aecho=0.8:0.8:{delay}:{decay}"

    else:
        # Default: no effect
        return "anull"


def shift_formant(
    input_path: str,
    output_path: Optional[str] = None,
    shift_ratio: float = 1.0,
) -> FormantShiftResult:
    """Shift the formants of a voice to change its character.

    Formants determine the "character" of a voice. Shifting formants can make
    a voice sound more masculine or feminine without changing the pitch.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_formant' suffix.
        shift_ratio: Formant shift ratio.
            < 1.0 = more masculine (deeper resonance), e.g., 0.8
            > 1.0 = more feminine (higher resonance), e.g., 1.2
            1.0 = no change
            Typical range: 0.7 to 1.4.

    Returns:
        FormantShiftResult with input/output paths and shift info.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ImportError: If required libraries are not installed.

    Example:
        # Make voice sound more feminine
        shift_formant("male.wav", "feminine.wav", shift_ratio=1.2)

        # Make voice sound more masculine
        shift_formant("female.wav", "masculine.wav", shift_ratio=0.85)

    Note:
        For best quality, install pyworld: pip install pyworld
        Falls back to librosa-based approximation if pyworld is not available.
    """
    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_formant{suffix}"))

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Try pyworld first (best quality), fall back to librosa approximation
    try:
        return _shift_formant_pyworld(input_path, output_path, shift_ratio)
    except ImportError:
        return _shift_formant_librosa(input_path, output_path, shift_ratio)


def _shift_formant_pyworld(
    input_path: str,
    output_path: str,
    shift_ratio: float,
) -> FormantShiftResult:
    """Shift formants using pyworld WORLD vocoder (best quality)."""
    import numpy as np
    import pyworld as pw
    import soundfile as sf

    # Load audio
    y, sr = sf.read(input_path)

    # Ensure mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    # Convert to float64 for pyworld
    y = y.astype(np.float64)

    # Extract WORLD features
    f0, sp, ap = pw.wav2world(y, sr)

    # Shift formants by resampling spectral envelope
    if shift_ratio != 1.0:
        sp_shifted = np.zeros_like(sp)
        freq_axis = np.arange(sp.shape[1])
        new_freq_axis = freq_axis * shift_ratio

        for i in range(sp.shape[0]):
            # Interpolate spectral envelope to shifted frequencies
            sp_shifted[i] = np.interp(
                freq_axis,
                new_freq_axis,
                sp[i],
                left=sp[i, 0],
                right=sp[i, -1],
            )
    else:
        sp_shifted = sp

    # Synthesize with shifted formants
    y_out = pw.synthesize(f0, sp_shifted, ap, sr)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val * 0.95

    # Save output
    sf.write(output_path, y_out, sr)

    duration_ms = int(len(y_out) / sr * 1000)

    return FormantShiftResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        shift_ratio=shift_ratio,
        sample_rate=sr,
    )


def _shift_formant_librosa(
    input_path: str,
    output_path: str,
    shift_ratio: float,
) -> FormantShiftResult:
    """Shift formants using librosa (approximation when pyworld not available).

    This uses a combination of pitch shifting and time stretching to approximate
    formant shifting. Not as accurate as pyworld but works without extra dependencies.
    """
    import librosa
    import soundfile as sf

    # Load audio
    y, sr = librosa.load(input_path, sr=None)

    if shift_ratio != 1.0:
        # Approximate formant shift by:
        # 1. Pitch shift in opposite direction of formant shift
        # 2. Time stretch to compensate
        # This creates a similar perceptual effect to formant shifting

        # Calculate semitones for inverse pitch shift
        # shift_ratio > 1 means higher formants, so we pitch down and speed up
        import math

        semitones = -12 * math.log2(shift_ratio)

        # Pitch shift
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

        # Time stretch to compensate for perceived speed change
        y_out = librosa.effects.time_stretch(y_shifted, rate=shift_ratio)
    else:
        y_out = y

    # Save output
    sf.write(output_path, y_out, sr)

    duration_ms = int(len(y_out) / sr * 1000)

    return FormantShiftResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration_ms,
        shift_ratio=shift_ratio,
        sample_rate=sr,
    )
