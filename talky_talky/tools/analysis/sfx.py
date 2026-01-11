"""Sound Effect Analysis Tools.

Analysis tools designed for non-speech audio like sound effects, music, and ambience.
Provides loudness analysis, clipping detection, spectral analysis, silence detection,
and format validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time


@dataclass
class LoudnessResult:
    """Result of loudness analysis."""

    status: str
    peak_db: float = 0.0  # Peak level in dBFS
    peak_linear: float = 0.0  # Peak level as linear value (0-1)
    rms_db: float = 0.0  # RMS level in dBFS
    lufs: float = 0.0  # Integrated loudness in LUFS
    dynamic_range_db: float = 0.0  # Difference between peak and RMS
    true_peak_db: float = 0.0  # Inter-sample true peak in dBTP
    is_clipping: bool = False  # True if peak >= 0 dBFS
    processing_time_ms: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "peak_db": self.peak_db,
            "peak_linear": self.peak_linear,
            "rms_db": self.rms_db,
            "lufs": self.lufs,
            "dynamic_range_db": self.dynamic_range_db,
            "true_peak_db": self.true_peak_db,
            "is_clipping": self.is_clipping,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ClippingResult:
    """Result of clipping detection."""

    status: str
    has_clipping: bool = False
    clipped_samples: int = 0  # Number of clipped samples
    clipped_percentage: float = 0.0  # Percentage of samples that are clipped
    total_samples: int = 0
    clipped_regions: list = field(default_factory=list)  # List of (start_ms, end_ms) tuples
    max_consecutive_clipped: int = 0  # Longest run of clipped samples
    processing_time_ms: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "has_clipping": self.has_clipping,
            "clipped_samples": self.clipped_samples,
            "clipped_percentage": self.clipped_percentage,
            "total_samples": self.total_samples,
            "clipped_regions": self.clipped_regions,
            "max_consecutive_clipped": self.max_consecutive_clipped,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class SpectralResult:
    """Result of spectral analysis."""

    status: str
    dominant_frequency_hz: float = 0.0  # Most prominent frequency
    frequency_centroid_hz: float = 0.0  # Spectral centroid (brightness)
    bandwidth_hz: float = 0.0  # Spectral bandwidth
    low_freq_energy: float = 0.0  # Energy in 20-250 Hz (bass)
    mid_freq_energy: float = 0.0  # Energy in 250-4000 Hz (mids)
    high_freq_energy: float = 0.0  # Energy in 4000-20000 Hz (highs)
    rolloff_hz: float = 0.0  # Frequency below which 85% of energy exists
    zero_crossing_rate: float = 0.0  # Rate of sign changes (noisiness indicator)
    processing_time_ms: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "frequency_centroid_hz": self.frequency_centroid_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "low_freq_energy": self.low_freq_energy,
            "mid_freq_energy": self.mid_freq_energy,
            "high_freq_energy": self.high_freq_energy,
            "rolloff_hz": self.rolloff_hz,
            "zero_crossing_rate": self.zero_crossing_rate,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class SilenceResult:
    """Result of silence detection."""

    status: str
    leading_silence_ms: float = 0.0  # Silence at start
    trailing_silence_ms: float = 0.0  # Silence at end
    total_silence_ms: float = 0.0  # Total silence duration
    silence_percentage: float = 0.0  # Percentage of audio that is silent
    silence_regions: list = field(default_factory=list)  # List of (start_ms, end_ms, duration_ms)
    content_start_ms: float = 0.0  # Where actual content begins
    content_end_ms: float = 0.0  # Where actual content ends
    content_duration_ms: float = 0.0  # Duration of non-silent content
    processing_time_ms: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "leading_silence_ms": self.leading_silence_ms,
            "trailing_silence_ms": self.trailing_silence_ms,
            "total_silence_ms": self.total_silence_ms,
            "silence_percentage": self.silence_percentage,
            "silence_regions": self.silence_regions,
            "content_start_ms": self.content_start_ms,
            "content_end_ms": self.content_end_ms,
            "content_duration_ms": self.content_duration_ms,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class FormatValidationResult:
    """Result of format validation."""

    status: str
    is_valid: bool = False
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int | None = None  # None for compressed formats
    duration_ms: float = 0.0
    format: str = ""
    file_size_bytes: int = 0
    issues: list = field(default_factory=list)  # List of validation issues
    recommendations: list = field(default_factory=list)  # Suggested fixes
    processing_time_ms: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "is_valid": self.is_valid,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "duration_ms": self.duration_ms,
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


def _check_available() -> bool:
    """Check if required dependencies are available."""
    try:
        import librosa  # noqa: F401
        import numpy  # noqa: F401
        import soundfile  # noqa: F401

        return True
    except ImportError:
        return False


def analyze_loudness(audio_path: str) -> LoudnessResult:
    """Analyze loudness characteristics of an audio file.

    Measures:
    - Peak level (dBFS and linear)
    - RMS level (dBFS)
    - Integrated loudness (LUFS)
    - Dynamic range
    - True peak (dBTP)

    Args:
        audio_path: Path to audio file

    Returns:
        LoudnessResult with loudness measurements
    """
    start_time = time.time()

    if not _check_available():
        return LoudnessResult(
            status="error",
            error="Required dependencies not installed: librosa, numpy, soundfile",
        )

    try:
        import librosa
        import numpy as np

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return LoudnessResult(status="error", error=f"File not found: {audio_path}")

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=False)

        # Convert to mono for analysis if stereo
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y

        # Peak level
        peak_linear = float(np.max(np.abs(y_mono)))
        peak_db = float(20 * np.log10(peak_linear + 1e-10))

        # RMS level
        rms = float(np.sqrt(np.mean(y_mono**2)))
        rms_db = float(20 * np.log10(rms + 1e-10))

        # Dynamic range
        dynamic_range_db = peak_db - rms_db

        # LUFS (simplified ITU-R BS.1770-4 approximation)
        # For accurate LUFS, we'd need pyloudnorm, but this is a reasonable approximation
        # Apply K-weighting approximation using high-shelf and high-pass
        # Simplified: use RMS with frequency weighting approximation
        lufs = rms_db - 0.691  # Approximate offset for LUFS vs dBFS RMS

        # True peak estimation using oversampling
        # Upsample by 4x and find peak
        y_upsampled = librosa.resample(y_mono, orig_sr=sr, target_sr=sr * 4)
        true_peak_linear = float(np.max(np.abs(y_upsampled)))
        true_peak_db = float(20 * np.log10(true_peak_linear + 1e-10))

        # Check for clipping
        is_clipping = peak_db >= -0.1  # Allow tiny headroom

        processing_time_ms = int((time.time() - start_time) * 1000)

        return LoudnessResult(
            status="success",
            peak_db=round(peak_db, 2),
            peak_linear=round(peak_linear, 4),
            rms_db=round(rms_db, 2),
            lufs=round(lufs, 2),
            dynamic_range_db=round(dynamic_range_db, 2),
            true_peak_db=round(true_peak_db, 2),
            is_clipping=is_clipping,
            processing_time_ms=processing_time_ms,
            metadata={
                "sample_rate": sr,
                "duration_seconds": round(len(y_mono) / sr, 3),
                "audio_file": str(audio_path),
            },
        )

    except Exception as e:
        return LoudnessResult(
            status="error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


def detect_clipping(
    audio_path: str,
    threshold: float = 0.99,
    min_consecutive: int = 2,
) -> ClippingResult:
    """Detect clipping (digital distortion) in audio.

    Finds samples at or near maximum amplitude that indicate clipping.

    Args:
        audio_path: Path to audio file
        threshold: Amplitude threshold for clipping detection (0-1, default 0.99)
        min_consecutive: Minimum consecutive clipped samples to count as a region

    Returns:
        ClippingResult with clipping analysis
    """
    start_time = time.time()

    if not _check_available():
        return ClippingResult(
            status="error",
            error="Required dependencies not installed: librosa, numpy, soundfile",
        )

    try:
        import librosa
        import numpy as np

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return ClippingResult(status="error", error=f"File not found: {audio_path}")

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=False)

        # Convert to mono for analysis
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y

        total_samples = len(y_mono)

        # Find clipped samples
        clipped_mask = np.abs(y_mono) >= threshold
        clipped_samples = int(np.sum(clipped_mask))
        clipped_percentage = (clipped_samples / total_samples) * 100

        # Find clipped regions
        clipped_regions = []
        max_consecutive = 0

        if clipped_samples > 0:
            # Find runs of clipped samples
            diff = np.diff(np.concatenate([[0], clipped_mask.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                run_length = end - start
                if run_length >= min_consecutive:
                    start_ms = (start / sr) * 1000
                    end_ms = (end / sr) * 1000
                    clipped_regions.append(
                        {
                            "start_ms": round(start_ms, 2),
                            "end_ms": round(end_ms, 2),
                            "duration_ms": round(end_ms - start_ms, 2),
                            "samples": run_length,
                        }
                    )
                    max_consecutive = max(max_consecutive, run_length)

        has_clipping = len(clipped_regions) > 0

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ClippingResult(
            status="success",
            has_clipping=has_clipping,
            clipped_samples=clipped_samples,
            clipped_percentage=round(clipped_percentage, 4),
            total_samples=total_samples,
            clipped_regions=clipped_regions,
            max_consecutive_clipped=max_consecutive,
            processing_time_ms=processing_time_ms,
            metadata={
                "threshold": threshold,
                "min_consecutive": min_consecutive,
                "sample_rate": sr,
                "audio_file": str(audio_path),
            },
        )

    except Exception as e:
        return ClippingResult(
            status="error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


def analyze_spectrum(audio_path: str) -> SpectralResult:
    """Analyze spectral characteristics of audio.

    Measures:
    - Dominant frequency
    - Spectral centroid (brightness)
    - Spectral bandwidth
    - Energy distribution (low/mid/high)
    - Spectral rolloff
    - Zero crossing rate

    Args:
        audio_path: Path to audio file

    Returns:
        SpectralResult with spectral analysis
    """
    start_time = time.time()

    if not _check_available():
        return SpectralResult(
            status="error",
            error="Required dependencies not installed: librosa, numpy, soundfile",
        )

    try:
        import librosa
        import numpy as np

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return SpectralResult(status="error", error=f"File not found: {audio_path}")

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)

        # Compute spectrogram
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        # Dominant frequency (frequency with most energy)
        magnitude_sum = np.sum(S, axis=1)
        dominant_idx = np.argmax(magnitude_sum)
        dominant_frequency = float(freqs[dominant_idx])

        # Spectral centroid (brightness - weighted mean of frequencies)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        frequency_centroid = float(np.mean(centroid))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_hz = float(np.mean(bandwidth))

        # Spectral rolloff (frequency below which 85% of energy exists)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        rolloff_hz = float(np.mean(rolloff))

        # Zero crossing rate (indicator of noisiness/percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate = float(np.mean(zcr))

        # Energy distribution by frequency band
        # Low: 20-250 Hz, Mid: 250-4000 Hz, High: 4000-20000 Hz
        low_mask = (freqs >= 20) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs <= min(20000, sr / 2))

        total_energy = np.sum(magnitude_sum)
        low_energy = (
            float(np.sum(magnitude_sum[low_mask]) / total_energy) if total_energy > 0 else 0
        )
        mid_energy = (
            float(np.sum(magnitude_sum[mid_mask]) / total_energy) if total_energy > 0 else 0
        )
        high_energy = (
            float(np.sum(magnitude_sum[high_mask]) / total_energy) if total_energy > 0 else 0
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return SpectralResult(
            status="success",
            dominant_frequency_hz=round(dominant_frequency, 1),
            frequency_centroid_hz=round(frequency_centroid, 1),
            bandwidth_hz=round(bandwidth_hz, 1),
            low_freq_energy=round(low_energy, 4),
            mid_freq_energy=round(mid_energy, 4),
            high_freq_energy=round(high_energy, 4),
            rolloff_hz=round(rolloff_hz, 1),
            zero_crossing_rate=round(zero_crossing_rate, 4),
            processing_time_ms=processing_time_ms,
            metadata={
                "sample_rate": sr,
                "duration_seconds": round(len(y) / sr, 3),
                "audio_file": str(audio_path),
                "frequency_bands": {
                    "low": "20-250 Hz",
                    "mid": "250-4000 Hz",
                    "high": "4000-20000 Hz",
                },
            },
        )

    except Exception as e:
        return SpectralResult(
            status="error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


def detect_silence(
    audio_path: str,
    threshold_db: float = -40.0,
    min_silence_ms: float = 100.0,
) -> SilenceResult:
    """Detect silence regions in audio.

    Finds leading silence, trailing silence, and gaps within the audio.

    Args:
        audio_path: Path to audio file
        threshold_db: dB threshold below which audio is considered silent (default -40)
        min_silence_ms: Minimum duration in ms to count as silence (default 100)

    Returns:
        SilenceResult with silence analysis
    """
    start_time = time.time()

    if not _check_available():
        return SilenceResult(
            status="error",
            error="Required dependencies not installed: librosa, numpy, soundfile",
        )

    try:
        import librosa
        import numpy as np

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return SilenceResult(status="error", error=f"File not found: {audio_path}")

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)

        total_duration_ms = (len(y) / sr) * 1000

        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20)

        # Compute RMS energy in frames
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)  # 10ms hop

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Find silent frames
        silent_frames = rms < threshold_linear

        # Convert frames to time
        frame_times_ms = (
            librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000
        )

        # Find silence regions
        silence_regions = []
        min_frames = int(min_silence_ms / (hop_length / sr * 1000))

        # Find runs of silent frames
        diff = np.diff(np.concatenate([[0], silent_frames.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start_idx, end_idx in zip(starts, ends):
            if end_idx - start_idx >= min_frames:
                start_ms = (
                    frame_times_ms[start_idx]
                    if start_idx < len(frame_times_ms)
                    else total_duration_ms
                )
                end_ms = (
                    frame_times_ms[min(end_idx, len(frame_times_ms) - 1)]
                    if end_idx <= len(frame_times_ms)
                    else total_duration_ms
                )
                duration_ms = end_ms - start_ms
                silence_regions.append(
                    {
                        "start_ms": round(start_ms, 2),
                        "end_ms": round(end_ms, 2),
                        "duration_ms": round(duration_ms, 2),
                    }
                )

        # Calculate leading and trailing silence
        leading_silence_ms = 0.0
        trailing_silence_ms = 0.0

        if silence_regions:
            # Check if first region starts at beginning
            if silence_regions[0]["start_ms"] < 10:  # Within 10ms of start
                leading_silence_ms = silence_regions[0]["duration_ms"]

            # Check if last region ends at end
            if silence_regions[-1]["end_ms"] > total_duration_ms - 10:  # Within 10ms of end
                trailing_silence_ms = silence_regions[-1]["duration_ms"]

        # Calculate total silence
        total_silence_ms = sum(r["duration_ms"] for r in silence_regions)
        silence_percentage = (
            (total_silence_ms / total_duration_ms) * 100 if total_duration_ms > 0 else 0
        )

        # Content boundaries
        content_start_ms = leading_silence_ms
        content_end_ms = total_duration_ms - trailing_silence_ms
        content_duration_ms = content_end_ms - content_start_ms

        processing_time_ms = int((time.time() - start_time) * 1000)

        return SilenceResult(
            status="success",
            leading_silence_ms=round(leading_silence_ms, 2),
            trailing_silence_ms=round(trailing_silence_ms, 2),
            total_silence_ms=round(total_silence_ms, 2),
            silence_percentage=round(silence_percentage, 2),
            silence_regions=silence_regions,
            content_start_ms=round(content_start_ms, 2),
            content_end_ms=round(content_end_ms, 2),
            content_duration_ms=round(content_duration_ms, 2),
            processing_time_ms=processing_time_ms,
            metadata={
                "threshold_db": threshold_db,
                "min_silence_ms": min_silence_ms,
                "total_duration_ms": round(total_duration_ms, 2),
                "sample_rate": sr,
                "audio_file": str(audio_path),
            },
        )

    except Exception as e:
        return SilenceResult(
            status="error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


def validate_format(
    audio_path: str,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
    target_bit_depth: int | None = None,
    min_duration_ms: float | None = None,
    max_duration_ms: float | None = None,
    max_file_size_bytes: int | None = None,
) -> FormatValidationResult:
    """Validate audio format against target specifications.

    Checks sample rate, channels, bit depth, duration, and file size against
    provided targets and reports any mismatches.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Required sample rate (e.g., 44100, 48000)
        target_channels: Required channel count (1=mono, 2=stereo)
        target_bit_depth: Required bit depth (16, 24, 32)
        min_duration_ms: Minimum duration in milliseconds
        max_duration_ms: Maximum duration in milliseconds
        max_file_size_bytes: Maximum file size in bytes

    Returns:
        FormatValidationResult with validation results
    """
    start_time = time.time()

    try:
        import soundfile as sf

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return FormatValidationResult(status="error", error=f"File not found: {audio_path}")

        # Get file info
        info = sf.info(str(audio_path))
        file_size = audio_path.stat().st_size
        duration_ms = (info.frames / info.samplerate) * 1000

        # Determine bit depth from subtype
        bit_depth = None
        subtype = info.subtype
        if "PCM_16" in subtype:
            bit_depth = 16
        elif "PCM_24" in subtype:
            bit_depth = 24
        elif "PCM_32" in subtype or "FLOAT" in subtype:
            bit_depth = 32
        elif "PCM_S8" in subtype or "PCM_U8" in subtype:
            bit_depth = 8

        # Validate against targets
        issues = []
        recommendations = []

        if target_sample_rate and info.samplerate != target_sample_rate:
            issues.append(
                f"Sample rate {info.samplerate} Hz does not match target {target_sample_rate} Hz"
            )
            recommendations.append(f"Resample audio to {target_sample_rate} Hz")

        if target_channels and info.channels != target_channels:
            channel_names = {1: "mono", 2: "stereo"}
            current = channel_names.get(info.channels, f"{info.channels}-channel")
            target = channel_names.get(target_channels, f"{target_channels}-channel")
            issues.append(f"Audio is {current}, target is {target}")
            if target_channels == 1:
                recommendations.append("Convert to mono by mixing down channels")
            else:
                recommendations.append(f"Convert to {target}-channel audio")

        if target_bit_depth and bit_depth and bit_depth != target_bit_depth:
            issues.append(f"Bit depth {bit_depth} does not match target {target_bit_depth}")
            recommendations.append(f"Convert to {target_bit_depth}-bit audio")

        if min_duration_ms and duration_ms < min_duration_ms:
            issues.append(
                f"Duration {duration_ms:.0f}ms is shorter than minimum {min_duration_ms:.0f}ms"
            )
            recommendations.append("Extend audio or use a longer clip")

        if max_duration_ms and duration_ms > max_duration_ms:
            issues.append(f"Duration {duration_ms:.0f}ms exceeds maximum {max_duration_ms:.0f}ms")
            recommendations.append("Trim audio to fit within duration limit")

        if max_file_size_bytes and file_size > max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            max_mb = max_file_size_bytes / (1024 * 1024)
            issues.append(f"File size {size_mb:.2f}MB exceeds maximum {max_mb:.2f}MB")
            recommendations.append("Compress audio, reduce sample rate, or trim duration")

        is_valid = len(issues) == 0

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Determine format from extension and subtype
        format_name = audio_path.suffix.lstrip(".").upper()
        if format_name == "WAV":
            format_name = f"WAV ({subtype})"

        return FormatValidationResult(
            status="success",
            is_valid=is_valid,
            sample_rate=info.samplerate,
            channels=info.channels,
            bit_depth=bit_depth,
            duration_ms=round(duration_ms, 2),
            format=format_name,
            file_size_bytes=file_size,
            issues=issues,
            recommendations=recommendations,
            processing_time_ms=processing_time_ms,
            metadata={
                "subtype": subtype,
                "audio_file": str(audio_path),
                "targets": {
                    "sample_rate": target_sample_rate,
                    "channels": target_channels,
                    "bit_depth": target_bit_depth,
                    "min_duration_ms": min_duration_ms,
                    "max_duration_ms": max_duration_ms,
                    "max_file_size_bytes": max_file_size_bytes,
                },
            },
        )

    except Exception as e:
        return FormatValidationResult(
            status="error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


def get_sfx_analysis_info() -> dict:
    """Get information about SFX analysis capabilities.

    Returns:
        Dict with available tools and their descriptions
    """
    available = _check_available()

    return {
        "available": available,
        "tools": {
            "analyze_loudness": {
                "description": "Measure peak, RMS, LUFS, dynamic range, and detect potential clipping",
                "available": available,
            },
            "detect_clipping": {
                "description": "Find clipped samples and regions of digital distortion",
                "available": available,
            },
            "analyze_spectrum": {
                "description": "Analyze frequency content, brightness, and energy distribution",
                "available": available,
            },
            "detect_silence": {
                "description": "Find leading/trailing silence and gaps in audio",
                "available": available,
            },
            "validate_format": {
                "description": "Validate sample rate, channels, bit depth, duration against targets",
                "available": True,  # Only needs soundfile
            },
        },
        "requirements": "librosa, numpy, soundfile (pip install librosa soundfile)",
    }
