"""Audio processing tools - format conversion, editing, effects, and modulation."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from ...tools.audio import (
    get_audio_info,
    convert_audio,
    concatenate_audio,
    normalize_audio,
    resample_audio,
    validate_audio_compatibility,
    is_ffmpeg_available,
    trim_audio,
    batch_trim_audio,
    batch_detect_silence,
    insert_silence,
    crossfade_join,
    mix_audio,
    adjust_volume,
    apply_fade,
    apply_effects,
    overlay_audio,
    shift_pitch,
    stretch_time,
    apply_voice_effect,
    shift_formant,
    VOICE_EFFECTS,
    batch_normalize_audio,
    generate_silence,
    loop_audio_to_duration,
    normalize_to_lufs,
    get_mean_level,
    compare_levels,
    overlay_multiple,
    batch_normalize_to_lufs,
)
from ...tools.autotune import (
    autotune as autotune_audio,
    detect_pitch as detect_audio_pitch,
    list_keys as get_autotune_keys,
    list_scales as get_autotune_scales,
)
from ..config import (
    DEFAULT_OUTPUT_DIR,
    _load_config,
    _save_config,
    get_output_dir,
    resolve_output_path,
    to_dict,
)


def register_audio_tools(mcp):
    """Register audio processing tools with the MCP server."""

    # ========== File Info & Config ==========

    @mcp.tool()
    def get_audio_file_info(audio_path: str) -> dict:
        """Get audio file info: format, duration, sample rate, channels."""
        return to_dict(get_audio_info(audio_path))

    @mcp.tool()
    def set_output_directory(directory: str) -> dict:
        """Set default output directory for generated audio.

        Args:
            directory: Path or "default" to reset to ~/Documents/talky-talky.
        """
        if directory.lower() == "default":
            output_dir = DEFAULT_OUTPUT_DIR
        else:
            output_dir = Path(directory).expanduser().resolve()

        created = not output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)

        config = _load_config()
        config["output_directory"] = str(output_dir)
        _save_config(config)

        return {"status": "success", "output_directory": str(output_dir), "created": created}

    @mcp.tool()
    def get_output_directory() -> dict:
        """Get current default output directory."""
        output_dir = get_output_dir()
        return {
            "output_directory": str(output_dir),
            "exists": output_dir.exists(),
            "is_default": str(output_dir) == str(DEFAULT_OUTPUT_DIR),
        }

    @mcp.tool()
    def check_ffmpeg_available() -> dict:
        """Check if ffmpeg is installed."""
        available = is_ffmpeg_available()
        return {
            "available": available,
            "install": None if available else "brew install ffmpeg (macOS) or apt install ffmpeg",
        }

    @mcp.tool()
    def play_audio(audio_path: str) -> dict:
        """Play audio in system default player."""
        path = Path(audio_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {path}"}

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform == "win32":
                subprocess.Popen(["start", "", str(path)], shell=True)
            else:
                subprocess.Popen(["xdg-open", str(path)])
            return {"status": "success", "path": str(path)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ========== Format Conversion ==========

    @mcp.tool()
    def convert_audio_format(
        input_path: str,
        output_format: str = "mp3",
        output_path: Optional[str] = None,
    ) -> dict:
        """Convert audio to different format (mp3, wav, m4a)."""
        return to_dict(convert_audio(input_path, output_format, output_path))

    @mcp.tool()
    def resample_audio_file(
        input_path: str,
        target_sample_rate: int = 44100,
        output_path: Optional[str] = None,
    ) -> dict:
        """Resample audio to target sample rate (24000, 44100, 48000 Hz)."""
        return to_dict(resample_audio(input_path, output_path, target_sample_rate))

    @mcp.tool()
    def check_audio_compatibility(audio_paths: list[str]) -> dict:
        """Check if audio files are compatible for joining."""
        return to_dict(validate_audio_compatibility(audio_paths))

    # ========== Concatenation & Joining ==========

    @mcp.tool()
    def join_audio_files(
        audio_paths: list[str],
        output_path: str,
        output_format: str = "wav",
        gap_ms: float | list[float] = 0,
        resample: bool = True,
        target_sample_rate: Optional[int] = None,
    ) -> dict:
        """Concatenate audio files with optional gaps.

        Args:
            audio_paths: Files to join in order.
            output_path: Output file path.
            gap_ms: Silence between segments (single value or list).
            resample: Auto-resample to match sample rates.
        """
        return to_dict(
            concatenate_audio(
                audio_paths, output_path, output_format, gap_ms, resample, target_sample_rate
            )
        )

    @mcp.tool()
    def crossfade_join_audio(
        audio_paths: list[str],
        output_path: str,
        crossfade_ms: float = 50,
        output_format: str = "wav",
        resample: bool = True,
        target_sample_rate: Optional[int] = None,
    ) -> dict:
        """Join audio files with crossfade transitions.

        Args:
            crossfade_ms: Overlap duration. 20-50 for dialogue, 100-200 for music.
        """
        return to_dict(
            crossfade_join(
                audio_paths, output_path, crossfade_ms, output_format, resample, target_sample_rate
            )
        )

    # ========== Trimming & Silence ==========

    @mcp.tool()
    def trim_audio_file(
        input_path: str,
        output_path: Optional[str] = None,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        padding_ms: float = 50,
    ) -> dict:
        """Trim audio to boundaries or auto-detect content.

        If start/end not provided, auto-detects and trims silence.
        """
        return to_dict(trim_audio(input_path, output_path, start_ms, end_ms, padding_ms))

    @mcp.tool()
    def batch_trim_audio_files(
        audio_paths: list[str],
        output_dir: Optional[str] = None,
        padding_ms: float = 50,
        suffix: str = "_trimmed",
    ) -> dict:
        """Trim multiple audio files to content boundaries."""
        results = batch_trim_audio(audio_paths, output_dir, padding_ms, suffix)
        succeeded = sum(1 for r in results if r.get("status") == "success")
        return {
            "status": "success" if succeeded == len(results) else "partial",
            "total": len(results),
            "succeeded": succeeded,
            "results": results,
        }

    @mcp.tool()
    def batch_analyze_silence(
        audio_paths: list[str],
        threshold_db: float = -40.0,
        min_silence_ms: float = 100.0,
    ) -> dict:
        """Detect silence in multiple audio files."""
        results = batch_detect_silence(audio_paths, threshold_db, min_silence_ms)
        return {"status": "success", "count": len(results), "results": results}

    @mcp.tool()
    def insert_audio_silence(
        input_path: str,
        output_path: Optional[str] = None,
        before_ms: float = 0,
        after_ms: float = 0,
    ) -> dict:
        """Add silence before/after audio."""
        return to_dict(insert_silence(input_path, output_path, before_ms, after_ms))

    @mcp.tool()
    def generate_silence_audio(
        output_path: str,
        duration_ms: float,
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> dict:
        """Generate a silent audio file."""
        return generate_silence(
            resolve_output_path(output_path), duration_ms, sample_rate, channels
        )

    # ========== Normalization ==========

    @mcp.tool()
    def normalize_audio_levels(
        input_path: str,
        output_path: Optional[str] = None,
    ) -> dict:
        """Normalize audio to broadcast standard (-16 LUFS)."""
        return to_dict(normalize_audio(input_path, output_path))

    @mcp.tool()
    def batch_normalize_audio_files(
        audio_paths: list[str],
        output_dir: Optional[str] = None,
        suffix: str = "_normalized",
    ) -> dict:
        """Normalize multiple audio files to -16 LUFS."""
        results = batch_normalize_audio(audio_paths, output_dir, suffix)
        succeeded = sum(1 for r in results if r.get("status") == "success")
        return {
            "status": "success" if succeeded == len(results) else "partial",
            "total": len(results),
            "succeeded": succeeded,
            "results": results,
        }

    @mcp.tool()
    def normalize_sfx_to_level(
        input_path: str,
        output_path: Optional[str] = None,
        target_lufs: float = -16.0,
    ) -> dict:
        """Normalize audio to specific LUFS level."""
        return normalize_to_lufs(input_path, output_path, target_lufs)

    @mcp.tool()
    def batch_normalize_sfx_to_lufs(
        audio_paths: list[str],
        target_lufs: float = -16.0,
        output_dir: Optional[str] = None,
        suffix: str = "_norm",
    ) -> dict:
        """Normalize multiple files to target LUFS."""
        return batch_normalize_to_lufs(audio_paths, target_lufs, output_dir, suffix)

    # ========== Mixing & Layering ==========

    @mcp.tool()
    def mix_audio_tracks(
        audio_paths: list[str],
        output_path: str,
        volumes: Optional[list[float]] = None,
        normalize: bool = True,
    ) -> dict:
        """Mix tracks together (simultaneous playback).

        Args:
            volumes: Volume per track [1.0, 0.3, 0.5] = full, 30%, 50%.
        """
        return to_dict(mix_audio(audio_paths, output_path, volumes, normalize))

    @mcp.tool()
    def overlay_audio_track(
        base_path: str,
        overlay_path: str,
        output_path: str,
        position_ms: float = 0,
        overlay_volume: float = 1.0,
    ) -> dict:
        """Overlay one track on another at specific position."""
        return to_dict(
            overlay_audio(base_path, overlay_path, output_path, position_ms, overlay_volume)
        )

    @mcp.tool()
    def overlay_multiple_tracks(
        base_path: str,
        overlays: list[dict],
        output_path: str,
    ) -> dict:
        """Overlay multiple tracks with individual positions and volumes.

        Args:
            overlays: List of {path, position_ms, volume} dicts.
        """
        return overlay_multiple(base_path, overlays, output_path)

    @mcp.tool()
    def loop_audio_to_target_duration(
        input_path: str,
        target_duration_ms: float,
        output_path: Optional[str] = None,
        crossfade_ms: float = 0,
    ) -> dict:
        """Loop audio to reach target duration."""
        return loop_audio_to_duration(input_path, output_path, target_duration_ms, crossfade_ms)

    # ========== Volume & Fades ==========

    @mcp.tool()
    def adjust_audio_volume(
        input_path: str,
        output_path: Optional[str] = None,
        volume: float = 1.0,
        volume_db: Optional[float] = None,
    ) -> dict:
        """Adjust volume (multiplier or dB)."""
        return to_dict(adjust_volume(input_path, output_path, volume, volume_db))

    @mcp.tool()
    def apply_audio_fade(
        input_path: str,
        output_path: Optional[str] = None,
        fade_in_ms: float = 0,
        fade_out_ms: float = 0,
    ) -> dict:
        """Apply fade in/out effects."""
        return to_dict(apply_fade(input_path, output_path, fade_in_ms, fade_out_ms))

    @mcp.tool()
    def get_audio_mean_level(audio_path: str) -> dict:
        """Get mean audio level in dBFS."""
        return get_mean_level(audio_path)

    @mcp.tool()
    def compare_audio_levels(audio_paths: list[str]) -> dict:
        """Compare loudness levels across multiple files."""
        return compare_levels(audio_paths)

    # ========== Effects ==========

    @mcp.tool()
    def apply_audio_effects(
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
    ) -> dict:
        """Apply audio effects: filters, EQ, speed, reverb, echo."""
        return to_dict(
            apply_effects(
                input_path,
                output_path,
                lowpass_hz,
                highpass_hz,
                bass_gain_db,
                treble_gain_db,
                speed,
                reverb,
                echo_delay_ms,
                echo_decay,
            )
        )

    # ========== Voice Modulation ==========

    @mcp.tool()
    def shift_audio_pitch(
        input_path: str,
        semitones: float = 0,
        output_path: Optional[str] = None,
    ) -> dict:
        """Shift pitch without changing speed. +12 = octave up."""
        return to_dict(shift_pitch(input_path, output_path, semitones))

    @mcp.tool()
    def stretch_audio_time(
        input_path: str,
        rate: float = 1.0,
        output_path: Optional[str] = None,
    ) -> dict:
        """Change duration without changing pitch. 0.5 = half speed, 2 = double."""
        return to_dict(stretch_time(input_path, output_path, rate))

    @mcp.tool()
    def apply_voice_effect_preset(
        input_path: str,
        effect: str = "robot",
        intensity: float = 0.5,
        output_path: Optional[str] = None,
    ) -> dict:
        """Apply voice effect preset.

        Effects: robot, chorus, vibrato, flanger, telephone, megaphone,
                 deep, chipmunk, whisper, cave.
        """
        return to_dict(apply_voice_effect(input_path, output_path, effect, intensity))

    @mcp.tool()
    def list_voice_effects() -> dict:
        """List available voice effect presets."""
        return {"effects": VOICE_EFFECTS, "count": len(VOICE_EFFECTS)}

    @mcp.tool()
    def shift_voice_formant(
        input_path: str,
        shift_ratio: float = 1.0,
        output_path: Optional[str] = None,
    ) -> dict:
        """Shift voice formants. <1 = masculine, >1 = feminine."""
        return to_dict(shift_formant(input_path, output_path, shift_ratio))

    # ========== Autotune ==========

    @mcp.tool()
    def autotune_vocals(
        input_path: str,
        key: str = "C",
        scale: str = "major",
        correction_strength: float = 1.0,
        speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> dict:
        """Apply pitch correction to vocals.

        Args:
            key: Root note (C, C#, D, etc.)
            scale: major, minor, blues, chromatic, etc.
            correction_strength: 0-1, higher = harder correction.
            speed: 0-1, higher = faster snap to pitch.
        """
        return to_dict(
            autotune_audio(input_path, output_path, key, scale, correction_strength, speed)
        )

    @mcp.tool()
    def detect_vocal_pitch(
        input_path: str,
        method: str = "harvest",
        frame_period_ms: float = 5.0,
    ) -> dict:
        """Detect pitch in audio. Returns frequency stats and detected notes."""
        return to_dict(detect_audio_pitch(input_path, method, frame_period_ms))

    @mcp.tool()
    def list_autotune_scales() -> dict:
        """List available scales for autotune."""
        return {"scales": get_autotune_scales()}

    @mcp.tool()
    def list_autotune_keys() -> dict:
        """List available keys for autotune."""
        return {"keys": get_autotune_keys()}
