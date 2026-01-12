"""Workflow tools - high-level operations for common tasks."""

from typing import Optional

from ...tools.tts import generate
from ...tools.audio import (
    concatenate_audio,
    normalize_audio,
    trim_audio,
    crossfade_join,
    mix_audio,
    overlay_audio,
    loop_audio_to_duration,
)
from ..config import resolve_output_path


def register_workflow_tools(mcp):
    """Register workflow tools with the MCP server."""

    @mcp.tool()
    def create_audiobook_chapter(
        segments: list[dict],
        output_path: str,
        engine: str = "chatterbox",
        reference_audio_paths: Optional[list[str]] = None,
        voice_description: Optional[str] = None,
        gap_between_segments_ms: float = 400,
        normalize: bool = True,
        trim_silence: bool = True,
    ) -> dict:
        """Generate TTS for multiple segments, join, and normalize.

        Args:
            segments: List of dicts with 'text' (and optional 'gap_after_ms').
            output_path: Final output file.
            engine: TTS engine to use.
            reference_audio_paths: Reference audio for cloning engines.
            voice_description: Voice description for maya1.
            gap_between_segments_ms: Default silence between segments.
            normalize: Normalize final output to -16 LUFS.
            trim_silence: Trim leading/trailing silence from segments.

        Returns status, output_path, duration_ms, segment_count.
        """
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp(prefix="audiobook_")
        segment_paths = []
        gaps = []

        try:
            # Generate TTS for each segment
            for i, seg in enumerate(segments):
                seg_path = os.path.join(temp_dir, f"segment_{i:04d}.wav")
                result = generate(
                    text=seg["text"],
                    output_path=seg_path,
                    engine=engine,
                    reference_audio_paths=reference_audio_paths,
                    voice_description=voice_description,
                )
                if result.status != "success":
                    return {
                        "status": "error",
                        "message": f"Failed on segment {i}: {result.message}",
                        "segment_index": i,
                    }

                # Optionally trim silence
                if trim_silence:
                    trim_result = trim_audio(seg_path, padding_ms=30)
                    if trim_result.status == "success":
                        os.replace(trim_result.output_path, seg_path)

                segment_paths.append(seg_path)
                gaps.append(seg.get("gap_after_ms", gap_between_segments_ms))

            # Remove last gap (not needed after final segment)
            if gaps:
                gaps = gaps[:-1]

            # Concatenate segments
            resolved_output = resolve_output_path(output_path)
            concat_result = concatenate_audio(
                audio_paths=segment_paths,
                output_path=resolved_output,
                gap_ms=gaps,
            )

            # Normalize if requested
            if normalize and concat_result.status == "success":
                norm_result = normalize_audio(resolved_output, resolved_output)
                final_duration = norm_result.duration_ms
            else:
                final_duration = concat_result.total_duration_ms

            return {
                "status": "success",
                "output_path": str(resolved_output),
                "duration_ms": final_duration,
                "segment_count": len(segments),
            }

        finally:
            # Cleanup temp files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @mcp.tool()
    def create_podcast_with_music(
        narration_segments: list[dict],
        output_path: str,
        intro_music_path: Optional[str] = None,
        outro_music_path: Optional[str] = None,
        background_music_path: Optional[str] = None,
        background_volume: float = 0.15,
        engine: str = "chatterbox",
        reference_audio_paths: Optional[list[str]] = None,
    ) -> dict:
        """Create podcast with intro/outro music and optional background.

        Args:
            narration_segments: List of dicts with 'text'.
            intro_music_path: Music to play at start.
            outro_music_path: Music to play at end.
            background_music_path: Low-volume background music.
            background_volume: Background music volume (0.1-0.3 recommended).
        """
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp(prefix="podcast_")

        try:
            # Generate narration
            narration_path = os.path.join(temp_dir, "narration.wav")
            chapter_result = create_audiobook_chapter.__wrapped__(
                segments=narration_segments,
                output_path=narration_path,
                engine=engine,
                reference_audio_paths=reference_audio_paths,
                gap_between_segments_ms=500,
                normalize=True,
            )

            if chapter_result["status"] != "success":
                return chapter_result

            parts = []

            # Add intro music
            if intro_music_path:
                parts.append(intro_music_path)

            parts.append(narration_path)

            # Add outro music
            if outro_music_path:
                parts.append(outro_music_path)

            # Join parts with crossfade
            if len(parts) > 1:
                joined_path = os.path.join(temp_dir, "joined.wav")
                crossfade_join(
                    audio_paths=parts,
                    output_path=joined_path,
                    crossfade_ms=100,
                )
                final_narration = joined_path
            else:
                final_narration = narration_path

            # Add background music if provided
            resolved_output = resolve_output_path(output_path)
            if background_music_path:
                # Loop background to match narration length
                from ...tools.audio import get_audio_info

                narration_info = get_audio_info(final_narration)
                looped_bg = os.path.join(temp_dir, "bg_looped.wav")
                loop_audio_to_duration(
                    background_music_path,
                    looped_bg,
                    target_duration_ms=narration_info.duration_ms,
                    crossfade_ms=50,
                )

                # Mix with narration
                mix_audio(
                    audio_paths=[final_narration, looped_bg],
                    output_path=str(resolved_output),
                    volumes=[1.0, background_volume],
                    normalize=True,
                )
            else:
                import shutil

                shutil.copy(final_narration, str(resolved_output))

            from ...tools.audio import get_audio_info

            final_info = get_audio_info(str(resolved_output))

            return {
                "status": "success",
                "output_path": str(resolved_output),
                "duration_ms": final_info.duration_ms,
                "has_intro": intro_music_path is not None,
                "has_outro": outro_music_path is not None,
                "has_background": background_music_path is not None,
            }

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @mcp.tool()
    def create_sound_effect_scene(
        base_ambience_path: str,
        output_path: str,
        duration_ms: float,
        sound_events: Optional[list[dict]] = None,
    ) -> dict:
        """Create a soundscape with ambient base and timed sound effects.

        Args:
            base_ambience_path: Background ambient audio to loop.
            duration_ms: Target scene duration.
            sound_events: List of {path, position_ms, volume} for effects.
                Example: [{"path": "gunshot.wav", "position_ms": 5000, "volume": 0.8}]

        Returns status, output_path, duration_ms.
        """
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp(prefix="scene_")

        try:
            # Loop ambient to target duration
            looped_ambient = os.path.join(temp_dir, "ambient_looped.wav")
            loop_result = loop_audio_to_duration(
                base_ambience_path,
                looped_ambient,
                target_duration_ms=duration_ms,
                crossfade_ms=100,
            )

            if loop_result["status"] != "success":
                return loop_result

            resolved_output = resolve_output_path(output_path)

            # Add sound events
            if sound_events:
                current_path = looped_ambient
                for i, event in enumerate(sound_events):
                    next_path = os.path.join(temp_dir, f"with_event_{i}.wav")
                    overlay_audio(
                        base_path=current_path,
                        overlay_path=event["path"],
                        output_path=next_path,
                        position_ms=event.get("position_ms", 0),
                        overlay_volume=event.get("volume", 1.0),
                    )
                    current_path = next_path

                import shutil

                shutil.copy(current_path, str(resolved_output))
            else:
                import shutil

                shutil.copy(looped_ambient, str(resolved_output))

            from ...tools.audio import get_audio_info

            final_info = get_audio_info(str(resolved_output))

            return {
                "status": "success",
                "output_path": str(resolved_output),
                "duration_ms": final_info.duration_ms,
                "event_count": len(sound_events) if sound_events else 0,
            }

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
