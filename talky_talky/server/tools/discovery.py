"""Discovery tools for checking system capabilities and availability."""

from typing import Literal, Optional

from ...tools.tts import (
    check_tts,
    get_tts_info,
    get_available_engines as get_available_tts,
    list_engines as list_tts_engines,
)
from ...tools.transcription import (
    check_transcription,
    get_transcription_info,
    get_available_engines as get_available_transcription,
    list_engines as list_transcription_engines,
)
from ...tools.analysis import (
    list_emotion_engines,
    list_similarity_engines,
    list_quality_engines,
    get_sfx_analysis_info,
)
from ...tools.songgen import (
    check_songgen,
    get_info as get_songgen_engine_info,
    get_available_engines as get_available_songgen,
)
from ...tools.assets import get_autotag_capabilities
from ..config import to_dict


def register_discovery_tools(mcp):
    """Register discovery tools with the MCP server."""

    @mcp.tool()
    def capabilities() -> dict:
        """Get a summary of all available capabilities.

        Returns categorized info about available engines and features
        without loading full documentation. Use this to discover what's
        available before calling specific tools.
        """
        tts_available = get_available_tts()
        tts_engines = list_tts_engines()
        transcription_available = get_available_transcription()
        songgen_available = get_available_songgen()

        # Categorize TTS engines by type
        cloning_engines = []
        design_engines = []
        preset_engines = []
        for engine_id in tts_available:
            info = tts_engines.get(engine_id)
            if info:
                if info.engine_type == "audio_prompted":
                    cloning_engines.append(engine_id)
                elif info.engine_type == "text_prompted":
                    design_engines.append(engine_id)
                else:
                    preset_engines.append(engine_id)

        return {
            "tts": {
                "available_engines": tts_available,
                "voice_cloning": cloning_engines,
                "voice_design": design_engines,
                "preset_voices": preset_engines,
            },
            "transcription": {
                "available_engines": transcription_available,
            },
            "song_generation": {
                "available_engines": songgen_available,
            },
            "analysis": {
                "emotion_detection": list_emotion_engines(),
                "voice_similarity": list_similarity_engines(),
                "speech_quality": list_quality_engines(),
            },
            "audio_processing": [
                "convert",
                "join",
                "normalize",
                "trim",
                "crossfade",
                "mix",
                "effects",
                "pitch_shift",
                "time_stretch",
                "voice_effects",
            ],
            "hint": "Use get_engines_info(subsystem) for detailed engine documentation",
        }

    @mcp.tool()
    def check_availability(
        subsystem: Optional[Literal["tts", "transcription", "analysis", "songgen", "all"]] = "all",
    ) -> dict:
        """Check if engines are available and properly configured.

        Args:
            subsystem: Which subsystem to check. Options: tts, transcription,
                analysis, songgen, or all (default).

        Returns device info, available engines, and setup instructions.
        """
        result = {}

        if subsystem in ("tts", "all"):
            result["tts"] = to_dict(check_tts())

        if subsystem in ("transcription", "all"):
            result["transcription"] = to_dict(check_transcription())

        if subsystem in ("analysis", "all"):
            result["analysis"] = {
                "emotion_engines": list_emotion_engines(),
                "similarity_engines": list_similarity_engines(),
                "quality_engines": list_quality_engines(),
                "sfx_analysis": get_sfx_analysis_info(),
                "autotag": get_autotag_capabilities(),
            }

        if subsystem in ("songgen", "all"):
            result["songgen"] = check_songgen()

        return result

    @mcp.tool()
    def get_engines_info(
        subsystem: Literal["tts", "transcription", "songgen"],
    ) -> dict:
        """Get detailed info about engines in a subsystem.

        Args:
            subsystem: Which subsystem. Options: tts, transcription, songgen.

        Returns engine names, descriptions, parameters, and supported features.
        """
        if subsystem == "tts":
            return get_tts_info()
        elif subsystem == "transcription":
            return get_transcription_info()
        elif subsystem == "songgen":
            engines = {}
            for engine_id in get_available_songgen():
                info = get_songgen_engine_info(engine_id)
                engines[engine_id] = to_dict(info)
            return {"engines": engines}
        else:
            return {"error": f"Unknown subsystem: {subsystem}"}

    @mcp.tool()
    def list_available(
        subsystem: Literal["tts", "transcription", "songgen"],
    ) -> dict:
        """List available engines in a subsystem.

        Args:
            subsystem: Which subsystem. Options: tts, transcription, songgen.

        Returns list of available engine IDs with basic info.
        """
        if subsystem == "tts":
            available = get_available_tts()
            engines = list_tts_engines()
            return {
                "engines": available,
                "info": {
                    eid: {
                        "name": engines[eid].name,
                        "type": engines[eid].engine_type,
                    }
                    for eid in available
                    if eid in engines
                },
            }
        elif subsystem == "transcription":
            available = get_available_transcription()
            engines = list_transcription_engines()
            return {
                "engines": available,
                "info": {eid: {"name": engines[eid].name} for eid in available if eid in engines},
            }
        elif subsystem == "songgen":
            return {
                "engines": get_available_songgen(),
                "note": "Requires CUDA GPU (10-28GB VRAM)",
            }
        else:
            return {"error": f"Unknown subsystem: {subsystem}"}
