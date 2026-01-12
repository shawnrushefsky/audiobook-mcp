"""TTS tools - consolidated speak() and model management."""

from typing import Literal, Optional

from ...tools.tts import generate, batch_generate
from ...tools.tts.maya1 import (
    check_models_downloaded as check_maya1_models,
    download_models as download_maya1_models,
)
from ..config import to_dict, resolve_output_path


# Engine type definitions for documentation
ENGINE_INFO = {
    "maya1": {
        "type": "voice_design",
        "description": "Design voice via text description",
        "tags": "<laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited>, <sad>",
        "required": ["voice_description"],
    },
    "chatterbox": {
        "type": "voice_cloning",
        "description": "Clone voice with expressiveness control",
        "tags": "[laugh], [sigh], [gasp], [chuckle], [cough], [hmm], [um]",
        "required": ["reference_audio_paths"],
    },
    "chatterbox_turbo": {
        "type": "voice_cloning",
        "description": "Fast voice cloning (simpler API)",
        "tags": "[laugh], [sigh], [gasp], [chuckle], [cough]",
        "required": ["reference_audio_paths"],
    },
    "mira": {
        "type": "voice_cloning",
        "description": "Fast 48kHz voice cloning",
        "tags": None,
        "required": ["reference_audio_paths"],
    },
    "xtts": {
        "type": "voice_cloning",
        "description": "Multilingual cloning (17 languages)",
        "tags": None,
        "required": ["reference_audio_paths"],
    },
    "kokoro": {
        "type": "preset_voices",
        "description": "54 pre-built voices, 8 languages",
        "tags": None,
        "required": [],
    },
    "soprano": {
        "type": "preset_voices",
        "description": "Ultra-fast CUDA TTS (2000x realtime)",
        "tags": None,
        "required": [],
    },
    "vibevoice_realtime": {
        "type": "preset_voices",
        "description": "Real-time TTS (~300ms latency)",
        "tags": None,
        "required": [],
    },
    "vibevoice_longform": {
        "type": "preset_voices",
        "description": "Long-form multi-speaker (up to 90min)",
        "tags": None,
        "required": [],
    },
    "cosyvoice3": {
        "type": "voice_cloning",
        "description": "Multilingual cloning (9 languages)",
        "tags": "[breath]",
        "required": ["reference_audio_paths"],
    },
    "seamlessm4t": {
        "type": "translation",
        "description": "Multilingual TTS with translation (35 languages)",
        "tags": None,
        "required": [],
    },
}


def register_tts_tools(mcp):
    """Register TTS tools with the MCP server."""

    @mcp.tool()
    def speak(
        text: str,
        output_path: str,
        engine: Literal[
            "maya1",
            "chatterbox",
            "chatterbox_turbo",
            "mira",
            "xtts",
            "kokoro",
            "soprano",
            "vibevoice_realtime",
            "vibevoice_longform",
            "cosyvoice3",
            "seamlessm4t",
        ] = "chatterbox",
        # Voice cloning params (for chatterbox, mira, xtts, cosyvoice3)
        reference_audio_paths: Optional[list[str]] = None,
        # Voice design params (for maya1)
        voice_description: Optional[str] = None,
        # Voice selection params (for kokoro, vibevoice)
        voice: Optional[str] = None,
        # Language params (for xtts, cosyvoice3, seamlessm4t)
        language: Optional[str] = None,
        src_language: Optional[str] = None,
        # Chatterbox-specific
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        # Sampling params (for maya1, soprano)
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        # Kokoro-specific
        speed: float = 1.0,
        # CosyVoice-specific
        prompt_text: Optional[str] = None,
        instruction: Optional[str] = None,
        # SeamlessM4T-specific
        speaker_id: int = 0,
        # VibeVoice-specific
        speakers: Optional[list[str]] = None,
    ) -> dict:
        """Generate speech using any TTS engine.

        Args:
            text: Text to synthesize. Use emotion tags for expressive speech:
                - Maya1: <laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited>
                - Chatterbox: [laugh], [sigh], [gasp], [chuckle], [hmm], [um]
                - CosyVoice: [breath]
            output_path: Where to save audio (filename or full path).
            engine: TTS engine to use. Options:
                - Voice cloning: chatterbox, chatterbox_turbo, mira, xtts, cosyvoice3
                - Voice design: maya1 (describe the voice you want)
                - Preset voices: kokoro, soprano, vibevoice_realtime, vibevoice_longform
                - Translation: seamlessm4t (text-to-speech with translation)

            reference_audio_paths: Reference audio for voice cloning engines.
            voice_description: Voice description for maya1.
            voice: Voice ID for kokoro (e.g., "af_heart") or vibevoice speaker.
            language: Target language for multilingual engines.
            src_language: Source language for translation (seamlessm4t).
            exaggeration: Expressiveness 0-1 for chatterbox (0.6-0.7 recommended).
            cfg_weight: Pacing control 0-1 for chatterbox.
            temperature: Sampling temperature for maya1/soprano.
            speed: Speech rate for kokoro (0.5-2.0).
            prompt_text: Reference audio transcript for cosyvoice3.
            instruction: Style instruction for cosyvoice3.
            speaker_id: Speaker index 0-199 for seamlessm4t.
            speakers: Speaker list for vibevoice_longform multi-speaker.

        Returns status, output_path, duration_ms, sample_rate, and metadata.
        """
        # Build kwargs based on engine
        kwargs = {
            "text": text,
            "output_path": resolve_output_path(output_path),
            "engine": engine,
        }

        # Add engine-specific parameters
        if engine in ("chatterbox", "chatterbox_turbo", "mira", "xtts", "cosyvoice3"):
            if not reference_audio_paths:
                return {
                    "status": "error",
                    "message": f"Engine '{engine}' requires reference_audio_paths",
                }
            kwargs["reference_audio_paths"] = reference_audio_paths

        if engine == "maya1":
            if not voice_description:
                return {
                    "status": "error",
                    "message": "Engine 'maya1' requires voice_description",
                }
            kwargs["voice_description"] = voice_description
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            kwargs["repetition_penalty"] = repetition_penalty

        if engine == "chatterbox":
            kwargs["exaggeration"] = exaggeration
            kwargs["cfg_weight"] = cfg_weight

        if engine == "kokoro":
            kwargs["voice"] = voice or "af_heart"
            kwargs["speed"] = speed

        if engine == "soprano":
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            kwargs["repetition_penalty"] = repetition_penalty

        if engine in ("vibevoice_realtime", "vibevoice_longform"):
            kwargs["voice_description"] = voice or "Carter"
            if engine == "vibevoice_longform" and speakers:
                kwargs["speakers"] = speakers

        if engine == "xtts":
            kwargs["language"] = language or "en"

        if engine == "cosyvoice3":
            kwargs["language"] = language or "auto"
            if prompt_text:
                kwargs["prompt_text"] = prompt_text
            if instruction:
                kwargs["instruction"] = instruction

        if engine == "seamlessm4t":
            kwargs["language"] = language or "en"
            if src_language:
                kwargs["src_language"] = src_language
            kwargs["speaker_id"] = speaker_id

        result = generate(**kwargs)
        return to_dict(result)

    @mcp.tool()
    def batch_speak(
        segments: list[dict],
        output_dir: str,
        engine: str = "chatterbox",
        reference_audio_paths: Optional[list[str]] = None,
        voice_description: Optional[str] = None,
        voice: Optional[str] = None,
        filename_prefix: str = "segment",
    ) -> dict:
        """Generate TTS for multiple text segments.

        Args:
            segments: List of dicts with 'text' key (and optional 'filename').
            output_dir: Directory to save output files.
            engine: TTS engine to use.
            reference_audio_paths: Reference audio for cloning engines.
            voice_description: Voice description for maya1.
            voice: Voice ID for preset voice engines.
            filename_prefix: Prefix for auto-generated filenames.

        Returns list of results for each segment.
        """
        result = batch_generate(
            segments=segments,
            output_dir=resolve_output_path(output_dir),
            engine=engine,
            reference_audio_paths=reference_audio_paths,
            voice_description=voice_description,
            voice=voice,
            filename_prefix=filename_prefix,
        )
        return to_dict(result)

    @mcp.tool()
    def get_tts_model_status() -> dict:
        """Get download status of Maya1 and SNAC models."""
        return check_maya1_models()

    @mcp.tool()
    def download_tts_models(force: bool = False) -> dict:
        """Download Maya1 TTS models (~10GB total).

        Args:
            force: Re-download even if models exist.
        """
        return download_maya1_models(force=force)

    @mcp.tool()
    def get_tts_engine_info() -> dict:
        """Get info about all TTS engines and their parameters."""
        return {
            "engines": ENGINE_INFO,
            "emotion_tags": {
                "maya1": "<laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited>, <sad>",
                "chatterbox": "[laugh], [sigh], [gasp], [chuckle], [cough], [hmm], [um], [oh]",
                "cosyvoice3": "[breath]",
            },
            "usage": "Use speak(text, output_path, engine='engine_name', ...params)",
        }
