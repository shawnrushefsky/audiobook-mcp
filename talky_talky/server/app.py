#!/usr/bin/env python3
"""Talky Talky - Text-to-Speech, Speech-to-Text, and Audio Asset MCP Server for AI Agents.

This MCP server provides TTS, transcription, and audio asset management capabilities.

TTS Engines:
- Maya1: Text-prompted voice design (describe the voice you want)
- Chatterbox: Audio-prompted voice cloning (clone from reference audio)
- Chatterbox Turbo: Fast voice cloning optimized for production
- MiraTTS: Fast voice cloning with high-quality 48kHz output
- XTTS-v2: Multilingual voice cloning with 17 language support
- Kokoro: Voice selection from 54 pre-built voices across 8 languages
- Soprano: Ultra-fast CUDA TTS with 2000x realtime speed
- VibeVoice Realtime: Real-time TTS with ~300ms latency (Microsoft)
- VibeVoice Long-form: Long-form multi-speaker TTS up to 90 minutes (Microsoft)
- CosyVoice3: Zero-shot multilingual voice cloning with 9 languages (Alibaba)

Transcription Engines:
- Whisper: OpenAI's robust speech recognition (99+ languages)
- Faster-Whisper: CTranslate2-optimized Whisper (4x faster)

Audio Asset Management:
- Local folder indexing with auto-tagging
- Freesound.org integration for Creative Commons licensed sounds
- Unified search across all sources
- Tag management and organization

Plus audio utilities for format conversion and concatenation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Import configuration utilities
from .config import (
    DEFAULT_OUTPUT_DIR,
    _load_config,
    _save_config,
    get_output_dir,
    resolve_output_path,
    to_dict,
)

# Import TTS module
from ..tools.tts import (
    check_tts,
    get_tts_info,
    get_available_engines,
    list_engines,
    generate,
    batch_generate,
)
from ..tools.tts.maya1 import (
    check_models_downloaded as check_maya1_models,
    download_models as download_maya1_models,
)

# Import audio utilities
from ..tools.audio import (
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
    # Voice modulation
    shift_pitch,
    stretch_time,
    apply_voice_effect,
    shift_formant,
    VOICE_EFFECTS,
    # Batch and new utilities
    batch_normalize_audio,
    generate_silence,
    loop_audio_to_duration,
    # SFX mixing tools
    normalize_to_lufs,
    get_mean_level,
    compare_levels,
    overlay_multiple,
    batch_normalize_to_lufs,
)

# Import autotune module
from ..tools.autotune import (
    autotune as autotune_audio,
    detect_pitch as detect_audio_pitch,
    list_keys as get_autotune_keys,
    list_scales as get_autotune_scales,
)

# Import transcription module
from ..tools.transcription import (
    check_transcription,
    get_transcription_info,
    get_available_engines as get_available_transcription_engines,
    list_engines as list_transcription_engines,
    transcribe,
)

# Import analysis module
from ..tools.analysis import (
    detect_emotion,
    compare_voices,
    get_voice_embedding,
    assess_quality,
    list_emotion_engines,
    list_similarity_engines,
    list_quality_engines,
    to_dict as analysis_to_dict,
    # SFX analysis
    analyze_loudness,
    detect_clipping,
    analyze_spectrum,
    detect_silence,
    validate_format,
    get_sfx_analysis_info,
    # TTS verification
    detect_spoken_tags,
    compare_audio_to_text,
    convert_tags_for_engine,
    # Speech boundary detection
    detect_speech_onset,
    detect_truncated_audio,
    verify_segment_boundaries,
    trim_to_speech_with_padding,
)

# Import song generation module
from ..tools.songgen import (
    check_songgen,
    get_info as get_songgen_info,
    get_available_engines as get_available_songgen_engines,
    generate as generate_song,
)
from ..tools.songgen.levo import (
    check_models_downloaded as check_songgen_models,
    download_models as download_songgen_models_impl,
    MODELS as SONGGEN_MODELS,
    STRUCTURE_MARKERS as SONGGEN_STRUCTURE_MARKERS,
    STYLE_PROMPTS as SONGGEN_STYLE_PROMPTS,
)
from ..tools.songgen.acestep import (
    check_models_downloaded as check_acestep_models,
    download_models as download_acestep_models_impl,
)

# Import assets module
from ..tools.assets import (
    search_assets as search_assets_async,
    get_asset as get_asset_async,
    download_asset as download_asset_async,
    import_folder as import_folder_async,
    list_sources,
    configure_freesound,
    configure_jamendo,
    set_asset_library_path,
    get_asset_library_path,
    add_tags,
    remove_tags,
    list_tags,
    list_indexed_folders as list_indexed_folders_async,
    rescan_folder as rescan_folder_async,
    remove_indexed_folder as remove_indexed_folder_async,
    auto_tag_asset as auto_tag_asset_async,
    get_autotag_capabilities,
)


# Initialize MCP server
mcp = FastMCP("talky-talky")


# ============================================================================
# TTS Engine Tools
# ============================================================================


@mcp.tool()
def check_tts_availability() -> dict:
    """Check if TTS engines are available and properly configured.

    Returns detailed status including:
    - Available engines
    - Device info (CUDA/MPS/CPU)
    - Setup instructions for unavailable engines
    """
    status = check_tts()
    return to_dict(status)


@mcp.tool()
def get_tts_engines_info() -> dict:
    """Get detailed information about all TTS engines.

    Returns info for each engine including:
    - Name, type, and description
    - Requirements and availability
    - Supported emotion tags and format
    - Engine-specific parameters
    """
    return get_tts_info()


@mcp.tool()
def list_available_engines() -> dict:
    """List TTS engines that are currently available (installed).

    Returns:
        Dict with list of available engine IDs and their basic info.
    """
    available = get_available_engines()
    engines = list_engines()

    return {
        "available_engines": available,
        "engines": {
            engine_id: {
                "name": info.name,
                "type": info.engine_type,
                "description": info.description,
            }
            for engine_id, info in engines.items()
            if engine_id in available
        },
    }


@mcp.tool()
def get_tts_model_status() -> dict:
    """Get the download status of TTS models (Maya1 and SNAC).

    Returns information about which models are downloaded and their cache locations.
    """
    return check_maya1_models()


@mcp.tool()
def download_tts_models(force: bool = False) -> dict:
    """Download Maya1 TTS model weights from HuggingFace.

    Downloads both the Maya1 language model and SNAC audio codec.
    This may take a while depending on your internet connection (~10GB total).

    Args:
        force: If True, re-download even if models exist in cache.

    Returns:
        Status dict with download results for each model.
    """
    return download_maya1_models(force=force)


# ============================================================================
# Speech Generation Tools
# ============================================================================


@mcp.tool()
def speak_maya1(
    text: str,
    output_path: str,
    voice_description: str,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> dict:
    """Generate speech using Maya1 (text-prompted voice design).

    Creates audio from text using a natural language voice description.

    IMPORTANT - USE EMOTION TAGS for expressive speech! Maya1 supports inline tags:
    - <laugh> or <chuckle> - laughter
    - <sigh> - sighing
    - <gasp> - gasping/surprise
    - <whisper> - whispering
    - <angry> - angry tone
    - <excited> - excited tone
    - <sad> - sad tone

    Example text WITH emotion tags (RECOMMENDED):
        "Oh my goodness! <gasp> I can't believe it! <laugh> This is amazing!"
        "I'm so sorry... <sigh> I didn't mean to hurt you. <sad>"
        "<whisper> Don't tell anyone, but... <excited> we won!"

    Args:
        text: The text to synthesize. STRONGLY RECOMMENDED: Include emotion tags
            like <laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited> to make
            speech more natural and expressive. Place tags where the emotion occurs.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        voice_description: Natural language description of the voice.
            Example: "Gruff male pirate, 50s, British accent, slow pacing"
        temperature: Sampling temperature (0.1-1.0, default 0.4). Lower = more stable.
        top_p: Nucleus sampling parameter (0.1-1.0, default 0.9).
        repetition_penalty: Penalty for repetition (1.0-2.0, default 1.1).

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="maya1",
        voice_description=voice_description,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return to_dict(result)


@mcp.tool()
def speak_chatterbox(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> dict:
    """Generate speech using Chatterbox (audio-prompted voice cloning).

    Clones a voice from reference audio samples with emotion control.

    IMPORTANT - USE PARALINGUISTIC TAGS for natural speech! Known tags include:
    - [laugh] - laughter
    - [chuckle] - soft laughter
    - [cough] - coughing
    - [sigh] - sighing
    - [gasp] - gasping
    - [groan], [yawn], [sniff], [clearing throat], etc. - TRY OTHERS!

    Chatterbox supports MORE tags than documented - experiment with natural sounds
    like [hmm], [uh], [um], [oh], [ah], [wow], [ooh], [eww], [huh], [mhm], etc.

    Example text WITH tags (RECOMMENDED):
        "That's hilarious! [laugh] I can't stop laughing!"
        "Well... [sigh] I suppose you're right. [hmm] Let me think."
        "[gasp] You scared me! [chuckle] Don't do that again."
        "So, [um], I was thinking... [clearing throat] we should go."

    Args:
        text: The text to synthesize. STRONGLY RECOMMENDED: Include paralinguistic
            tags like [laugh], [sigh], [gasp], [chuckle], [cough] and experiment
            with others like [hmm], [um], [oh], [yawn], etc. to make speech more
            natural and human-like. Place tags where the sound should occur.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 10+ seconds of clear speech recommended.
        exaggeration: Controls speech expressiveness (0.0-1.0+, default 0.5).
            0.0 = flat/monotone, 0.5 = natural, 0.6-0.8 = expressive (recommended),
            0.8+ = dramatic. Note: Default 0.5 can sound flat; try 0.6-0.7 for
            natural expressiveness.
        cfg_weight: Controls pacing/adherence to reference (0.0-1.0, default 0.5).
            Lower values = slower, more deliberate speech.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="chatterbox",
        reference_audio_paths=reference_audio_paths,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    return to_dict(result)


@mcp.tool()
def speak_mira(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
) -> dict:
    """Generate speech using MiraTTS (fast audio-prompted voice cloning).

    Fast voice cloning with high-quality 48kHz output.
    Over 100x realtime performance with only 6GB VRAM.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. Clear speech samples work best.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: MiraTTS does not support emotion tags but produces high-quality 48kHz audio.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="mira",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)


@mcp.tool()
def speak_xtts(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
    language: str = "en",
) -> dict:
    """Generate speech using XTTS-v2 (multilingual voice cloning).

    Multilingual voice cloning supporting 17 languages with cross-language cloning.
    Only requires ~6 seconds of reference audio.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 6+ seconds of clear speech recommended.
        language: Target language code (default: "en").
            Supported: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: XTTS-v2 supports cross-language cloning - clone a voice from English
    audio and generate speech in Japanese, for example.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="xtts",
        reference_audio_paths=reference_audio_paths,
        language=language,
    )
    return to_dict(result)


@mcp.tool()
def speak_kokoro(
    text: str,
    output_path: str,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> dict:
    """Generate speech using Kokoro (voice selection from 54 pre-built voices).

    Lightweight, fast TTS with 54 high-quality voices across 8 languages.
    No voice cloning needed - select from pre-built voices.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        voice: Voice ID to use (default: "af_heart").
            Format: [lang][gender]_[name]
            Examples: af_heart (American Female Heart), bm_george (British Male George)
            Languages: a=American, b=British, j=Japanese, z=Mandarin, e=Spanish,
                      f=French, h=Hindi, i=Italian, p=Portuguese
        speed: Speech rate multiplier (default: 1.0). Range: 0.5-2.0.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Popular voices:
    - af_heart, af_bella (American English female, quality A)
    - am_fenrir, am_michael (American English male, quality B)
    - bf_emma, bm_george (British English)
    - jf_alpha, jm_kumo (Japanese)
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="kokoro",
        voice=voice,
        speed=speed,
    )
    return to_dict(result)


@mcp.tool()
def speak_soprano(
    text: str,
    output_path: str,
    temperature: float = 0.3,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> dict:
    """Generate speech using Soprano (ultra-fast CUDA TTS).

    Ultra-lightweight 80M model with 2000x realtime speed and 32kHz output.
    Requires NVIDIA GPU with CUDA. No voice selection or cloning.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        temperature: Sampling temperature (default: 0.3). Lower = more consistent.
        top_p: Nucleus sampling parameter (default: 0.95).
        repetition_penalty: Penalty for repetition (default: 1.2).

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: Soprano requires CUDA GPU. CPU and MPS are not supported.
    Best for batch processing where speed is critical.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="soprano",
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return to_dict(result)


@mcp.tool()
def speak_chatterbox_turbo(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
) -> dict:
    """Generate speech using Chatterbox Turbo (fast voice cloning).

    Streamlined 350M model optimized for low-latency voice cloning.
    Faster than standard Chatterbox with simpler API (no tuning parameters).

    IMPORTANT - USE PARALINGUISTIC TAGS for natural speech! Known tags include:
    - [laugh] - laughter
    - [chuckle] - soft laughter
    - [cough] - coughing
    - [sigh], [gasp], [groan], [yawn], [sniff], etc. - TRY OTHERS!

    Chatterbox supports MORE tags than documented - experiment with natural sounds
    like [hmm], [uh], [um], [oh], [ah], [wow], [ooh], [eww], [huh], [mhm], etc.

    Example text WITH tags (RECOMMENDED):
        "That joke was great! [laugh] Tell me another one."
        "Sorry, I'm getting over a cold. [cough] Where were we?"
        "[hmm] Let me think about that... [oh] I've got it!"

    Args:
        text: The text to synthesize. STRONGLY RECOMMENDED: Include paralinguistic
            tags like [laugh], [chuckle], [cough], [sigh] and experiment with others
            like [hmm], [um], [oh], [yawn], etc. to make speech more natural and
            human-like. Place tags where the sound should occur.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 10+ seconds of clear speech recommended.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: For more control over expressiveness and pacing, use speak_chatterbox
    which has exaggeration and cfg_weight parameters.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="chatterbox_turbo",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)


@mcp.tool()
def speak_vibevoice_realtime(
    text: str,
    output_path: str,
    speaker_name: str = "Carter",
) -> dict:
    """Generate speech using VibeVoice Realtime (fast single-speaker TTS).

    Microsoft's real-time TTS with ~300ms first-audio latency.
    Single speaker, up to 10 minutes per generation.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        speaker_name: Name of the speaker voice to use (default: "Carter").
            Available speakers: Carter, Emily, Nova, Michael, Sarah

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: Primarily supports English. Other languages are experimental.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="vibevoice_realtime",
        voice_description=speaker_name,
    )
    return to_dict(result)


@mcp.tool()
def speak_vibevoice_longform(
    text: str,
    output_path: str,
    speaker_name: str = "Carter",
    speakers: Optional[list[str]] = None,
) -> dict:
    """Generate speech using VibeVoice Long-form (multi-speaker TTS).

    Microsoft's long-form TTS supporting up to 90 minutes and 4 speakers.
    Ideal for podcasts, audiobooks, and conversations.

    Args:
        text: The text to synthesize. Can include speaker labels for multi-speaker.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        speaker_name: Primary speaker name for single-speaker generation (default: "Carter").
        speakers: List of speaker names for multi-speaker generation (max 4).
            If provided, overrides speaker_name.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: Supports English and Chinese. Use speaker labels in text for multi-speaker.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="vibevoice_longform",
        voice_description=speaker_name,
        speakers=speakers,
    )
    return to_dict(result)


@mcp.tool()
def speak_cosyvoice(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
    prompt_text: Optional[str] = None,
    instruction: Optional[str] = None,
    language: str = "auto",
) -> dict:
    """Generate speech using CosyVoice3 (multilingual voice cloning).

    Alibaba's zero-shot voice cloning with 9 languages and instruction control.
    Excellent for multilingual content and dialect control.

    IMPORTANT - USE [breath] TAGS for natural speech! CosyVoice supports:
    - [breath] - natural breathing sounds between phrases

    Example text WITH tags (RECOMMENDED):
        "Hello everyone. [breath] Welcome to today's presentation."
        "That was a long journey... [breath] I'm glad we finally made it."
        "Let me think about that. [breath] Yes, I believe you're right."

    Args:
        text: The text to synthesize. STRONGLY RECOMMENDED: Include [breath] tags
            between sentences or phrases for more natural, human-like speech.
            Place tags where a natural pause/breath would occur.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 5-10 seconds of clear speech recommended.
        prompt_text: Transcript of reference audio (improves quality).
        instruction: Natural language instruction for style control.
            Examples: "请用广东话表达。" (Cantonese), "请用尽可能快地语速说。" (fast speed)
        language: Target language code (default: "auto" for auto-detection).
            Supported: zh, en, ja, ko, de, es, fr, it, ru

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Features:
    - 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
    - 18+ Chinese dialects via instruction control
    - Cross-lingual cloning (clone voice in one language, output in another)
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="cosyvoice3",
        reference_audio_paths=reference_audio_paths,
        prompt_text=prompt_text,
        instruction=instruction,
        language=language,
    )
    return to_dict(result)


@mcp.tool()
def speak_seamlessm4t(
    text: str,
    output_path: str,
    language: str = "en",
    src_language: Optional[str] = None,
    speaker_id: int = 0,
) -> dict:
    """Generate speech using SeamlessM4T v2 (multilingual TTS with translation).

    Meta's 2.3B parameter multilingual model supporting 35 languages for speech
    output with optional translation. Can translate text while generating speech.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        language: Target language code for speech output (default: "en").
            Supported: en, es, fr, de, it, pt, pl, nl, ru, uk, tr, ar, zh, ja, ko,
            hi, bn, th, vi, id, ms, tl, sw, he, fa, ro, hu, cs, el, sv, da, fi, no, sk, bg
        src_language: Source text language code (default: same as language).
            Set different from language to translate while synthesizing speech.
            Example: src_language="en", language="fr" translates English to French speech.
        speaker_id: Speaker voice index (0-199, default: 0).
            Different IDs produce different voice characteristics.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Features:
    - 35 languages for speech output
    - 200 different speaker voices
    - Translation + TTS in one step (set different src_language and language)
    - High quality multilingual synthesis

    License: CC-BY-NC-4.0 (non-commercial use only)

    Examples:
        # Pure English TTS
        speak_seamlessm4t("Hello world", "hello.wav", language="en")

        # Spanish TTS
        speak_seamlessm4t("Hola mundo", "hola.wav", language="es")

        # Translate English to French speech
        speak_seamlessm4t("Hello world", "bonjour.wav",
                         src_language="en", language="fr")
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="seamlessm4t",
        language=language,
        src_language=src_language,
        speaker_id=speaker_id,
    )
    return to_dict(result)


# ============================================================================
# Song Generation Tools
# ============================================================================


@mcp.tool()
def check_songgen_availability() -> dict:
    """Check if song generation engines are available and properly configured.

    Returns detailed status including:
    - Available engines
    - Device info (CUDA required)
    - Setup instructions for unavailable engines

    Note: Song generation requires CUDA GPU with 10-28GB VRAM.
    """
    return check_songgen()


@mcp.tool()
def get_songgen_engines_info() -> dict:
    """Get detailed information about all song generation engines.

    Returns info for each engine including:
    - Name and description
    - Requirements and availability
    - Supported languages and max duration
    - Lyrics format guide
    """
    engines = {}
    for engine_id in get_available_songgen_engines():
        info = get_songgen_info(engine_id)
        engines[engine_id] = to_dict(info)
    return {"engines": engines}


@mcp.tool()
def list_available_songgen_engines() -> dict:
    """List song generation engines that are currently available (installed).

    Returns:
        Dict with list of available engine IDs and their basic info.
    """
    return {
        "engines": get_available_songgen_engines(),
        "note": "Song generation requires CUDA GPU (10-28GB VRAM depending on model)",
    }


@mcp.tool()
def get_songgen_model_status(model_name: str = "base-new") -> dict:
    """Get the download status of song generation models.

    Args:
        model_name: Model variant to check ("base-new", "base-full", "large").

    Returns:
        Dict with download status for runtime and model weights.
    """
    return check_songgen_models(model_name)


@mcp.tool()
def download_songgen_models(model_name: str = "base-new", force: bool = False) -> dict:
    """Download song generation model weights from HuggingFace.

    Downloads both the runtime dependencies and model weights.
    This may take a while depending on your internet connection (~10GB total).

    Args:
        model_name: Model variant to download:
            - "base-new": 2m30s max, 10GB VRAM, Chinese/English (default)
            - "base-full": 4m30s max, 12GB VRAM, Chinese/English
            - "large": 4m30s max, 22GB VRAM, best quality
        force: If True, re-download even if models exist in cache.

    Returns:
        Status dict with download results.
    """
    return download_songgen_models_impl(model_name, force=force)


@mcp.tool()
def get_songgen_lyrics_format() -> dict:
    """Get the lyrics format guide for song generation.

    Returns documentation on how to format lyrics with structure markers,
    section separators, and examples of properly formatted lyrics.
    """
    return {
        "overview": (
            "Lyrics use structure markers to define song sections. "
            "Separate sections with ';' and sentences within sections with '.'"
        ),
        "structure_markers": SONGGEN_STRUCTURE_MARKERS,
        "separator": ";",
        "sentence_separator": ".",
        "examples": [
            {
                "name": "Pop song structure",
                "lyrics": (
                    "[intro-short] ; "
                    "[verse] Hello world. I'm singing today ; "
                    "[chorus] This is the chorus. Sing along with me ; "
                    "[verse] Second verse now. Different words here ; "
                    "[chorus] This is the chorus. Sing along with me ; "
                    "[outro-short]"
                ),
                "description": "female, pop, happy, upbeat",
            },
            {
                "name": "Ballad",
                "lyrics": (
                    "[verse] Memories fade away. Like leaves in autumn wind ; "
                    "[chorus] But I still remember you. Your voice echoes in my mind ; "
                    "[bridge] Time moves on. Hearts grow cold ; "
                    "[chorus] But I still remember you ; "
                    "[outro]"
                ),
                "description": "male, ballad, sad, piano, slow",
            },
        ],
        "style_prompts": SONGGEN_STYLE_PROMPTS,
        "model_options": {k: v["description"] for k, v in SONGGEN_MODELS.items()},
    }


@mcp.tool()
def generate_song_levo(
    lyrics: str,
    output_path: str,
    description: str = "female, pop, happy",
    generate_type: str = "mixed",
    prompt_audio_path: Optional[str] = None,
    auto_prompt_style: Optional[str] = None,
    model_name: str = "base-new",
    low_mem: bool = False,
) -> dict:
    """Generate a song using LeVo (Tencent's SongGeneration).

    Creates complete songs with vocals and accompaniment from structured lyrics.
    Supports Chinese and English, up to 4.5 minutes depending on model.

    IMPORTANT - Lyrics Format:
    Use structure markers to define song sections. Separate sections with ';'
    and sentences within lyrical sections with '.'.

    Structure markers:
    - [intro], [intro-short]: Instrumental introduction
    - [verse]: Main verses
    - [pre-chorus]: Build-up before chorus
    - [chorus]: Main hook/chorus
    - [bridge]: Contrasting section
    - [outro], [outro-short]: Ending
    - [interlude]: Instrumental break

    Example lyrics:
        "[intro-short] ; [verse] Hello world. I'm singing today ; "
        "[chorus] This is the chorus. Sing along with me ; [outro-short]"

    Args:
        lyrics: Structured lyrics with section markers (see format above).
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "song.wav") which saves to the configured output directory.
        description: Musical style description. Comma-separated attributes like:
            "female, pop, happy, piano" or "male, rock, aggressive, guitar, fast".
        generate_type: Output type:
            - "mixed": Combined vocals and accompaniment (default)
            - "separate": Creates 3 files: mixed, vocals-only, and BGM-only
            - "vocal": Vocals only (a cappella)
            - "bgm": Accompaniment only (instrumental/karaoke)
        prompt_audio_path: Optional path to ~10s reference audio for style cloning.
        auto_prompt_style: Auto-select reference style instead of audio file.
            Options: "Pop", "Rock", "Jazz", "R&B", "Electronic", "Folk", "Classical", "Hip-Hop", "Auto"
        model_name: Model variant to use:
            - "base-new": 2m30s max, 10GB VRAM (default)
            - "base-full": 4m30s max, 12GB VRAM
            - "large": 4m30s max, 22GB VRAM, best quality
        low_mem: Enable low-memory mode (slower but uses less VRAM).

    Returns:
        Dict with status, output_path, duration_ms, and for "separate" mode
        also includes vocal_path and bgm_path.

    Note: Requires CUDA GPU with 10-28GB VRAM depending on model.
    First run will download ~10GB of model weights.
    """
    result = generate_song(
        engine_id="levo",
        lyrics=lyrics,
        output_path=resolve_output_path(output_path),
        description=description,
        generate_type=generate_type,
        prompt_audio_path=prompt_audio_path,
        auto_prompt_style=auto_prompt_style,
        model_name=model_name,
        low_mem=low_mem,
    )
    return to_dict(result)


@mcp.tool()
def get_acestep_model_status() -> dict:
    """Get the download status of ACE-Step models.

    Returns:
        Dict with download status and checkpoint directory location.
    """
    return check_acestep_models()


@mcp.tool()
def download_acestep_models(force: bool = False) -> dict:
    """Download ACE-Step model weights from HuggingFace.

    Downloads the ACE-Step 3.5B model (~7GB).
    Note: Models also auto-download on first generation.

    Args:
        force: If True, re-download even if models exist in cache.

    Returns:
        Status dict with download results.
    """
    return download_acestep_models_impl(force=force)


@mcp.tool()
def generate_song_acestep(
    prompt: str,
    output_path: str,
    lyrics: Optional[str] = None,
    audio_duration: float = 60.0,
    infer_steps: int = 27,
    guidance_scale: float = 15.0,
    scheduler_type: str = "euler",
    seed: Optional[int] = None,
    cpu_offload: bool = False,
    quantized: bool = False,
) -> dict:
    """Generate a song using ACE-Step (supports Apple Silicon MPS + CUDA).

    Creates complete songs with vocals from style prompts and optional lyrics.
    ACE-Step is a 3.5B parameter foundation model that works on both Apple Silicon
    (MPS with 36GB+ unified memory) and NVIDIA GPUs.

    Args:
        prompt: Style description as comma-separated tags.
            Examples: "female vocals, pop, upbeat, synth, drums"
                     "male vocals, rock, energetic, electric guitar"
                     "instrumental, jazz, piano, smooth, relaxing"
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "song.wav") which saves to the configured output directory.
        lyrics: Optional song lyrics with structure markers.
            Structure markers: [verse], [chorus], [bridge], [intro], [outro], etc.
            Example:
                "[verse]\\nWalking down the street today\\nFeeling good in every way\\n\\n"
                "[chorus]\\nThis is my moment\\nNothing can stop me now"
        audio_duration: Duration in seconds (max 240, default 60).
        infer_steps: Number of inference steps.
            - 27: Fast generation (default, good quality)
            - 60: Higher quality, slower
        guidance_scale: Classifier-free guidance strength (default 15.0).
            Higher values = stronger adherence to prompt.
        scheduler_type: Diffusion scheduler type.
            - "euler": Default, fast and stable
            - "heun": Higher quality, slower
            - "pingpong": Alternative scheduler
        seed: Random seed for reproducibility. None for random.
        cpu_offload: Offload model weights to CPU between steps.
            Enables running on systems with less VRAM/RAM.
        quantized: Use quantized model (lower quality, less memory).
            Enables running on 8GB systems.

    Returns:
        Dict with status, output_path, duration_ms, and generation metadata.

    Performance (Real-Time Factor):
    - RTX 4090: ~34x realtime (27 steps)
    - A100: ~27x realtime (27 steps)
    - M2 Max: ~2.3x realtime (27 steps)

    Note: First run downloads ~7GB of model weights from HuggingFace.
    Apple Silicon requires 36GB+ unified memory (M1/M2/M3 Max/Ultra).
    """
    result = generate_song(
        engine_id="acestep",
        lyrics=prompt,  # ACE-Step uses "prompt" for style, we map it here
        output_path=resolve_output_path(output_path),
        description=lyrics,  # Actual lyrics go here
        audio_duration=audio_duration,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        seed=seed,
        cpu_offload=cpu_offload,
        quantized=quantized,
    )
    return to_dict(result)


# ============================================================================
# Audio Utility Tools
# ============================================================================


@mcp.tool()
def get_audio_file_info(audio_path: str) -> dict:
    """Get information about an audio file.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with path, exists, format, duration_ms, size_bytes, and validity.
    """
    info = get_audio_info(audio_path)
    return to_dict(info)


@mcp.tool()
def convert_audio_format(
    input_path: str,
    output_format: str = "mp3",
    output_path: Optional[str] = None,
) -> dict:
    """Convert an audio file to a different format.

    Args:
        input_path: Path to the input audio file.
        output_format: Target format ('mp3', 'wav', 'm4a'). Default: 'mp3'.
        output_path: Optional output path. If not provided, creates a file
            with the same name but new extension in the same directory.

    Returns:
        Dict with input/output paths, formats, sizes, and compression ratio.
    """
    result = convert_audio(
        input_path=input_path,
        output_format=output_format,
        output_path=output_path,
    )
    return to_dict(result)


@mcp.tool()
def join_audio_files(
    audio_paths: list[str],
    output_path: str,
    output_format: str = "wav",
    gap_ms: float | list[float] = 0,
    resample: bool = True,
    target_sample_rate: Optional[int] = None,
) -> dict:
    """Concatenate multiple audio files into one.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'wav'.
        gap_ms: Silence between segments. Can be:
            - A single number: uniform gap between all segments (e.g., 300)
            - A list of numbers: variable gaps (e.g., [300, 300, 800, 300])
              List length must be len(audio_paths) - 1
            Default: 0 (no gaps).
            Typical values: 300-400ms between dialogue, 800ms+ for scene breaks.
        resample: If True (default), resample all files to a common sample rate
            before joining. This prevents static/noise artifacts from sample rate
            mismatches. Set to False only if you're certain all files match.
        target_sample_rate: Target sample rate when resample=True. If None, uses
            the sample rate of the first file.

    Returns:
        Dict with output_path, input_count, total_duration_ms, and output_format.

    Raises:
        ValueError: If sample rate mismatch detected with resample=False.
    """
    result = concatenate_audio(
        audio_paths=audio_paths,
        output_path=output_path,
        output_format=output_format,
        gap_ms=gap_ms,
        resample=resample,
        target_sample_rate=target_sample_rate,
    )
    return to_dict(result)


@mcp.tool()
def normalize_audio_levels(
    input_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """Normalize audio levels to broadcast standard (-16 LUFS).

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_normalized' suffix.

    Returns:
        Dict with input_path, output_path, and duration_ms.
    """
    result = normalize_audio(input_path=input_path, output_path=output_path)
    return to_dict(result)


@mcp.tool()
def resample_audio_file(
    input_path: str,
    output_path: Optional[str] = None,
    target_sample_rate: int = 44100,
) -> dict:
    """Resample audio to a target sample rate.

    Use this to convert between sample rates without other processing.
    Essential for ensuring compatibility when joining audio files from
    different sources (e.g., TTS output at 24kHz with effects at 48kHz).

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_resampled' suffix.
        target_sample_rate: Target sample rate in Hz. Common values:
            - 24000: Common TTS output rate (Chatterbox, Maya1)
            - 44100: CD quality, general audio
            - 48000: Professional video/broadcast

    Returns:
        Dict with input_path, output_path, original_sample_rate, new_sample_rate,
        and duration_ms.

    Example:
        # Resample voice effect output to match TTS output
        resample_audio_file("effect_192k.wav", "effect_24k.wav", target_sample_rate=24000)
    """
    result = resample_audio(
        input_path=input_path,
        output_path=output_path,
        target_sample_rate=target_sample_rate,
    )
    return to_dict(result)


@mcp.tool()
def check_audio_compatibility(audio_paths: list[str]) -> dict:
    """Check if multiple audio files are compatible for joining.

    Validates sample rates and channel counts to ensure files can be
    concatenated without artifacts or corruption. Use this before
    join_audio_files to detect potential issues.

    Args:
        audio_paths: List of audio file paths to check.

    Returns:
        Dict with:
        - compatible: True if all files are compatible
        - files_checked: Number of files checked
        - issues: List of compatibility issues found
        - sample_rates: Dict mapping each file to its sample rate
        - channels: Dict mapping each file to its channel count
        - recommendation: Suggested fix if incompatible

    Example:
        result = check_audio_compatibility(["a.wav", "b.wav", "c.wav"])
        if not result["compatible"]:
            print("Issues:", result["issues"])
            print("Fix:", result["recommendation"])
    """
    result = validate_audio_compatibility(audio_paths=audio_paths)
    return to_dict(result)


@mcp.tool()
def set_output_directory(directory: str) -> dict:
    """Set the default directory where audio files will be saved.

    When generating audio, if you provide just a filename (e.g., "speech.wav")
    instead of a full path, it will be saved to this directory.

    Args:
        directory: Path to the directory for saving audio files.
            Use "default" to reset to ~/Documents/talky-talky.

    Returns:
        Dict with status, the configured directory path, and whether it was created.
    """
    if directory.lower() == "default":
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(directory).expanduser().resolve()

    # Create the directory if it doesn't exist
    created = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to config
    config = _load_config()
    config["output_directory"] = str(output_dir)
    _save_config(config)

    return {
        "status": "success",
        "output_directory": str(output_dir),
        "created": created,
        "message": f"Audio files will be saved to: {output_dir}",
    }


@mcp.tool()
def get_output_directory() -> dict:
    """Get the current default directory where audio files are saved.

    Returns:
        Dict with the current output directory path and whether it exists.
    """
    output_dir = get_output_dir()
    return {
        "output_directory": str(output_dir),
        "exists": output_dir.exists(),
        "default": str(DEFAULT_OUTPUT_DIR),
        "is_default": str(output_dir) == str(DEFAULT_OUTPUT_DIR),
    }


@mcp.tool()
def check_ffmpeg_available() -> dict:
    """Check if ffmpeg is installed and available.

    ffmpeg is required for audio format conversion and concatenation.

    Returns:
        Dict with available status and install instructions if not available.
    """
    available = is_ffmpeg_available()
    return {
        "available": available,
        "message": "ffmpeg is available"
        if available
        else "ffmpeg not found. Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)",
    }


@mcp.tool()
def play_audio(audio_path: str) -> dict:
    """Play an audio file using the system's default audio player.

    Opens the audio file with the platform's default application for audio playback.
    This is useful for previewing generated TTS audio.

    Args:
        audio_path: Path to the audio file to play.

    Returns:
        Dict with status and message indicating success or failure.

    Platform behavior:
    - macOS: Uses 'open' command (opens in default app like QuickTime/Music)
    - Linux: Uses 'xdg-open' command (opens in default audio player)
    - Windows: Uses 'start' command (opens in default app like Windows Media Player)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {
            "status": "error",
            "message": f"Audio file not found: {audio_path}",
        }

    try:
        if sys.platform == "darwin":
            # macOS
            subprocess.Popen(["open", str(audio_path)])
        elif sys.platform == "win32":
            # Windows
            subprocess.Popen(["start", "", str(audio_path)], shell=True)
        else:
            # Linux and other Unix-like systems
            subprocess.Popen(["xdg-open", str(audio_path)])

        return {
            "status": "success",
            "message": f"Opened {audio_path.name} in default audio player",
            "path": str(audio_path),
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Could not find system command to open audio: {e}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to play audio: {e}",
        }


@mcp.tool()
def trim_audio_file(
    input_path: str,
    output_path: Optional[str] = None,
    start_ms: Optional[float] = None,
    end_ms: Optional[float] = None,
    padding_ms: float = 50,
) -> dict:
    """Trim audio to specified boundaries or auto-detect content boundaries.

    If start_ms and end_ms are not provided, uses silence detection to
    automatically find content boundaries and trims to those. This is
    ideal for removing variable leading/trailing silence from TTS output.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_trimmed' suffix in the same directory.
        start_ms: Start time in milliseconds (None = auto-detect from silence).
        end_ms: End time in milliseconds (None = auto-detect from silence).
        padding_ms: Milliseconds of silence to keep as buffer when auto-detecting.
            Default: 50ms. Set to 0 for tight trim, higher for more natural pauses.

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Trimmed file path
        - original_duration_ms: Original file duration
        - trimmed_duration_ms: New file duration
        - start_ms: Trim start point used
        - end_ms: Trim end point used
        - silence_removed_ms: Amount of audio removed
        - auto_detected: Whether boundaries were auto-detected

    Example:
        # Auto-trim TTS output to remove leading/trailing silence
        result = trim_audio_file("tts_output.wav", padding_ms=50)

        # Manual trim to specific segment
        result = trim_audio_file("audio.wav", start_ms=1000, end_ms=5000)
    """
    result = trim_audio(
        input_path=input_path,
        output_path=output_path,
        start_ms=start_ms,
        end_ms=end_ms,
        padding_ms=padding_ms,
    )
    return to_dict(result)


@mcp.tool()
def batch_analyze_silence(
    audio_paths: list[str],
    threshold_db: float = -40.0,
    min_silence_ms: float = 100.0,
) -> dict:
    """Detect silence in multiple audio files at once.

    More efficient than calling detect_audio_silence individually for
    large batches of files (e.g., 200+ audiobook segments).

    Args:
        audio_paths: List of paths to audio files to analyze.
        threshold_db: dB threshold below which audio is silent (default -40).
        min_silence_ms: Minimum duration to count as silence (default 100ms).

    Returns:
        Dict with:
        - status: "success"
        - count: Number of files analyzed
        - results: List of silence analysis for each file, containing:
            - path: The input file path
            - status: "success" or "error"
            - leading_silence_ms: Silence at start
            - trailing_silence_ms: Silence at end
            - content_start_ms: Where actual content begins
            - content_end_ms: Where actual content ends
            - content_duration_ms: Duration of non-silent content

    Example:
        # Analyze all segments in a batch
        result = batch_analyze_silence([
            "segment_001.wav",
            "segment_002.wav",
            "segment_003.wav",
        ])
        for r in result["results"]:
            print(f"{r['path']}: content from {r['content_start_ms']}ms to {r['content_end_ms']}ms")
    """
    results = batch_detect_silence(
        audio_paths=audio_paths,
        threshold_db=threshold_db,
        min_silence_ms=min_silence_ms,
    )
    return {
        "status": "success",
        "count": len(results),
        "results": results,
    }


@mcp.tool()
def batch_trim_audio_files(
    audio_paths: list[str],
    output_dir: Optional[str] = None,
    padding_ms: float = 50,
    suffix: str = "_trimmed",
) -> dict:
    """Trim multiple audio files to content boundaries at once.

    Auto-detects silence at the start and end of each file and trims to content.
    Much more efficient than calling trim_audio_file individually for large batches.

    Args:
        audio_paths: List of paths to audio files to trim.
        output_dir: Directory for output files. If None, outputs are created
            in the same directory as inputs with the suffix appended.
        padding_ms: Milliseconds of silence to keep as buffer. Default: 50ms.
        suffix: Suffix to append to output filenames. Default: "_trimmed".

    Returns:
        Dict with:
        - status: "success", "partial", or "error"
        - total: Total number of files
        - succeeded: Number successfully trimmed
        - failed: Number of failures
        - results: List of results for each file

    Example:
        result = batch_trim_audio_files([
            "segment_001.wav",
            "segment_002.wav",
            "segment_003.wav",
        ], output_dir="/path/to/trimmed/", padding_ms=30)
    """
    results = batch_trim_audio(
        audio_paths=audio_paths,
        output_dir=output_dir,
        padding_ms=padding_ms,
        suffix=suffix,
    )
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - succeeded

    if succeeded == len(results):
        status = "success"
    elif succeeded > 0:
        status = "partial"
    else:
        status = "error"

    return {
        "status": status,
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@mcp.tool()
def batch_normalize_audio_files(
    audio_paths: list[str],
    output_dir: Optional[str] = None,
    suffix: str = "_normalized",
) -> dict:
    """Normalize multiple audio files to broadcast standard (-16 LUFS) at once.

    Much more efficient than calling normalize_audio_levels individually for large batches.

    Args:
        audio_paths: List of paths to audio files to normalize.
        output_dir: Directory for output files. If None, outputs are created
            in the same directory as inputs with the suffix appended.
        suffix: Suffix to append to output filenames. Default: "_normalized".

    Returns:
        Dict with:
        - status: "success", "partial", or "error"
        - total: Total number of files
        - succeeded: Number successfully normalized
        - failed: Number of failures
        - results: List of results for each file

    Example:
        result = batch_normalize_audio_files([
            "chapter_01.wav",
            "chapter_02.wav",
            "chapter_03.wav",
        ], output_dir="/path/to/normalized/")
    """
    results = batch_normalize_audio(
        audio_paths=audio_paths,
        output_dir=output_dir,
        suffix=suffix,
    )
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - succeeded

    if succeeded == len(results):
        status = "success"
    elif succeeded > 0:
        status = "partial"
    else:
        status = "error"

    return {
        "status": status,
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@mcp.tool()
def generate_silence_audio(
    output_path: str,
    duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
) -> dict:
    """Generate a silent audio file of specified duration.

    Creates a pure silence audio file without requiring any input audio.
    Useful for creating gaps, pauses, or placeholder audio.

    Args:
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "silence.wav") which will be saved to the configured
            output directory.
        duration_ms: Duration of silence in milliseconds.
        sample_rate: Sample rate in Hz. Default: 44100.
        channels: Number of audio channels (1=mono, 2=stereo). Default: 2.

    Returns:
        Dict with:
        - status: "success" or "error"
        - output_path: Path to the generated file
        - duration_ms: Duration of the output
        - sample_rate: Sample rate used
        - channels: Number of channels

    Example:
        # Create 2 seconds of stereo silence
        result = generate_silence_audio("pause.wav", duration_ms=2000)

        # Create 500ms mono silence at 24kHz (to match TTS output)
        result = generate_silence_audio("gap.wav", duration_ms=500, sample_rate=24000, channels=1)
    """
    return generate_silence(
        output_path=resolve_output_path(output_path),
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
    )


@mcp.tool()
def loop_audio_to_target_duration(
    input_path: str,
    target_duration_ms: float,
    output_path: Optional[str] = None,
    crossfade_ms: float = 0,
) -> dict:
    """Loop a short audio file to reach a target duration.

    Repeats the audio as many times as needed to reach or exceed the target duration,
    then trims to exactly the target length. Useful for looping ambient sounds or
    background music to match chapter/scene length.

    Args:
        input_path: Path to the input audio file to loop.
        target_duration_ms: Target duration in milliseconds.
        output_path: Optional output path. If not provided, creates a file
            with '_looped' suffix.
        crossfade_ms: Optional crossfade between loops in milliseconds. Default: 0.
            Use 50-200ms for smoother loop transitions.

    Returns:
        Dict with:
        - status: "success" or "error"
        - input_path: Original file path
        - output_path: Looped file path
        - original_duration_ms: Original file duration
        - target_duration_ms: Requested target duration
        - actual_duration_ms: Actual output duration
        - loop_count: Number of times the audio was looped

    Example:
        # Loop 30-second ambient to match 10-minute chapter
        result = loop_audio_to_target_duration(
            "forest_ambience.wav",
            target_duration_ms=600000,  # 10 minutes
            crossfade_ms=100
        )
    """
    return loop_audio_to_duration(
        input_path=input_path,
        output_path=output_path,
        target_duration_ms=target_duration_ms,
        crossfade_ms=crossfade_ms,
    )


@mcp.tool()
def insert_audio_silence(
    input_path: str,
    output_path: Optional[str] = None,
    before_ms: float = 0,
    after_ms: float = 0,
) -> dict:
    """Add silence before and/or after an audio file.

    Useful for adding consistent gaps between segments in audiobook production.
    Use after trimming to add controlled pauses.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_padded' suffix in the same directory.
        before_ms: Milliseconds of silence to add before audio. Default: 0.
        after_ms: Milliseconds of silence to add after audio. Default: 0.

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Padded file path
        - original_duration_ms: Original file duration
        - new_duration_ms: New file duration
        - silence_before_ms: Silence added before
        - silence_after_ms: Silence added after

    Example:
        # Add scene break silence
        result = insert_audio_silence("scene_end.wav", after_ms=800)

        # Add dialogue pause before and after
        result = insert_audio_silence("line.wav", before_ms=300, after_ms=300)
    """
    result = insert_silence(
        input_path=input_path,
        output_path=output_path,
        before_ms=before_ms,
        after_ms=after_ms,
    )
    return to_dict(result)


@mcp.tool()
def crossfade_join_audio(
    audio_paths: list[str],
    output_path: str,
    crossfade_ms: float = 50,
    output_format: str = "wav",
    resample: bool = True,
    target_sample_rate: Optional[int] = None,
) -> dict:
    """Concatenate audio files with smooth crossfade transitions.

    Creates seamless transitions between audio segments by overlapping
    and fading between them. Useful for scene transitions or joining
    audio that would otherwise have abrupt cuts.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        crossfade_ms: Duration of crossfade overlap in milliseconds. Default: 50ms.
            - 20-50ms: Subtle, good for dialogue joins
            - 50-100ms: Noticeable, good for scene transitions
            - 100-200ms: Smooth, good for music transitions
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'wav'.
        resample: If True (default), resample all files to a common sample rate
            before joining. This prevents static/noise artifacts from sample rate
            mismatches. Set to False only if you're certain all files match.
        target_sample_rate: Target sample rate when resample=True. If None, uses
            the sample rate of the first file.

    Returns:
        Dict with:
        - output_path: Path to output file
        - input_count: Number of files joined
        - total_duration_ms: Duration of output
        - crossfade_ms: Crossfade duration used
        - output_format: Format of output

    Example:
        # Join scene segments with smooth transitions
        result = crossfade_join_audio(
            ["scene1.wav", "scene2.wav", "scene3.wav"],
            "chapter.wav",
            crossfade_ms=100
        )
    """
    result = crossfade_join(
        audio_paths=audio_paths,
        output_path=output_path,
        crossfade_ms=crossfade_ms,
        output_format=output_format,
        resample=resample,
        target_sample_rate=target_sample_rate,
    )
    return to_dict(result)


# ============================================================================
# Audio Design Tools
# ============================================================================


@mcp.tool()
def mix_audio_tracks(
    audio_paths: list[str],
    output_path: str,
    volumes: Optional[list[float]] = None,
    normalize: bool = True,
) -> dict:
    """Mix multiple audio tracks together (layer them).

    All input files play simultaneously, mixed into a single output.
    Perfect for layering voice + background music + ambient sounds.

    Args:
        audio_paths: List of audio file paths to mix together.
        output_path: Path for the output mixed file.
        volumes: Optional list of volume multipliers for each track (1.0 = original).
            If not provided, all tracks are mixed at original volume.
            Example: [1.0, 0.3, 0.5] - full voice, 30% music, 50% ambience.
        normalize: If True, normalize the output to prevent clipping (default True).

    Returns:
        Dict with:
        - output_path: Path to mixed output file
        - input_count: Number of tracks mixed
        - duration_ms: Duration of output
        - normalized: Whether normalization was applied

    Example:
        # Layer voice over background music
        result = mix_audio_tracks(
            ["narration.wav", "music.wav"],
            "scene.wav",
            volumes=[1.0, 0.2]  # Full narration, 20% music
        )

        # Mix three layers: voice + music + ambience
        result = mix_audio_tracks(
            ["voice.wav", "music.wav", "rain.wav"],
            "final.wav",
            volumes=[1.0, 0.3, 0.4]
        )
    """
    result = mix_audio(
        audio_paths=audio_paths,
        output_path=output_path,
        volumes=volumes,
        normalize=normalize,
    )
    return to_dict(result)


@mcp.tool()
def adjust_audio_volume(
    input_path: str,
    output_path: Optional[str] = None,
    volume: float = 1.0,
    volume_db: Optional[float] = None,
) -> dict:
    """Adjust the volume of an audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_vol' suffix in the same directory.
        volume: Volume multiplier (1.0 = original, 2.0 = double, 0.5 = half).
            Ignored if volume_db is specified.
        volume_db: Volume adjustment in dB (overrides volume if specified).
            Positive = louder, negative = quieter.
            +6dB = roughly 2x louder, -6dB = roughly half.

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Adjusted file path
        - duration_ms: Duration of output
        - volume_change: Description of change applied

    Example:
        # Double the volume
        result = adjust_audio_volume("quiet.wav", volume=2.0)

        # Reduce by 6dB
        result = adjust_audio_volume("loud.wav", volume_db=-6)
    """
    result = adjust_volume(
        input_path=input_path,
        output_path=output_path,
        volume=volume,
        volume_db=volume_db,
    )
    return to_dict(result)


@mcp.tool()
def apply_audio_fade(
    input_path: str,
    output_path: Optional[str] = None,
    fade_in_ms: float = 0,
    fade_out_ms: float = 0,
) -> dict:
    """Apply fade in and/or fade out to audio.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_faded' suffix in the same directory.
        fade_in_ms: Duration of fade in at start in milliseconds (0 = no fade in).
        fade_out_ms: Duration of fade out at end in milliseconds (0 = no fade out).

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Faded file path
        - duration_ms: Duration of output
        - fade_in_ms: Fade in duration applied
        - fade_out_ms: Fade out duration applied

    Example:
        # Fade in at start
        result = apply_audio_fade("audio.wav", fade_in_ms=500)

        # Fade out at end
        result = apply_audio_fade("audio.wav", fade_out_ms=1000)

        # Both fade in and out
        result = apply_audio_fade("audio.wav", fade_in_ms=200, fade_out_ms=500)
    """
    result = apply_fade(
        input_path=input_path,
        output_path=output_path,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
    )
    return to_dict(result)


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
    """Apply audio effects to a file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_fx' suffix in the same directory.
        lowpass_hz: Low-pass filter cutoff (removes high frequencies above this).
            Example: 3000 for "telephone" effect, 8000 for "muffled" sound.
        highpass_hz: High-pass filter cutoff (removes low frequencies below this).
            Example: 300 to remove rumble, 80 for subtle bass cut.
        bass_gain_db: Bass boost/cut in dB. Positive = more bass, negative = less.
            Example: +6 for bass boost, -3 for subtle reduction.
        treble_gain_db: Treble boost/cut in dB. Positive = brighter, negative = darker.
            Example: +3 for clarity, -6 for warm sound.
        speed: Playback speed multiplier (0.5 = half speed, 2.0 = double).
            Note: This also affects pitch.
        reverb: Add reverb effect (room-like ambience).
        echo_delay_ms: Echo delay in milliseconds (None = no echo).
            Example: 200 for subtle echo, 500 for dramatic effect.
        echo_decay: Echo decay factor 0-1 (how quickly echo fades).

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Processed file path
        - duration_ms: Duration of output
        - effects_applied: List of effects that were applied

    Example:
        # Create a "phone call" effect
        result = apply_audio_effects("voice.wav", lowpass_hz=3000, highpass_hz=300)

        # Add dramatic reverb
        result = apply_audio_effects("voice.wav", reverb=True)

        # Boost bass and add echo
        result = apply_audio_effects("voice.wav", bass_gain_db=6, echo_delay_ms=200)

        # Speed up playback
        result = apply_audio_effects("audio.wav", speed=1.5)
    """
    result = apply_effects(
        input_path=input_path,
        output_path=output_path,
        lowpass_hz=lowpass_hz,
        highpass_hz=highpass_hz,
        bass_gain_db=bass_gain_db,
        treble_gain_db=treble_gain_db,
        speed=speed,
        reverb=reverb,
        echo_delay_ms=echo_delay_ms,
        echo_decay=echo_decay,
    )
    return to_dict(result)


@mcp.tool()
def overlay_audio_track(
    base_path: str,
    overlay_path: str,
    output_path: str,
    position_ms: float = 0,
    overlay_volume: float = 1.0,
) -> dict:
    """Overlay one audio track on top of another at a specific position.

    Useful for adding sound effects, background music, or ambience at specific times.

    Args:
        base_path: Path to the base audio file.
        overlay_path: Path to the audio file to overlay.
        output_path: Path for the output file.
        position_ms: Position in base audio where overlay starts (in milliseconds).
            Default: 0 (overlay starts at beginning).
        overlay_volume: Volume multiplier for the overlay (1.0 = original).
            Use lower values (0.2-0.5) for background music under narration.

    Returns:
        Dict with:
        - base_path: Base audio path
        - overlay_path: Overlay audio path
        - output_path: Output file path
        - duration_ms: Duration of output
        - overlay_position_ms: Where overlay was placed
        - overlay_volume: Volume used for overlay

    Example:
        # Add a sound effect at 5 seconds
        result = overlay_audio_track(
            "narration.wav", "explosion.wav", "scene.wav",
            position_ms=5000
        )

        # Add background music at lower volume
        result = overlay_audio_track(
            "narration.wav", "music.wav", "scene.wav",
            overlay_volume=0.2
        )

        # Add ambience starting 2 seconds in
        result = overlay_audio_track(
            "dialog.wav", "rain.wav", "scene.wav",
            position_ms=2000, overlay_volume=0.3
        )
    """
    result = overlay_audio(
        base_path=base_path,
        overlay_path=overlay_path,
        output_path=output_path,
        position_ms=position_ms,
        overlay_volume=overlay_volume,
    )
    return to_dict(result)


# ============================================================================
# Voice Modulation Tools
# ============================================================================


@mcp.tool()
def shift_audio_pitch(
    input_path: str,
    output_path: Optional[str] = None,
    semitones: float = 0,
) -> dict:
    """Shift the pitch of audio without changing its speed.

    Uses high-quality pitch shifting to change pitch while preserving duration.
    This is different from speed changes which affect both pitch and tempo.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_pitched' suffix in the same directory.
        semitones: Pitch shift in semitones.
            Positive values = higher pitch, negative = lower pitch.
            12 semitones = 1 octave up, -12 = 1 octave down.
            Typical range: -12 to +12.

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Pitched file path
        - duration_ms: Duration (unchanged from original)
        - semitones: Pitch shift applied
        - sample_rate: Output sample rate

    Example:
        # Raise pitch by a major third (4 semitones)
        result = shift_audio_pitch("voice.wav", semitones=4)

        # Lower pitch by a perfect fourth (5 semitones)
        result = shift_audio_pitch("voice.wav", semitones=-5)

        # Raise by one octave
        result = shift_audio_pitch("voice.wav", semitones=12)

    Note:
        Requires librosa. Install with: pip install librosa
        Or use the analysis extra: pip install talky-talky[analysis]
    """
    result = shift_pitch(
        input_path=input_path,
        output_path=output_path,
        semitones=semitones,
    )
    return to_dict(result)


@mcp.tool()
def stretch_audio_time(
    input_path: str,
    output_path: Optional[str] = None,
    rate: float = 1.0,
) -> dict:
    """Stretch or compress the duration of audio without changing pitch.

    Uses phase vocoder for high-quality time stretching that preserves pitch.
    This is different from speed changes which affect both pitch and tempo.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_stretched' suffix in the same directory.
        rate: Time stretch factor.
            > 1.0 = faster playback (shorter duration)
            < 1.0 = slower playback (longer duration)
            0.5 = half speed (double the duration)
            2.0 = double speed (half the duration)
            Typical range: 0.5 to 2.0.

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Stretched file path
        - original_duration_ms: Original duration
        - new_duration_ms: New duration after stretching
        - rate: Stretch rate applied
        - sample_rate: Output sample rate

    Example:
        # Slow down to 75% speed (longer duration, same pitch)
        result = stretch_audio_time("voice.wav", rate=0.75)

        # Speed up to 125% (shorter duration, same pitch)
        result = stretch_audio_time("voice.wav", rate=1.25)

        # Double the speed
        result = stretch_audio_time("voice.wav", rate=2.0)

    Note:
        Requires librosa. Install with: pip install librosa
        Or use the analysis extra: pip install talky-talky[analysis]
    """
    result = stretch_time(
        input_path=input_path,
        output_path=output_path,
        rate=rate,
    )
    return to_dict(result)


@mcp.tool()
def apply_voice_effect_preset(
    input_path: str,
    output_path: Optional[str] = None,
    effect: str = "robot",
    intensity: float = 0.5,
) -> dict:
    """Apply a voice effect preset to transform audio.

    Provides easy-to-use voice transformation presets for creative effects.
    Preserves the original sample rate of the input file.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with the effect name as suffix (e.g., '_robot').
        effect: Voice effect preset to apply. Options:
            - "robot": Robotic/synthetic voice
            - "chorus": Choir/ensemble effect (multiple voices)
            - "vibrato": Pitch wobble/tremolo
            - "flanger": Sweeping phaser effect
            - "telephone": Lo-fi telephone quality
            - "megaphone": PA/bullhorn sound (good for PA/intercom at 0.4-0.5)
            - "deep": Deeper voice with bass boost
            - "chipmunk": Higher pitched, cartoonish
            - "whisper": Soft whisper effect
            - "cave": Cavernous echo (use 0.1-0.15 for subtle room ambience)
        intensity: Effect strength from 0.0 to 1.0. Default: 0.5.
            Higher values = more pronounced effect.

            Recommended intensities by effect:
            - megaphone: 0.4-0.5 for PA/announcement systems
            - cave: 0.1-0.15 for subtle room ambience, higher causes extreme echo
            - telephone: 0.5-0.7 for realistic phone call
            - chorus: 0.3-0.5 for ensemble effect without muddiness

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Processed file path
        - duration_ms: Duration of output
        - effect: Effect preset applied
        - intensity: Intensity level used

    Example:
        # Apply robot voice
        result = apply_voice_effect_preset("voice.wav", effect="robot")

        # PA/intercom announcement
        result = apply_voice_effect_preset("voice.wav", effect="megaphone", intensity=0.4)

        # Subtle room ambience
        result = apply_voice_effect_preset("voice.wav", effect="cave", intensity=0.15)

        # Make voice deeper
        result = apply_voice_effect_preset("voice.wav", effect="deep")
    """
    result = apply_voice_effect(
        input_path=input_path,
        output_path=output_path,
        effect=effect,
        intensity=intensity,
    )
    return to_dict(result)


@mcp.tool()
def list_voice_effects() -> dict:
    """List all available voice effect presets.

    Returns a dictionary of available voice effects and their descriptions.
    Use these effect names with apply_voice_effect_preset().

    Returns:
        Dict with:
        - effects: Dict mapping effect name to description
        - count: Number of available effects
    """
    return {
        "effects": VOICE_EFFECTS,
        "count": len(VOICE_EFFECTS),
    }


@mcp.tool()
def shift_voice_formant(
    input_path: str,
    output_path: Optional[str] = None,
    shift_ratio: float = 1.0,
) -> dict:
    """Shift the formants of a voice to change its character.

    Formants are resonant frequencies that give voices their characteristic
    quality. Shifting formants can make a voice sound more masculine or
    feminine without changing the pitch.

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_formant' suffix in the same directory.
        shift_ratio: Formant shift ratio.
            < 1.0 = more masculine (deeper resonance)
              - 0.8 = noticeably more masculine
              - 0.9 = slightly more masculine
            > 1.0 = more feminine (higher resonance)
              - 1.1 = slightly more feminine
              - 1.2 = noticeably more feminine
            1.0 = no change
            Typical range: 0.7 to 1.4.

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Processed file path
        - duration_ms: Duration of output
        - shift_ratio: Ratio applied
        - sample_rate: Output sample rate

    Example:
        # Make a male voice sound more feminine
        result = shift_voice_formant("male_voice.wav", shift_ratio=1.2)

        # Make a female voice sound more masculine
        result = shift_voice_formant("female_voice.wav", shift_ratio=0.85)

        # Subtle feminization
        result = shift_voice_formant("voice.wav", shift_ratio=1.1)

    Note:
        For best quality, install pyworld: pip install pyworld
        Falls back to librosa-based approximation if pyworld is not available.
    """
    result = shift_formant(
        input_path=input_path,
        output_path=output_path,
        shift_ratio=shift_ratio,
    )
    return to_dict(result)


# ============================================================================
# Autotune Tools
# ============================================================================


@mcp.tool()
def autotune_vocals(
    input_path: str,
    output_path: Optional[str] = None,
    key: str = "C",
    scale: str = "major",
    correction_strength: float = 1.0,
    speed: float = 1.0,
) -> dict:
    """Apply autotune (pitch correction) to vocals.

    Uses pyworld WORLD vocoder for high-quality pitch correction.
    Detects pitch frame-by-frame and corrects to the nearest note
    in the specified key and scale.

    Args:
        input_path: Path to the input audio file (vocals/singing).
        output_path: Optional output path. If not provided, creates a file
            with '_autotuned' suffix in the same directory.
        key: Musical key/root note. Options:
            C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B
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
        speed: How quickly to snap to the correct pitch (0.01-1.0).
            1.0 = instant snap (robotic effect, default)
            0.5 = medium speed (more natural)
            0.1 = slow glide (very natural, subtle)

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Autotuned file path
        - duration_ms: Duration of output
        - key: Key used
        - scale: Scale used
        - correction_strength: Strength applied
        - speed: Speed applied
        - frames_corrected: Number of frames that were pitch-corrected
        - total_voiced_frames: Total voiced frames in audio
        - average_correction_cents: Average pitch correction in cents
        - max_correction_cents: Maximum pitch correction in cents

    Example:
        # Classic "T-Pain" hard autotune in A minor
        result = autotune_vocals("vocals.wav", key="A", scale="minor")

        # Subtle pitch correction in C major
        result = autotune_vocals("vocals.wav", key="C", scale="major",
                                 correction_strength=0.5)

        # Natural-sounding correction with slow glide
        result = autotune_vocals("vocals.wav", key="G", scale="major",
                                 correction_strength=0.8, speed=0.3)

        # Blues scale autotune
        result = autotune_vocals("vocals.wav", key="E", scale="blues")

    Note:
        Requires pyworld: pip install pyworld
        For best results, use clean vocal recordings without background music.
    """
    result = autotune_audio(
        input_path=input_path,
        output_path=output_path,
        key=key,
        scale=scale,
        correction_strength=correction_strength,
        speed=speed,
    )
    return to_dict(result)


@mcp.tool()
def detect_vocal_pitch(
    input_path: str,
    method: str = "harvest",
    frame_period_ms: float = 5.0,
) -> dict:
    """Detect pitch (fundamental frequency) in an audio file.

    Analyzes vocals to extract pitch information frame-by-frame.
    Useful for understanding the pitch content before autotuning
    or for analyzing singing performance.

    Args:
        input_path: Path to the audio file to analyze.
        method: Pitch detection algorithm.
            "harvest" - High quality, slower (default, recommended)
            "dio" - Faster, slightly less accurate
        frame_period_ms: Analysis frame period in milliseconds (default 5.0).
            Lower values = more detailed analysis but larger output.

    Returns:
        Dict with:
        - input_path: Analyzed file path
        - duration_ms: Audio duration
        - sample_rate: Audio sample rate
        - frame_count: Number of analysis frames
        - frame_period_ms: Frame period used
        - voiced_frames: Frames with detected pitch
        - unvoiced_frames: Frames without pitch (silence/noise)
        - pitch_range_hz: Tuple of (min, max) detected frequencies
        - average_pitch_hz: Mean pitch frequency
        - median_pitch_hz: Median pitch frequency
        - detected_notes: List of (note_name, frequency, count) for top notes

    Example:
        # Analyze a vocal recording
        result = detect_vocal_pitch("singing.wav")
        print(f"Pitch range: {result['pitch_range_hz']}")
        print(f"Most common notes: {result['detected_notes']}")

        # Faster analysis with dio
        result = detect_vocal_pitch("vocals.wav", method="dio")

    Note:
        Requires pyworld: pip install pyworld
    """
    result = detect_audio_pitch(
        input_path=input_path,
        method=method,
        frame_period_ms=frame_period_ms,
    )
    return to_dict(result)


@mcp.tool()
def list_autotune_scales() -> dict:
    """List all available musical scales for autotune.

    Returns all supported scales with their semitone intervals from the root.
    Use these scale names with the autotune_vocals tool.

    Returns:
        Dict with:
        - scales: Dict mapping scale names to semitone intervals
          Example: {"major": [0, 2, 4, 5, 7, 9, 11], ...}

    Scale categories:
    - Major modes: major/ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian
    - Minor scales: minor, harmonic_minor, melodic_minor
    - Pentatonic: major_pentatonic, minor_pentatonic
    - Blues: blues
    - Chromatic: chromatic (all 12 notes)
    """
    return {"scales": get_autotune_scales()}


@mcp.tool()
def list_autotune_keys() -> dict:
    """List all available musical keys for autotune.

    Returns all supported key/root note names.
    Use these key names with the autotune_vocals tool.

    Returns:
        Dict with:
        - keys: List of key names (C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B)

    Note:
        Enharmonic equivalents are supported (C# = Db, D# = Eb, etc.)
    """
    return {"keys": get_autotune_keys()}


# ============================================================================
# Transcription Tools
# ============================================================================


@mcp.tool()
def check_transcription_availability() -> dict:
    """Check if transcription engines are available and properly configured.

    Returns detailed status including:
    - Available engines
    - Device info (CUDA/MPS/CPU)
    - Setup instructions for unavailable engines
    """
    status = check_transcription()
    return to_dict(status)


@mcp.tool()
def get_transcription_engines_info() -> dict:
    """Get detailed information about all transcription engines.

    Returns info for each engine including:
    - Name and description
    - Requirements and availability
    - Supported languages and model sizes
    - Engine-specific parameters
    """
    return get_transcription_info()


@mcp.tool()
def list_available_transcription_engines() -> dict:
    """List transcription engines that are currently available (installed).

    Returns:
        Dict with list of available engine IDs and their basic info.
    """
    available = get_available_transcription_engines()
    engines = list_transcription_engines()

    return {
        "available_engines": available,
        "engines": {
            engine_id: {
                "name": info.name,
                "description": info.description,
                "supports_word_timestamps": info.supports_word_timestamps,
            }
            for engine_id, info in engines.items()
            if engine_id in available
        },
    }


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    engine: str = "faster_whisper",
    language: Optional[str] = None,
    model_size: str = "large-v3",
) -> dict:
    """Transcribe audio to text using speech recognition.

    Use this to verify TTS output or transcribe any audio file.
    Supports 99+ languages with automatic language detection.

    Args:
        audio_path: Path to the audio file to transcribe.
        engine: Transcription engine to use (default: "faster_whisper").
            Options: "whisper", "faster_whisper"
            - whisper: Best accuracy via transformers
            - faster_whisper: 4x faster with same accuracy (recommended)
        language: Language code (e.g., "en", "es", "ja"). Auto-detects if not specified.
        model_size: Model size (default: "large-v3" for best accuracy).
            Options: tiny, base, small, medium, large-v3, large-v3-turbo
            Larger models = more accurate but slower.

    Returns:
        Dict with:
        - status: "success" or "error"
        - text: Full transcribed text
        - segments: List of segments with timing info
        - language: Detected/used language
        - duration_seconds: Audio duration
        - processing_time_ms: How long transcription took

    Example:
        # Verify TTS output contains expected text
        result = transcribe_audio("generated_speech.wav")
        if "hello world" in result["text"].lower():
            print("TTS verification passed!")
    """
    result = transcribe(
        audio_path=audio_path,
        engine=engine,
        language=language,
        model_size=model_size,
    )
    return to_dict(result)


@mcp.tool()
def transcribe_with_timestamps(
    audio_path: str,
    engine: str = "faster_whisper",
    language: Optional[str] = None,
    model_size: str = "large-v3",
    word_level: bool = False,
) -> dict:
    """Transcribe audio with detailed timing information.

    Returns segment-level or word-level timestamps for precise alignment.
    Useful for subtitles, karaoke, or audio analysis.

    Args:
        audio_path: Path to the audio file to transcribe.
        engine: Transcription engine ("whisper" or "faster_whisper").
        language: Language code (auto-detected if not specified).
        model_size: Model size (default: "large-v3"). Options: tiny, base, small, medium, large-v3, etc.
        word_level: If True, include word-level timestamps (slower but more precise).

    Returns:
        Dict with:
        - status: "success" or "error"
        - text: Full transcribed text
        - segments: List of segments, each containing:
            - text: Segment text
            - start: Start time in seconds
            - end: End time in seconds
            - words: (if word_level=True) List of words with timing
        - language: Detected language
        - duration_seconds: Audio duration

    Note: Word-level timestamps are more accurate with faster_whisper engine.
    """
    # Set appropriate parameters for each engine
    if engine == "faster_whisper":
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
            word_timestamps=word_level,
            vad_filter=True,
        )
    else:
        # Whisper via transformers
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
            return_timestamps="word" if word_level else True,
        )
    return to_dict(result)


@mcp.tool()
def verify_tts_output(
    audio_path: str,
    expected_text: str,
    engine: str = "faster_whisper",
    model_size: str = "large-v3",
    similarity_threshold: float = 0.8,
) -> dict:
    """Verify that TTS-generated audio contains the expected text.

    Transcribes the audio and compares it to the expected text.
    Useful for automated testing of TTS output quality.

    Args:
        audio_path: Path to the TTS-generated audio file.
        expected_text: The text that should be in the audio.
        engine: Transcription engine to use (default: "faster_whisper").
        model_size: Model size (default: "large-v3" for best accuracy).
        similarity_threshold: Minimum similarity ratio to consider a match (0.0-1.0).
            Default 0.8 (80% similar). Lower for lenient matching.

    Returns:
        Dict with:
        - status: "success" or "error"
        - verified: True if transcription matches expected text
        - similarity: Similarity ratio (0.0 to 1.0)
        - expected_text: The expected text (normalized)
        - transcribed_text: What was actually transcribed (normalized)
        - match_details: Additional matching information

    Example:
        # After generating TTS audio
        result = verify_tts_output(
            audio_path="greeting.wav",
            expected_text="Hello, how are you today?",
        )
        if result["verified"]:
            print("TTS output verified successfully!")
    """
    from difflib import SequenceMatcher

    # Transcribe the audio
    transcription_result = transcribe(
        audio_path=audio_path,
        engine=engine,
        model_size=model_size,
    )

    if transcription_result.status == "error":
        return {
            "status": "error",
            "verified": False,
            "error": transcription_result.error,
        }

    # Normalize texts for comparison
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        import re

        # Convert to lowercase
        text = text.lower()
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    expected_normalized = normalize_text(expected_text)
    transcribed_normalized = normalize_text(transcription_result.text)

    # Calculate similarity
    matcher = SequenceMatcher(None, expected_normalized, transcribed_normalized)
    similarity = matcher.ratio()

    # Check for exact word match (more lenient than character-level)
    expected_words = set(expected_normalized.split())
    transcribed_words = set(transcribed_normalized.split())
    word_overlap = len(expected_words & transcribed_words) / max(len(expected_words), 1)

    # Use the higher of the two similarity measures
    effective_similarity = max(similarity, word_overlap)

    verified = effective_similarity >= similarity_threshold

    return {
        "status": "success",
        "verified": verified,
        "similarity": round(effective_similarity, 3),
        "character_similarity": round(similarity, 3),
        "word_overlap": round(word_overlap, 3),
        "expected_text": expected_normalized,
        "transcribed_text": transcribed_normalized,
        "threshold": similarity_threshold,
        "match_details": {
            "expected_word_count": len(expected_words),
            "transcribed_word_count": len(transcribed_words),
            "matching_words": len(expected_words & transcribed_words),
        },
        "transcription_metadata": {
            "engine": engine,
            "model_size": model_size,
            "language": transcription_result.language,
            "processing_time_ms": transcription_result.processing_time_ms,
        },
    }


# ============================================================================
# Audio Analysis Tools
# ============================================================================


@mcp.tool()
def check_analysis_availability() -> dict:
    """Check if audio analysis engines are available and properly configured.

    Returns status of three analysis capabilities:
    - Emotion detection (emotion2vec)
    - Voice similarity (Resemblyzer)
    - Speech quality assessment (NISQA)

    Returns:
        Dict with available engines and their status.
    """
    emotion_engines = list_emotion_engines()
    similarity_engines = list_similarity_engines()
    quality_engines = list_quality_engines()

    return {
        "emotion_detection": {
            "available": len(emotion_engines) > 0,
            "engines": list(emotion_engines.keys()),
        },
        "voice_similarity": {
            "available": len(similarity_engines) > 0,
            "engines": list(similarity_engines.keys()),
        },
        "speech_quality": {
            "available": len(quality_engines) > 0,
            "engines": list(quality_engines.keys()),
        },
    }


@mcp.tool()
def get_analysis_engines_info() -> dict:
    """Get detailed information about all audio analysis engines.

    Returns info for each category:
    - Emotion detection engines
    - Voice similarity engines
    - Speech quality engines

    Each engine includes description, requirements, and capabilities.
    """
    return {
        "emotion_detection": {
            engine_id: {
                "name": info.name,
                "description": info.description,
                "requirements": info.requirements,
                "supported_emotions": info.supported_emotions,
            }
            for engine_id, info in list_emotion_engines().items()
        },
        "voice_similarity": {
            engine_id: {
                "name": info.name,
                "description": info.description,
                "requirements": info.requirements,
                "embedding_dim": info.embedding_dim,
                "default_threshold": info.default_threshold,
            }
            for engine_id, info in list_similarity_engines().items()
        },
        "speech_quality": {
            engine_id: {
                "name": info.name,
                "description": info.description,
                "requirements": info.requirements,
                "quality_dimensions": info.quality_dimensions,
                "score_range": info.score_range,
            }
            for engine_id, info in list_quality_engines().items()
        },
    }


@mcp.tool()
def analyze_emotion(
    audio_path: str,
    engine: str = "emotion2vec",
) -> dict:
    """Detect emotion in audio using speech emotion recognition.

    Use this to verify the emotional tone of TTS output matches intent.
    Supports 9 emotions: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown.

    Args:
        audio_path: Path to the audio file to analyze.
        engine: Emotion detection engine (default: "emotion2vec").

    Returns:
        Dict with:
        - status: "success" or "error"
        - primary_emotion: Highest scoring emotion
        - primary_score: Confidence score (0-1)
        - all_emotions: List of all emotions with scores
        - processing_time_ms: Analysis time

    Example:
        # Verify TTS with angry emotion tag sounds angry
        result = analyze_emotion("angry_speech.wav")
        if result["primary_emotion"] == "angry":
            print("Emotion verified!")
    """
    result = detect_emotion(audio_path, engine=engine)
    return analysis_to_dict(result)


@mcp.tool()
def analyze_voice_similarity(
    audio_path_1: str,
    audio_path_2: str,
    threshold: float = 0.75,
    engine: str = "resemblyzer",
) -> dict:
    """Compare two audio files for voice similarity.

    Use this to verify voice cloning quality by comparing generated audio
    to the reference voice sample.

    Args:
        audio_path_1: Path to first audio file (e.g., reference voice).
        audio_path_2: Path to second audio file (e.g., generated TTS).
        threshold: Similarity threshold for same-speaker determination (default: 0.75).
            - 0.75+: Likely same speaker
            - 0.60-0.75: Possibly same speaker
            - <0.60: Likely different speakers
        engine: Voice similarity engine (default: "resemblyzer").

    Returns:
        Dict with:
        - status: "success" or "error"
        - similarity_score: Cosine similarity (0-1)
        - is_same_speaker: True if above threshold
        - threshold_used: The threshold applied
        - processing_time_ms: Analysis time

    Example:
        # Verify voice cloning quality
        result = analyze_voice_similarity("reference.wav", "generated.wav")
        if result["is_same_speaker"]:
            print(f"Voice match! Similarity: {result['similarity_score']:.1%}")
    """
    result = compare_voices(audio_path_1, audio_path_2, threshold=threshold, engine=engine)
    return analysis_to_dict(result)


@mcp.tool()
def extract_voice_embedding(
    audio_path: str,
    engine: str = "resemblyzer",
) -> dict:
    """Extract a voice embedding vector from audio.

    Voice embeddings are 256-dimensional vectors that capture speaker characteristics.
    Store embeddings to efficiently compare against multiple audio files later.

    Args:
        audio_path: Path to audio file (5-30 seconds of speech recommended).
        engine: Voice similarity engine (default: "resemblyzer").

    Returns:
        Dict with:
        - status: "success" or "error"
        - embedding: List of 256 float values
        - embedding_dim: Dimension (256)
        - processing_time_ms: Extraction time

    Example:
        # Extract reference embedding once, compare many times
        ref_result = extract_voice_embedding("reference.wav")
        ref_embedding = ref_result["embedding"]
        # Store ref_embedding for later comparisons
    """
    result = get_voice_embedding(audio_path, engine=engine)
    return analysis_to_dict(result)


@mcp.tool()
def analyze_speech_quality(
    audio_path: str,
    engine: str = "nisqa",
) -> dict:
    """Assess speech quality and naturalness of audio.

    Predicts Mean Opinion Score (MOS) and quality dimensions without
    needing reference audio. Use this to verify TTS output quality.

    Args:
        audio_path: Path to audio file to assess.
        engine: Speech quality engine (default: "nisqa").

    Returns:
        Dict with:
        - status: "success" or "error"
        - overall_quality: MOS score (1-5 scale)
        - dimensions: List of quality dimensions:
            - overall: Overall MOS prediction
            - noisiness: Background noise level
            - discontinuity: Audio dropouts/glitches
            - coloration: Spectral distortion
            - loudness: Volume appropriateness
        - processing_time_ms: Assessment time

    Score interpretation (MOS scale):
    - 5.0: Excellent quality
    - 4.0: Good quality
    - 3.0: Fair quality
    - 2.0: Poor quality
    - 1.0: Bad quality

    Example:
        result = analyze_speech_quality("generated.wav")
        if result["overall_quality"] >= 3.5:
            print("Quality acceptable!")
    """
    result = assess_quality(audio_path, engine=engine)
    return analysis_to_dict(result)


@mcp.tool()
def verify_tts_comprehensive(
    generated_audio_path: str,
    expected_text: str,
    reference_audio_path: Optional[str] = None,
    expected_emotion: Optional[str] = None,
    min_quality_score: float = 3.0,
    min_similarity_score: float = 0.70,
    min_text_similarity: float = 0.8,
) -> dict:
    """Comprehensive TTS output verification combining all analysis tools.

    Performs multiple checks on generated TTS audio:
    1. Text accuracy: Transcribes and compares to expected text
    2. Voice similarity: Compares to reference audio (if provided)
    3. Emotion match: Verifies emotional tone (if expected_emotion specified)
    4. Speech quality: Checks naturalness and technical quality

    Args:
        generated_audio_path: Path to the generated TTS audio.
        expected_text: The text that should be in the audio.
        reference_audio_path: Optional reference audio for voice comparison.
        expected_emotion: Optional expected emotion (e.g., "happy", "angry").
        min_quality_score: Minimum acceptable MOS score (default: 3.0).
        min_similarity_score: Minimum voice similarity (default: 0.70).
        min_text_similarity: Minimum text match ratio (default: 0.8).

    Returns:
        Dict with:
        - status: "success" or "error"
        - overall_pass: True if all enabled checks pass
        - checks: Dict with results of each check:
            - text_accuracy: Text transcription comparison
            - voice_similarity: Voice match (if reference provided)
            - emotion_match: Emotion detection (if expected_emotion provided)
            - speech_quality: MOS and quality dimensions
        - recommendations: List of suggestions for improvements

    Example:
        result = verify_tts_comprehensive(
            generated_audio_path="output.wav",
            expected_text="Hello, how are you today?",
            reference_audio_path="reference_voice.wav",
            expected_emotion="happy",
        )
        if result["overall_pass"]:
            print("TTS output verified!")
        else:
            for rec in result["recommendations"]:
                print(f"- {rec}")
    """
    results = {
        "status": "success",
        "overall_pass": True,
        "checks": {},
        "recommendations": [],
    }

    # 1. Text accuracy check
    from difflib import SequenceMatcher

    transcription_result = transcribe(
        audio_path=generated_audio_path,
        engine="faster_whisper",
        model_size="large-v3",
    )

    if transcription_result.status == "success":
        # Normalize texts
        import re

        def normalize(text):
            text = text.lower()
            text = re.sub(r"[^\w\s']", "", text)
            return re.sub(r"\s+", " ", text).strip()

        expected_norm = normalize(expected_text)
        transcribed_norm = normalize(transcription_result.text)
        similarity = SequenceMatcher(None, expected_norm, transcribed_norm).ratio()

        text_pass = similarity >= min_text_similarity
        results["checks"]["text_accuracy"] = {
            "pass": text_pass,
            "similarity": round(similarity, 3),
            "threshold": min_text_similarity,
            "expected": expected_norm,
            "transcribed": transcribed_norm,
        }
        if not text_pass:
            results["overall_pass"] = False
            results["recommendations"].append(
                f"Text accuracy ({similarity:.1%}) below threshold ({min_text_similarity:.1%}). "
                "Check TTS pronunciation or try a different voice."
            )
    else:
        results["checks"]["text_accuracy"] = {
            "pass": False,
            "error": transcription_result.error,
        }
        results["overall_pass"] = False

    # 2. Voice similarity check (if reference provided)
    if reference_audio_path:
        similarity_result = compare_voices(
            reference_audio_path,
            generated_audio_path,
            threshold=min_similarity_score,
        )
        if similarity_result.status == "success":
            voice_pass = similarity_result.similarity_score >= min_similarity_score
            results["checks"]["voice_similarity"] = {
                "pass": voice_pass,
                "similarity": round(similarity_result.similarity_score, 3),
                "threshold": min_similarity_score,
            }
            if not voice_pass:
                results["overall_pass"] = False
                results["recommendations"].append(
                    f"Voice similarity ({similarity_result.similarity_score:.1%}) below threshold. "
                    "Try longer reference audio or different TTS settings."
                )
        else:
            results["checks"]["voice_similarity"] = {
                "pass": False,
                "error": similarity_result.error,
            }

    # 3. Emotion check (if expected emotion provided)
    if expected_emotion:
        emotion_result = detect_emotion(generated_audio_path)
        if emotion_result.status == "success":
            emotion_match = emotion_result.primary_emotion.lower() == expected_emotion.lower()
            results["checks"]["emotion_match"] = {
                "pass": emotion_match,
                "expected": expected_emotion.lower(),
                "detected": emotion_result.primary_emotion,
                "confidence": round(emotion_result.primary_score, 3)
                if emotion_result.primary_score
                else None,
            }
            if not emotion_match:
                results["overall_pass"] = False
                results["recommendations"].append(
                    f"Expected '{expected_emotion}' but detected '{emotion_result.primary_emotion}'. "
                    "Try adjusting emotion tags or voice parameters."
                )
        else:
            results["checks"]["emotion_match"] = {
                "pass": False,
                "error": emotion_result.error,
            }

    # 4. Speech quality check
    quality_result = assess_quality(generated_audio_path)
    if quality_result.status == "success":
        quality_pass = quality_result.overall_quality >= min_quality_score
        results["checks"]["speech_quality"] = {
            "pass": quality_pass,
            "overall_mos": round(quality_result.overall_quality, 2),
            "threshold": min_quality_score,
            "dimensions": {dim.name: round(dim.score, 2) for dim in quality_result.dimensions},
        }
        if not quality_pass:
            results["overall_pass"] = False
            # Find lowest dimension for specific feedback
            lowest = min(quality_result.dimensions, key=lambda d: d.score)
            results["recommendations"].append(
                f"Speech quality ({quality_result.overall_quality:.2f}) below threshold ({min_quality_score}). "
                f"Lowest dimension: {lowest.name} ({lowest.score:.2f})."
            )
    else:
        results["checks"]["speech_quality"] = {
            "pass": False,
            "error": quality_result.error,
        }

    return results


@mcp.tool()
def detect_spoken_tts_tags(
    audio_path: str,
    tags: Optional[list[str]] = None,
    tts_engine: Optional[str] = None,
    engine: str = "faster_whisper",
    model_size: str = "base",
) -> dict:
    """Detect if TTS spoke tags as words instead of performing them.

    Checks if TTS engines incorrectly spoke tags like "[chuckle]" or "<laugh>"
    as literal words instead of performing the intended action. This is a
    common issue with some TTS engines.

    Args:
        audio_path: Path to the audio file to analyze.
        tags: List of tags to check for. If None, uses common defaults.
        tts_engine: Optional TTS engine used to generate the audio. If provided,
            returns suggestions for correctly formatted tags for that engine.
        engine: Transcription engine to use (default: "faster_whisper").
        model_size: Whisper model size (default: "base" for speed).

    Returns:
        Dict with:
        - status: "success" or "error"
        - has_spoken_tags: True if tags were spoken as words
        - spoken_tags: List of tags that were spoken as words
        - transcription: Full transcribed text
        - confidence: Overall detection confidence (0-1)
        - suggested_fixes: Dict mapping spoken tags to correct format (if tts_engine provided)

    Example:
        result = detect_spoken_tts_tags("tts_output.wav", tts_engine="maya1")
        if result["has_spoken_tags"]:
            print(f"TTS spoke these tags: {result['spoken_tags']}")
            if result.get("suggested_fixes"):
                print(f"Use these instead: {result['suggested_fixes']}")
    """
    return detect_spoken_tags(
        audio_path=audio_path,
        tags=tags,
        tts_engine=tts_engine,
        engine=engine,
        model_size=model_size,
    )


@mcp.tool()
def quick_compare_audio_to_text(
    audio_path: str,
    expected_text: str,
    engine: str = "faster_whisper",
    model_size: str = "base",
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
) -> dict:
    """Quick comparison of audio content to expected text.

    A lightweight alternative to verify_tts_output that focuses only on
    text matching without quality/emotion checks. Faster for simple
    "does this audio say what it should?" checks.

    Args:
        audio_path: Path to the audio file to check.
        expected_text: The text that should be in the audio.
        engine: Transcription engine to use (default: "faster_whisper").
        model_size: Whisper model size (default: "base" for speed).
        ignore_case: Ignore case differences (default: True).
        ignore_punctuation: Ignore punctuation differences (default: True).

    Returns:
        Dict with:
        - status: "success" or "error"
        - matches: True if transcription matches expected text
        - similarity: Similarity ratio (0.0 to 1.0)
        - transcribed_text: What was actually transcribed
        - expected_text: The expected text
        - differences: List of word-level differences (if any)

    Example:
        result = quick_compare_audio_to_text(
            "greeting.wav",
            "Hello, how are you today?"
        )
        if result["matches"]:
            print("Audio matches expected text!")
        else:
            print(f"Similarity: {result['similarity']:.1%}")
    """
    return compare_audio_to_text(
        audio_path=audio_path,
        expected_text=expected_text,
        engine=engine,
        model_size=model_size,
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
    )


@mcp.tool()
def convert_tts_tags(
    text: str,
    target_engine: str,
) -> dict:
    """Convert paralinguistic tags in text to the correct format for a TTS engine.

    Different TTS engines use different tag formats:
    - Maya1: <tag> (angle brackets)
    - Chatterbox/Chatterbox Turbo: [tag] (square brackets)
    - CosyVoice: [breath] (square brackets)

    This function converts tags between formats automatically.

    Args:
        text: Text containing paralinguistic tags.
        target_engine: Target TTS engine ID ("maya1", "chatterbox", "chatterbox_turbo", "cosyvoice").

    Returns:
        Dict with:
        - converted_text: Text with tags in correct format for the engine
        - changes_made: List of tag conversions performed
        - unsupported_tags: List of tags not supported by the target engine

    Example:
        # Convert Chatterbox-style [laugh] to Maya1-style <laugh>
        result = convert_tts_tags("Hello [laugh] world", "maya1")
        # result["converted_text"] = "Hello <laugh> world"

        # Convert Maya1-style <sigh> to Chatterbox-style [sigh]
        result = convert_tts_tags("Oh <sigh> fine", "chatterbox")
        # result["converted_text"] = "Oh [sigh] fine"
    """
    return convert_tags_for_engine(text=text, target_engine=target_engine)


@mcp.tool()
def batch_generate_tts(
    segments: list[dict],
    engine: str = "maya1",
    output_dir: Optional[str] = None,
    continue_on_error: bool = True,
    auto_convert_tags: bool = True,
    voice_description: Optional[str] = None,
    reference_audio_paths: Optional[list[str]] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> dict:
    """Generate audio for multiple text segments at once.

    Processes a manifest of segments and generates audio for each.
    Much more efficient than calling individual speak_* functions as it
    keeps the model loaded between segments.

    Args:
        segments: List of segment dictionaries, each containing:
            - text: The text to synthesize (required)
            - output_path: Output filename or path (required)
            - Any engine-specific overrides (optional)
        engine: Engine ID to use for all segments (default: "maya1")
        output_dir: Directory for output files. If segment output_path is just
            a filename, it will be placed in this directory.
        continue_on_error: If True, continue with remaining segments if one fails.
        auto_convert_tags: If True, automatically convert paralinguistic tags to
            the correct format for the target engine (e.g., [laugh] -> <laugh>).
        voice_description: For Maya1 - voice description for all segments.
        reference_audio_paths: For Chatterbox/XTTS - reference audio paths.
        exaggeration: For Chatterbox - expressiveness (0.0-1.0+).
        cfg_weight: For Chatterbox - pacing control (0.0-1.0).

    Returns:
        Dict with:
        - status: "success", "partial", or "error"
        - total: Total number of segments
        - succeeded: Number of successfully generated
        - failed: Number of failed segments
        - results: List of result dicts for each segment
        - failed_segments: List of segment indices that failed

    Example:
        # Generate a batch of segments for an audiobook
        segments = [
            {"text": "Chapter one. The beginning.", "output_path": "chapter_01_001.wav"},
            {"text": "It was a dark and stormy night.", "output_path": "chapter_01_002.wav"},
            {"text": "[sigh] Here we go again.", "output_path": "chapter_01_003.wav"},
        ]

        result = batch_generate_tts(
            segments=segments,
            engine="chatterbox",
            output_dir="/path/to/output/",
            reference_audio_paths=["/path/to/narrator.wav"],
            exaggeration=0.6,
        )

        print(f"Generated {result['succeeded']}/{result['total']} segments")
    """
    # Resolve output directory
    resolved_output_dir = None
    if output_dir:
        resolved_output_dir = resolve_output_path(output_dir)

    # Build kwargs based on engine type
    kwargs = {}
    if voice_description:
        kwargs["voice_description"] = voice_description
    if reference_audio_paths:
        kwargs["reference_audio_paths"] = reference_audio_paths
    if engine in ("chatterbox", "chatterbox_turbo"):
        kwargs["exaggeration"] = exaggeration
        kwargs["cfg_weight"] = cfg_weight

    return batch_generate(
        segments=segments,
        engine=engine,
        output_dir=resolved_output_dir,
        continue_on_error=continue_on_error,
        auto_convert_tags=auto_convert_tags,
        **kwargs,
    )


# ============================================================================
# Speech Boundary Detection Tools
# ============================================================================


@mcp.tool()
def find_speech_onset(
    audio_path: str,
    approximate_ms: float,
    search_window_ms: float = 150,
    energy_threshold: float = 0.1,
) -> dict:
    """Given a rough timestamp, find the precise speech onset using energy detection.

    Analyzes waveform energy to find where voiced audio actually begins within
    a search window around the approximate timestamp. This is useful for aligning
    audio to precise timestamps.

    Args:
        audio_path: Path to the audio file.
        approximate_ms: Approximate timestamp where speech should start.
        search_window_ms: Search window size in ms (default: 150ms).
            Searches ±search_window_ms around approximate_ms.
        energy_threshold: Energy rise threshold (0-1, default: 0.1).
            Lower values detect quieter onsets.

    Returns:
        Dict with:
        - status: "success" or "error"
        - onset_ms: Precise millisecond where voiced audio begins
        - confidence: Detection confidence (0-1)
        - search_start_ms: Start of search window
        - search_end_ms: End of search window
        - approximate_ms: Original approximate timestamp

    Example:
        # Find precise onset near 1000ms
        result = find_speech_onset("speech.wav", approximate_ms=1000, search_window_ms=100)
        print(f"Actual onset at {result['onset_ms']}ms")
    """
    return detect_speech_onset(
        audio_path=audio_path,
        approximate_ms=approximate_ms,
        search_window_ms=search_window_ms,
        energy_threshold=energy_threshold,
    )


@mcp.tool()
def check_truncated_audio(
    audio_path: str,
    attack_threshold_ms: float = 10,
    decay_threshold_ms: float = 50,
) -> dict:
    """Analyze if audio beginning/end sounds clipped or truncated.

    Detects if audio is missing attack transients at the start or has
    abrupt cutoffs at the end. This can catch issues like "oday" vs "Today"
    where the initial consonant was clipped.

    Args:
        audio_path: Path to the audio file.
        attack_threshold_ms: Time to reach peak energy from start (default: 10ms).
            If peak is reached too quickly, may indicate clipped attack.
        decay_threshold_ms: Minimum decay time at end (default: 50ms).
            If audio ends too abruptly, may indicate clipped ending.

    Returns:
        Dict with:
        - status: "success" or "error"
        - is_truncated: True if either start or end appears truncated
        - start_clipped: True if beginning appears clipped
        - end_clipped: True if ending appears abruptly cut
        - attack_time_ms: Time from start to first energy peak
        - decay_time_ms: Time from last peak to end
        - start_confidence: Confidence in start clipping detection (0-1)
        - end_confidence: Confidence in end clipping detection (0-1)
        - suggestions: List of suggested fixes

    Example:
        result = check_truncated_audio("segment.wav")
        if result["is_truncated"]:
            print("Audio may be clipped!")
            if result["start_clipped"]:
                print("- Beginning may be cut off")
    """
    return detect_truncated_audio(
        audio_path=audio_path,
        attack_threshold_ms=attack_threshold_ms,
        decay_threshold_ms=decay_threshold_ms,
    )


@mcp.tool()
def check_segment_boundaries(
    audio_path: str,
    expected_text: str,
    engine: str = "faster_whisper",
    model_size: str = "base",
) -> dict:
    """Transcribe audio and verify it starts/ends cleanly with expected words.

    Checks if the transcription matches the expected text at the boundaries,
    which can indicate whether the audio was clipped. For example, if expected
    text is "Today is Monday" but transcription is "oday is Monday", the start
    is likely clipped.

    Args:
        audio_path: Path to the audio file.
        expected_text: The text the audio should contain.
        engine: Transcription engine (default: "faster_whisper").
        model_size: Whisper model size (default: "base").

    Returns:
        Dict with:
        - status: "success" or "error"
        - boundaries_clean: True if both start and end match expected
        - start_matches: True if first word matches
        - end_matches: True if last word matches
        - expected_first_word: Expected first word
        - transcribed_first_word: Actual first word
        - expected_last_word: Expected last word
        - transcribed_last_word: Actual last word
        - full_transcription: Full transcribed text
        - suggestions: List of issues found

    Example:
        result = check_segment_boundaries("segment.wav", "Today is a beautiful day")
        if not result["start_matches"]:
            print(f"Expected '{result['expected_first_word']}' but got '{result['transcribed_first_word']}'")
    """
    return verify_segment_boundaries(
        audio_path=audio_path,
        expected_text=expected_text,
        engine=engine,
        model_size=model_size,
    )


@mcp.tool()
def smart_trim_to_speech(
    audio_path: str,
    output_path: Optional[str] = None,
    target_start_ms: Optional[float] = None,
    padding_before_ms: float = 75,
    padding_after_ms: float = 75,
    search_window_ms: float = 100,
) -> dict:
    """Trim audio to speech boundaries with intelligent padding.

    Instead of trusting timestamps exactly, finds the nearest silence→speech
    transition and adds configurable padding. This ensures clean cuts that
    don't clip speech.

    Args:
        audio_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_smart_trimmed' suffix.
        target_start_ms: Optional approximate start time. If provided, searches
            for speech onset near this point. If None, trims from audio start.
        padding_before_ms: Padding to add before detected speech start (default: 75ms).
        padding_after_ms: Padding to add after detected speech end (default: 75ms).
        search_window_ms: Window to search for speech transitions (default: 100ms).

    Returns:
        Dict with:
        - status: "success" or "error"
        - input_path: Original file path
        - output_path: Trimmed file path
        - detected_start_ms: Where speech was detected to start
        - detected_end_ms: Where speech was detected to end
        - actual_start_ms: Trim start point (with padding)
        - actual_end_ms: Trim end point (with padding)
        - original_duration_ms: Original file duration
        - trimmed_duration_ms: New file duration

    Example:
        # Smart trim with 75ms padding
        result = smart_trim_to_speech("speech.wav", padding_before_ms=75)
    """
    # Resolve output path if provided
    resolved_output = resolve_output_path(output_path) if output_path else None

    return trim_to_speech_with_padding(
        audio_path=audio_path,
        output_path=resolved_output,
        target_start_ms=target_start_ms,
        padding_before_ms=padding_before_ms,
        padding_after_ms=padding_after_ms,
        search_window_ms=search_window_ms,
    )


# ============================================================================
# SFX Mixing Tools
# ============================================================================


@mcp.tool()
def normalize_sfx_to_level(
    audio_path: str,
    output_path: Optional[str] = None,
    target_lufs: float = -20.0,
    true_peak: float = -1.5,
) -> dict:
    """Normalize audio to a specific LUFS target for consistent mixing.

    Uses ffmpeg's loudnorm filter for broadcast-standard normalization.
    This is the preferred method for SFX mixing where you need precise
    loudness matching.

    Args:
        audio_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_lufs' suffix.
        target_lufs: Target integrated loudness in LUFS (default: -20).
            Common values:
            - -16 LUFS: Broadcast standard (TV, radio)
            - -14 LUFS: Streaming platforms (Spotify, YouTube)
            - -20 LUFS: Good for SFX mixing headroom
            - -23 LUFS: EBU R128 standard
        true_peak: Maximum true peak in dBTP (default: -1.5).

    Returns:
        Dict with:
        - input_path: Original file path
        - output_path: Normalized file path
        - duration_ms: Duration of output
        - target_lufs: Target LUFS used
        - input_lufs: Original file's LUFS (if measured)

    Example:
        # Normalize SFX to -20 LUFS for mixing
        result = normalize_sfx_to_level("explosion.wav", target_lufs=-20)
    """
    resolved_output = resolve_output_path(output_path) if output_path else None

    result = normalize_to_lufs(
        input_path=audio_path,
        output_path=resolved_output,
        target_lufs=target_lufs,
        true_peak=true_peak,
    )

    return {
        "input_path": result.input_path,
        "output_path": result.output_path,
        "duration_ms": result.duration_ms,
        "target_lufs": result.target_lufs,
        "input_lufs": result.input_lufs,
    }


@mcp.tool()
def get_audio_mean_level(audio_path: str) -> dict:
    """Get quick mean dB level of an audio file.

    A simpler alternative to analyze_audio_loudness that just returns
    the key values you need for mixing decisions.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with:
        - path: Audio file path
        - mean_db: Mean volume in dB
        - peak_db: Peak volume in dB
        - rms_db: RMS level in dB
        - duration_ms: Duration of audio

    Example:
        # Quick level check before mixing
        level = get_audio_mean_level("narration.wav")
        print(f"Mean: {level['mean_db']:.1f} dB, Peak: {level['peak_db']:.1f} dB")
    """
    result = get_mean_level(audio_path)

    return {
        "path": result.path,
        "mean_db": result.mean_db,
        "peak_db": result.peak_db,
        "rms_db": result.rms_db,
        "duration_ms": result.duration_ms,
    }


@mcp.tool()
def compare_audio_levels(path1: str, path2: str) -> dict:
    """Compare audio levels between two files.

    Shows the dB difference to predict if one sound will be audible
    when mixed with another. Useful for checking if SFX will be heard
    over narration or background music.

    Args:
        path1: Path to first audio file.
        path2: Path to second audio file.

    Returns:
        Dict with:
        - path1: First file path
        - path2: Second file path
        - path1_mean_db: Mean dB of first file
        - path2_mean_db: Mean dB of second file
        - difference_db: dB difference (positive if path1 is louder)
        - louder_file: Which file is louder
        - audibility_prediction: Prediction of how audible each will be in mix

    Example:
        # Check if SFX will be heard over narration
        result = compare_audio_levels("sfx.wav", "narration.wav")
        print(f"Difference: {result['difference_db']:.1f} dB")
        print(result['audibility_prediction'])
    """
    result = compare_levels(path1, path2)

    return {
        "path1": result.path1,
        "path2": result.path2,
        "path1_mean_db": result.path1_mean_db,
        "path2_mean_db": result.path2_mean_db,
        "difference_db": result.difference_db,
        "louder_file": result.louder_file,
        "audibility_prediction": result.audibility_prediction,
    }


@mcp.tool()
def overlay_multiple_tracks(
    base_path: str,
    overlays: list[dict],
    output_path: str,
) -> dict:
    """Overlay multiple audio tracks onto a base track in one call.

    More efficient than chaining multiple overlay_audio_track calls. Places
    all SFX/music at their specified positions in a single ffmpeg pass.

    Args:
        base_path: Path to the base audio file (e.g., narration).
        overlays: List of overlay dicts, each with:
            - path: Path to overlay audio file (required)
            - position_ms: Position in base where overlay starts (default: 0)
            - volume: Volume multiplier for overlay (default: 1.0)
        output_path: Path for the output file.

    Returns:
        Dict with:
        - base_path: Base audio path
        - output_path: Output file path
        - duration_ms: Duration of output
        - overlay_count: Number of overlays applied
        - overlays_applied: List of overlay details

    Example:
        # Add multiple SFX to narration in one call
        result = overlay_multiple_tracks(
            base_path="narration.wav",
            overlays=[
                {"path": "door_open.wav", "position_ms": 1500, "volume": 0.8},
                {"path": "footsteps.wav", "position_ms": 3000, "volume": 0.6},
                {"path": "thunder.wav", "position_ms": 8000, "volume": 1.0},
            ],
            output_path="scene_with_sfx.wav",
        )
    """
    resolved_output = resolve_output_path(output_path)

    result = overlay_multiple(
        base_path=base_path,
        overlays=overlays,
        output_path=resolved_output,
    )

    return {
        "base_path": result.base_path,
        "output_path": result.output_path,
        "duration_ms": result.duration_ms,
        "overlay_count": result.overlay_count,
        "overlays_applied": result.overlays_applied,
    }


@mcp.tool()
def batch_normalize_sfx_to_lufs(
    audio_paths: list[str],
    target_lufs: float = -20.0,
    output_dir: Optional[str] = None,
) -> dict:
    """Normalize multiple audio files to the same LUFS target.

    Ensures all files in a batch have consistent loudness levels,
    making them easier to mix together. Uses ffmpeg's loudnorm filter.

    Args:
        audio_paths: List of paths to audio files to normalize.
        target_lufs: Target integrated loudness in LUFS. Default: -20.
            Common values: -16 (broadcast), -14 (streaming), -20 (SFX mixing).
        output_dir: Directory for output files. If None, outputs are created
            in the same directory as inputs with '_lufs' suffix.

    Returns:
        Dict with:
        - total: Total number of files processed
        - succeeded: Number successfully normalized
        - failed: Number that failed
        - target_lufs: Target LUFS used
        - results: List of result dicts for each file

    Example:
        # Normalize all SFX to -20 LUFS for consistent mixing
        result = batch_normalize_sfx_to_lufs(
            ["explosion.wav", "footstep.wav", "door.wav"],
            target_lufs=-20,
            output_dir="/path/to/normalized/"
        )
    """
    resolved_output_dir = resolve_output_path(output_dir) if output_dir else None

    results = batch_normalize_to_lufs(
        audio_paths=audio_paths,
        target_lufs=target_lufs,
        output_dir=resolved_output_dir,
    )

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - succeeded

    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "target_lufs": target_lufs,
        "results": results,
    }


# ============================================================================
# Sound Effect Analysis Tools
# ============================================================================


@mcp.tool()
def check_sfx_analysis_availability() -> dict:
    """Check if SFX analysis tools are available.

    Returns status of analysis capabilities for non-speech audio:
    - Loudness analysis (peak, RMS, LUFS, dynamic range)
    - Clipping detection
    - Spectral analysis
    - Silence detection
    - Format validation

    Returns:
        Dict with available tools and requirements.
    """
    return get_sfx_analysis_info()


@mcp.tool()
def analyze_audio_loudness(audio_path: str) -> dict:
    """Analyze loudness characteristics of an audio file.

    Measures peak level, RMS, integrated loudness (LUFS), dynamic range,
    and true peak. Use this to verify audio levels meet broadcast or
    game audio standards.

    Args:
        audio_path: Path to audio file to analyze.

    Returns:
        Dict with:
        - status: "success" or "error"
        - peak_db: Peak level in dBFS
        - peak_linear: Peak level as linear value (0-1)
        - rms_db: RMS level in dBFS
        - lufs: Integrated loudness in LUFS
        - dynamic_range_db: Difference between peak and RMS
        - true_peak_db: Inter-sample true peak in dBTP
        - is_clipping: True if peak >= 0 dBFS

    Level guidelines:
    - Broadcast: -24 to -16 LUFS
    - Streaming: -14 to -16 LUFS
    - Game SFX: -6 to -12 dBFS peak
    - Headroom: Keep peaks below -1 dBFS
    """
    result = analyze_loudness(audio_path)
    return result.to_dict()


@mcp.tool()
def detect_audio_clipping(
    audio_path: str,
    threshold: float = 0.99,
    min_consecutive: int = 2,
) -> dict:
    """Detect clipping (digital distortion) in audio.

    Finds samples at or near maximum amplitude that indicate clipping.
    Clipping causes harsh distortion and should be avoided in final audio.

    Args:
        audio_path: Path to audio file to analyze.
        threshold: Amplitude threshold for clipping (0-1, default 0.99).
        min_consecutive: Minimum consecutive clipped samples to count as a region.

    Returns:
        Dict with:
        - status: "success" or "error"
        - has_clipping: True if clipping detected
        - clipped_samples: Number of clipped samples
        - clipped_percentage: Percentage of samples clipped
        - clipped_regions: List of clipping regions with timestamps
        - max_consecutive_clipped: Longest run of clipped samples

    If clipping is detected:
    - Reduce input gain before processing
    - Use a limiter instead of hard clipping
    - Re-record at lower levels
    """
    result = detect_clipping(audio_path, threshold=threshold, min_consecutive=min_consecutive)
    return result.to_dict()


@mcp.tool()
def analyze_audio_spectrum(audio_path: str) -> dict:
    """Analyze spectral characteristics of audio.

    Measures frequency content, brightness, and energy distribution.
    Useful for understanding the tonal qualities of sound effects.

    Args:
        audio_path: Path to audio file to analyze.

    Returns:
        Dict with:
        - status: "success" or "error"
        - dominant_frequency_hz: Most prominent frequency
        - frequency_centroid_hz: Spectral centroid (brightness)
        - bandwidth_hz: Spectral bandwidth (frequency spread)
        - low_freq_energy: Energy ratio in 20-250 Hz (bass)
        - mid_freq_energy: Energy ratio in 250-4000 Hz (mids)
        - high_freq_energy: Energy ratio in 4000-20000 Hz (highs)
        - rolloff_hz: Frequency below which 85% of energy exists
        - zero_crossing_rate: Rate of sign changes (noisiness indicator)

    Interpretation:
    - High centroid = bright/harsh sound
    - High ZCR = noisy/percussive
    - High low_freq_energy = bassy/deep
    """
    result = analyze_spectrum(audio_path)
    return result.to_dict()


@mcp.tool()
def detect_audio_silence(
    audio_path: str,
    threshold_db: float = -40.0,
    min_silence_ms: float = 100.0,
) -> dict:
    """Detect silence regions in audio.

    Finds leading silence, trailing silence, and gaps within the audio.
    Useful for trimming sound effects or detecting editing issues.

    Args:
        audio_path: Path to audio file to analyze.
        threshold_db: dB threshold below which audio is silent (default -40).
        min_silence_ms: Minimum duration to count as silence (default 100ms).

    Returns:
        Dict with:
        - status: "success" or "error"
        - leading_silence_ms: Silence at start
        - trailing_silence_ms: Silence at end
        - total_silence_ms: Total silence duration
        - silence_percentage: Percentage of audio that is silent
        - silence_regions: List of silence regions with timestamps
        - content_start_ms: Where actual content begins
        - content_end_ms: Where actual content ends
        - content_duration_ms: Duration of non-silent content

    Use cases:
    - Trim leading/trailing silence from SFX
    - Find gaps in audio that need fixing
    - Verify sound effect has proper timing
    """
    result = detect_silence(audio_path, threshold_db=threshold_db, min_silence_ms=min_silence_ms)
    return result.to_dict()


@mcp.tool()
def validate_audio_format(
    audio_path: str,
    target_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
    target_bit_depth: Optional[int] = None,
    min_duration_ms: Optional[float] = None,
    max_duration_ms: Optional[float] = None,
    max_file_size_bytes: Optional[int] = None,
) -> dict:
    """Validate audio format against target specifications.

    Checks sample rate, channels, bit depth, duration, and file size
    against provided targets. Reports any mismatches with recommendations.

    Args:
        audio_path: Path to audio file to validate.
        target_sample_rate: Required sample rate (e.g., 44100, 48000).
        target_channels: Required channels (1=mono, 2=stereo).
        target_bit_depth: Required bit depth (16, 24, 32).
        min_duration_ms: Minimum duration in milliseconds.
        max_duration_ms: Maximum duration in milliseconds.
        max_file_size_bytes: Maximum file size in bytes.

    Returns:
        Dict with:
        - status: "success" or "error"
        - is_valid: True if all checks pass
        - sample_rate: Actual sample rate
        - channels: Actual channel count
        - bit_depth: Actual bit depth (None for compressed)
        - duration_ms: Actual duration
        - format: File format
        - file_size_bytes: File size
        - issues: List of validation issues
        - recommendations: Suggested fixes

    Common targets:
    - Game audio: 44100 Hz, 16-bit, mono for SFX
    - Broadcast: 48000 Hz, 24-bit, stereo
    - Web: 44100 Hz, max 5MB file size
    """
    result = validate_format(
        audio_path,
        target_sample_rate=target_sample_rate,
        target_channels=target_channels,
        target_bit_depth=target_bit_depth,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        max_file_size_bytes=max_file_size_bytes,
    )
    return result.to_dict()


# ============================================================================
# Audio Asset Management Tools
# ============================================================================


@mcp.tool()
def list_asset_sources() -> dict:
    """List all available audio asset sources.

    Returns information about each source including:
    - Name and description
    - Whether it requires API key configuration
    - Supported asset types (sfx, music, ambience)
    - Availability status

    Returns:
        Dict with sources mapping source_id to source info.
    """
    sources = list_sources()
    return {
        "sources": {source_id: info.to_dict() for source_id, info in sources.items()},
    }


@mcp.tool()
async def search_audio_assets(
    query: str,
    asset_type: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[list[str]] = None,
    min_duration_secs: Optional[float] = None,
    max_duration_secs: Optional[float] = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """Search for audio assets (sound effects, music, ambience) across all sources.

    Searches both local indexed folders and remote sources like Freesound.org.
    Results include license information for proper attribution.

    Args:
        query: Search query string (e.g., "explosion", "forest ambience", "piano loop").
        asset_type: Filter by type - "sfx", "music", or "ambience". If None, searches all types.
        source: Limit to specific source - "local" or "freesound". If None, searches all.
        tags: Filter by tags (e.g., ["impact", "loud"]).
        min_duration_secs: Minimum duration in seconds.
        max_duration_secs: Maximum duration in seconds.
        page: Page number for pagination (1-indexed).
        page_size: Number of results per page (max 100).

    Returns:
        Dict with:
        - assets: List of matching assets with full metadata
        - total_count: Total number of matches across all pages
        - page: Current page number
        - page_size: Results per page
        - has_more: Whether more results are available

    Example:
        Search for short explosion sounds:
        search_audio_assets("explosion", asset_type="sfx", max_duration_secs=3.0)
    """
    result = await search_assets_async(
        query=query,
        asset_type=asset_type,
        source=source,
        tags=tags,
        min_duration_secs=min_duration_secs,
        max_duration_secs=max_duration_secs,
        page=page,
        page_size=min(page_size, 100),
    )
    return result.to_dict()


@mcp.tool()
async def get_audio_asset(asset_id: str) -> dict:
    """Get detailed information about a specific audio asset.

    Args:
        asset_id: The asset ID (format: "source:id", e.g., "freesound:12345" or "local:abc123").

    Returns:
        Dict with full asset details including:
        - name, description, tags
        - duration, format, sample_rate, channels
        - license information (type, attribution requirements)
        - local_path (if downloaded)
        - preview_url (for remote assets)

    Returns error if asset not found.
    """
    asset = await get_asset_async(asset_id)
    if asset is None:
        return {"status": "error", "error": f"Asset not found: {asset_id}"}
    return {"status": "success", "asset": asset.to_dict()}


@mcp.tool()
async def download_audio_asset(
    asset_id: str,
    output_path: Optional[str] = None,
) -> dict:
    """Download an audio asset to local storage.

    For Freesound assets, downloads the high-quality preview (full download requires OAuth2).
    For local assets, returns the existing path.

    Args:
        asset_id: The asset ID to download (e.g., "freesound:12345").
        output_path: Optional output path. If not provided, saves to the asset library
            downloads folder with auto-generated filename.

    Returns:
        Dict with:
        - status: "success" or "error"
        - local_path: Path to the downloaded file
        - already_local: True if file was already downloaded
        - asset_id: The asset ID

    Note: Check the asset's license info for attribution requirements before use.
    """
    return await download_asset_async(asset_id, output_path)


@mcp.tool()
async def import_audio_folder(
    folder_path: str,
    asset_type: Optional[str] = None,
    recursive: bool = True,
) -> dict:
    """Import audio files from a folder into the asset library.

    Scans the folder for audio files (wav, mp3, ogg, flac, m4a, aac, wma, aiff, opus)
    and indexes them for searching. Automatically extracts:
    - Duration, sample rate, channels, file size
    - Tags from filename and folder structure
    - Asset type from path keywords (sfx, music, ambience)

    Args:
        folder_path: Path to folder containing audio files.
        asset_type: Default asset type ("sfx", "music", "ambience").
            If not specified, auto-detected from path keywords.
        recursive: Whether to scan subdirectories (default: True).

    Returns:
        Dict with:
        - status: "success" or "error"
        - folder: The folder path
        - assets_imported: Number of files indexed
        - recursive: Whether subdirectories were scanned

    Example:
        import_audio_folder("/path/to/sound-effects", asset_type="sfx")
    """
    return await import_folder_async(
        folder_path=folder_path,
        asset_type=asset_type,
        recursive=recursive,
        auto_tag=False,
    )


@mcp.tool()
def configure_freesound_api(api_key: str) -> dict:
    """Configure the Freesound.org API key for searching and downloading sounds.

    Freesound is a collaborative database of Creative Commons licensed sounds.
    To get API credentials:
    1. Create a free account at https://freesound.org
    2. Apply for API credentials at https://freesound.org/apiv2/apply
    3. Use the "Client secret/Api key" as your API token (NOT the "Client id")

    Note: Freesound's "Token Authentication" uses the "Client secret" as the API token.
    The "Client id" is only needed for OAuth2 flows.

    The API key is stored persistently in the asset library database.

    Args:
        api_key: Your Freesound API token (the "Client secret" from your credentials).

    Returns:
        Dict with configuration status.
    """
    return configure_freesound(api_key)


@mcp.tool()
def configure_jamendo_api(client_id: str) -> dict:
    """Configure the Jamendo client ID for searching and downloading music.

    Jamendo is a platform for independent Creative Commons licensed music with 500k+ tracks.
    To get a client ID:
    1. Create a free account at https://www.jamendo.com
    2. Register your app at https://developer.jamendo.com/v3.0

    Note: The API is free for non-commercial use. For commercial use, contact Jamendo.

    The client ID is stored persistently in the asset library database.

    Args:
        client_id: Your Jamendo client ID from app registration.

    Returns:
        Dict with configuration status.
    """
    return configure_jamendo(client_id)


@mcp.tool()
def set_audio_library_path(path: str) -> dict:
    """Set the asset library path where database and downloads are stored.

    By default, the asset library is stored in ~/Documents/talky-talky/assets/.
    Use this to configure a custom location.

    Args:
        path: Directory path for the asset library, or "default" to reset.

    Returns:
        Dict with:
        - status: "success"
        - database_path: Path to the SQLite database
        - library_path: Root path of the asset library
    """
    return set_asset_library_path(path)


@mcp.tool()
def get_audio_library_path() -> dict:
    """Get the current asset library path configuration.

    Returns:
        Dict with:
        - database_path: Path to the SQLite database
        - library_path: Root path of the asset library
        - exists: Whether the database exists
    """
    return get_asset_library_path()


@mcp.tool()
def add_asset_tags(asset_id: str, tags: list[str]) -> dict:
    """Add tags to an audio asset for better organization and search.

    Tags help categorize and find assets. You can add custom tags like
    "boss-fight", "outdoor", "scary", etc.

    Args:
        asset_id: The asset ID to tag (e.g., "local:abc123").
        tags: List of tags to add.

    Returns:
        Dict with status and tags added.
    """
    return add_tags(asset_id, tags, source="manual")


@mcp.tool()
def remove_asset_tags(asset_id: str, tags: list[str]) -> dict:
    """Remove tags from an audio asset.

    Args:
        asset_id: The asset ID to update.
        tags: List of tags to remove.

    Returns:
        Dict with status and tags removed.
    """
    return remove_tags(asset_id, tags)


@mcp.tool()
def list_all_asset_tags() -> list[dict]:
    """List all tags in the asset library with usage counts.

    Returns:
        List of dicts with:
        - name: Tag name
        - source: How the tag was added ("manual", "ai", "api", "path")
        - count: Number of assets with this tag

    Useful for exploring what tags are available for filtering.
    """
    return list_tags()


@mcp.tool()
async def list_indexed_audio_folders() -> list[dict]:
    """List all folders that have been indexed for audio assets.

    Returns:
        List of folder info dicts with:
        - path: Folder path
        - asset_type: Default asset type for the folder
        - recursive: Whether subdirectories were included
        - file_count: Number of files indexed
        - indexed_at: When the folder was indexed
    """
    return await list_indexed_folders_async()


@mcp.tool()
async def rescan_audio_folder(folder_path: str) -> dict:
    """Rescan an indexed folder for new or modified audio files.

    Use this after adding new files to an indexed folder.

    Args:
        folder_path: Path to the folder to rescan.

    Returns:
        Dict with:
        - status: "success" or "error"
        - folder: The folder path
        - assets_updated: Number of files added/updated
    """
    return await rescan_folder_async(folder_path)


@mcp.tool()
async def remove_indexed_audio_folder(folder_path: str) -> dict:
    """Remove an indexed folder and all its assets from the library.

    This does NOT delete the actual audio files, only removes them from
    the search index.

    Args:
        folder_path: Path to the folder to remove.

    Returns:
        Dict with status ("success", "not_found", or "error").
    """
    return await remove_indexed_folder_async(folder_path)


@mcp.tool()
def check_autotag_availability() -> dict:
    """Check which AI auto-tagging capabilities are available.

    Auto-tagging uses various AI engines to generate semantic tags:
    - Transcription: Extracts keywords from speech content
    - Emotion: Detects emotional tone (happy, sad, angry, etc.)
    - Quality: Assesses audio quality level

    Returns:
        Dict with available capabilities and their descriptions.
    """
    return get_autotag_capabilities()


@mcp.tool()
async def auto_tag_audio_asset(
    asset_id: str,
    use_transcription: bool = True,
    use_emotion: bool = True,
    use_quality: bool = True,
    transcription_model: str = "base",
    max_keywords: int = 8,
) -> dict:
    """Auto-tag an audio asset using AI analysis.

    Analyzes the audio file and generates semantic tags:
    - Transcription-based: Extracts keywords from speech content (e.g., "hello", "world")
    - Emotion detection: Tags detected emotions (e.g., "happy", "angry", "sad")
    - Quality assessment: Tags quality level (e.g., "excellent-quality", "good-quality")

    The generated tags are automatically saved to the asset in the database.

    Args:
        asset_id: The asset ID to tag (e.g., "local:abc123").
        use_transcription: Extract keywords from speech content (requires faster-whisper).
        use_emotion: Detect and tag emotions (requires emotion2vec).
        use_quality: Assess and tag quality level (requires nisqa).
        transcription_model: Whisper model size for transcription.
            Options: "tiny", "base", "small", "medium", "large-v3"
            Larger models are more accurate but slower.
        max_keywords: Maximum number of keywords to extract from transcription.

    Returns:
        Dict with:
        - status: "success" or "error"
        - tags_added: List of generated tags
        - tag_sources: Dict mapping each tag to its source
        - transcription: Full transcribed text (if available)
        - emotion: Detected primary emotion (if available)
        - emotion_confidence: Confidence score for emotion (0-1)
        - quality_score: MOS quality score (1-5)
        - processing_time_ms: Total processing time

    Example:
        auto_tag_audio_asset("local:abc123", transcription_model="small")
    """
    return await auto_tag_asset_async(
        asset_id=asset_id,
        use_transcription=use_transcription,
        use_emotion=use_emotion,
        use_quality=use_quality,
        transcription_model=transcription_model,
        max_keywords=max_keywords,
    )


# ============================================================================
# Server Entry Point
# ============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
