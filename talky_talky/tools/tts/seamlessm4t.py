"""SeamlessM4T v2 Engine - Multilingual TTS with translation support.

SeamlessM4T v2 from Meta AI is a massively multilingual model supporting:
- Text-to-speech synthesis in 35 languages
- Speech-to-speech translation
- Cross-lingual voice synthesis

This engine uses the T2ST (text-to-speech translation) capability for TTS.
Set source and target to the same language for pure TTS.
"""

import sys
from pathlib import Path

from .base import VoiceSelectionEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import (
    split_text_into_chunks,
    get_best_device,
    get_available_memory_gb,
    redirect_stdout_to_stderr,
)


# ============================================================================
# Constants
# ============================================================================

# Audio settings - SeamlessM4T outputs 16kHz audio
SAMPLE_RATE = 16000

# Generation limits
MAX_DURATION_SECS = 60  # SeamlessM4T handles long-form generation
MAX_CHUNK_CHARS = 500  # Conservative limit for reliable generation

# Model identifier
MODEL_ID = "facebook/seamless-m4t-v2-large"

# Languages supported for speech output (35 languages)
# Format: language_code -> (language_name, seamless_code)
SUPPORTED_LANGUAGES = {
    "en": ("English", "eng"),
    "es": ("Spanish", "spa"),
    "fr": ("French", "fra"),
    "de": ("German", "deu"),
    "it": ("Italian", "ita"),
    "pt": ("Portuguese", "por"),
    "pl": ("Polish", "pol"),
    "nl": ("Dutch", "nld"),
    "ru": ("Russian", "rus"),
    "uk": ("Ukrainian", "ukr"),
    "tr": ("Turkish", "tur"),
    "ar": ("Arabic", "arb"),
    "zh": ("Chinese (Mandarin)", "cmn"),
    "ja": ("Japanese", "jpn"),
    "ko": ("Korean", "kor"),
    "hi": ("Hindi", "hin"),
    "bn": ("Bengali", "ben"),
    "th": ("Thai", "tha"),
    "vi": ("Vietnamese", "vie"),
    "id": ("Indonesian", "ind"),
    "ms": ("Malay", "zsm"),
    "tl": ("Tagalog", "tgl"),
    "sw": ("Swahili", "swh"),
    "he": ("Hebrew", "heb"),
    "fa": ("Persian", "pes"),
    "ro": ("Romanian", "ron"),
    "hu": ("Hungarian", "hun"),
    "cs": ("Czech", "ces"),
    "el": ("Greek", "ell"),
    "sv": ("Swedish", "swe"),
    "da": ("Danish", "dan"),
    "fi": ("Finnish", "fin"),
    "no": ("Norwegian", "nob"),
    "sk": ("Slovak", "slk"),
    "bg": ("Bulgarian", "bul"),
}

# Number of speaker voices available in the vocoder
NUM_SPEAKERS = 200  # SeamlessM4T v2 has multiple speaker embeddings


# ============================================================================
# Model Management
# ============================================================================

_model = None
_processor = None


def _load_model():
    """Lazily load SeamlessM4T v2 model."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    device, device_name, _ = get_best_device()
    print(f"Loading SeamlessM4T v2 on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToSpeech

        # Load processor
        _processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Load model with appropriate device settings
        if device == "cuda":
            _model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif device == "mps":
            # MPS doesn't support float16 well for all operations
            _model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(MODEL_ID)
            _model = _model.to(device)
        else:
            # CPU
            _model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(MODEL_ID)

    print("SeamlessM4T v2 loaded successfully", file=sys.stderr, flush=True)
    return _model, _processor


def _concatenate_audio_arrays(audio_arrays: list, sample_rate: int, silence_ms: int = 100):
    """Concatenate audio arrays with silence between them."""
    import numpy as np

    if not audio_arrays:
        raise ValueError("No audio arrays to concatenate")

    if len(audio_arrays) == 1:
        return audio_arrays[0]

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    result_parts = []
    for i, audio in enumerate(audio_arrays):
        result_parts.append(audio)
        if i < len(audio_arrays) - 1:
            result_parts.append(silence)

    return np.concatenate(result_parts)


# ============================================================================
# Engine Implementation
# ============================================================================


class SeamlessM4TEngine(VoiceSelectionEngine):
    """SeamlessM4T v2 Engine - Multilingual TTS with translation support.

    Meta's SeamlessM4T v2 is a 2.3B parameter multimodal translation model
    that supports text-to-speech synthesis in 35 languages. It can also
    translate while generating speech (e.g., English text -> French speech).

    Parameters:
        language (str): Target language code (default: "en").
            Supported: en, es, fr, de, it, pt, and 29 more languages.
        src_language (str): Source language code (default: same as language).
            Set different from language to translate while synthesizing.
        speaker_id (int): Speaker voice index (0-199, default: 0).
            Different IDs produce different voice characteristics.

    License: CC-BY-NC-4.0 (non-commercial use only)
    """

    @property
    def name(self) -> str:
        return "SeamlessM4T v2"

    @property
    def engine_id(self) -> str:
        return "seamlessm4t"

    def is_available(self) -> bool:
        try:
            from transformers import SeamlessM4Tv2ForTextToSpeech  # noqa: F401

            return True
        except ImportError:
            return False

    def get_available_voices(self) -> dict[str, dict]:
        """Get available speaker voices.

        SeamlessM4T uses numeric speaker IDs rather than named voices.
        Each ID produces a different voice characteristic.
        """
        voices = {}
        for i in range(10):  # Show first 10 as examples
            voices[str(i)] = {
                "id": i,
                "description": f"Speaker voice {i}",
                "note": "Try different IDs (0-199) for different voice characteristics",
            }
        voices["info"] = {
            "total_speakers": NUM_SPEAKERS,
            "note": "Use speaker_id parameter (0-199) to select different voices",
        }
        return voices

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="voice_selection",
            description="Multilingual TTS with translation support (35 languages, 200 speakers)",
            requirements="transformers, sentencepiece, torch",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            extra_info={
                "license": "CC-BY-NC-4.0 (Non-commercial)",
                "license_url": "https://huggingface.co/facebook/seamless-m4t-v2-large",
                "model_id": MODEL_ID,
                "model_size": "2.3B parameters",
                "multilingual": True,
                "translation_support": True,
                "num_speakers": NUM_SPEAKERS,
                "supported_languages": {k: v[0] for k, v in SUPPORTED_LANGUAGES.items()},
                "parameters": {
                    "language": {
                        "type": "str",
                        "default": "en",
                        "description": "Target language for speech output",
                        "options": list(SUPPORTED_LANGUAGES.keys()),
                    },
                    "src_language": {
                        "type": "str",
                        "default": "same as language",
                        "description": "Source text language (set different for translation)",
                    },
                    "speaker_id": {
                        "type": "int",
                        "default": 0,
                        "range": [0, NUM_SPEAKERS - 1],
                        "description": "Speaker voice index for different voice characteristics",
                    },
                },
            },
            prompting_guide=PromptingGuide(
                overview=(
                    "SeamlessM4T v2 is Meta's multilingual translation model with TTS capability. "
                    "It supports 35 languages for speech output with 200 different speaker voices. "
                    "Unique feature: can translate text to another language while generating speech. "
                    "License: CC-BY-NC-4.0 (non-commercial use only)."
                ),
                text_formatting=[
                    "Write naturally - handles punctuation well",
                    "Use proper sentence structure for best prosody",
                    "Numbers are handled automatically",
                    "Ensure correct character encoding for non-Latin scripts",
                    "SeamlessM4T does not support emotion tags",
                ],
                emotion_tags={},
                voice_guidance={
                    "speaker_selection": {
                        "method": "Use speaker_id parameter (0-199)",
                        "note": "Different IDs produce different voice characteristics",
                        "recommendation": "Try IDs 0-10 to find preferred voice, then explore more",
                    },
                    "best_practices": [
                        "Test different speaker_id values to find preferred voice",
                        "Speaker characteristics vary by language",
                        "Lower speaker IDs tend to be more neutral",
                    ],
                },
                parameters={
                    "language": {
                        "description": "Target language code for speech output",
                        "type": "str",
                        "default": "en",
                        "options": SUPPORTED_LANGUAGES,
                    },
                    "src_language": {
                        "description": "Source text language (for translation)",
                        "type": "str",
                        "default": "same as language",
                    },
                    "speaker_id": {
                        "description": "Speaker voice index (0-199)",
                        "type": "int",
                        "default": 0,
                        "range": [0, NUM_SPEAKERS - 1],
                    },
                },
                tips=[
                    "For pure TTS, set src_language = language (or omit src_language)",
                    "For translation + TTS, set different src_language and language",
                    "Try speaker_id 0-10 first to find good voices",
                    "Quality is excellent for major languages (EN, ES, FR, DE, etc.)",
                    "Non-commercial license - check before commercial use",
                ],
                examples=[
                    {
                        "use_case": "English TTS",
                        "text": "Hello, how are you today?",
                        "language": "en",
                        "speaker_id": 0,
                        "notes": "Standard English synthesis",
                    },
                    {
                        "use_case": "Spanish TTS",
                        "text": "Buenos dias, como estas?",
                        "language": "es",
                        "speaker_id": 0,
                        "notes": "Spanish synthesis",
                    },
                    {
                        "use_case": "Translation + TTS (English to French)",
                        "text": "Hello, how are you today?",
                        "src_language": "en",
                        "language": "fr",
                        "notes": "Translates English text and speaks in French",
                    },
                ],
            ),
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=5.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~10GB VRAM used. First generation slower due to model loading.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=1.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Large model, requires significant memory.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.2,
                    device_type="cpu",
                    reference_hardware="AMD Ryzen 9 5900X",
                    notes="Very slow on CPU. GPU strongly recommended.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## SeamlessM4T v2 Setup (Multilingual TTS with Translation)

SeamlessM4T v2 from Meta AI is a 2.3B parameter multilingual model supporting
35 languages for speech output with translation capability.

### Installation

```bash
pip install transformers sentencepiece torch
```

Or with uv:
```bash
uv pip install transformers sentencepiece torch
```

Or install with talky-talky:
```bash
pip install talky-talky[seamlessm4t]
```

### Hardware Requirements

- **NVIDIA GPU with CUDA**: Recommended (10GB+ VRAM for float16)
- **Apple Silicon (MPS)**: Supported (needs significant memory)
- **CPU**: Supported but very slow (not recommended)

### Key Features

- 35 languages for speech output
- 200 different speaker voices
- Translation + TTS in one step
- High quality multilingual synthesis

### Supported Languages (35)

English, Spanish, French, German, Italian, Portuguese, Polish, Dutch, Russian,
Ukrainian, Turkish, Arabic, Chinese, Japanese, Korean, Hindi, Bengali, Thai,
Vietnamese, Indonesian, Malay, Tagalog, Swahili, Hebrew, Persian, Romanian,
Hungarian, Czech, Greek, Swedish, Danish, Finnish, Norwegian, Slovak, Bulgarian

### License

**CC-BY-NC-4.0** - Non-commercial use only.
Check license requirements before commercial use.

### First Run

The model (~9GB) will be downloaded automatically on first use.
This may take several minutes depending on your connection.
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        voice: str = "0",  # speaker_id as string for VoiceSelectionEngine interface
        language: str = "en",
        src_language: str | None = None,
        speaker_id: int | None = None,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with SeamlessM4T v2.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            voice: Speaker ID as string (default: "0"). Alternative to speaker_id.
            language: Target language code (default: "en").
            src_language: Source language code. If None, same as language.
            speaker_id: Speaker voice index (0-199). Overrides voice if set.

        Returns:
            TTSResult with status and metadata.
        """
        import numpy as np
        import soundfile as sf

        output_path = Path(output_path)

        # Resolve speaker_id from voice parameter if not explicitly set
        if speaker_id is None:
            try:
                speaker_id = int(voice)
            except ValueError:
                speaker_id = 0

        # Validate speaker_id range
        speaker_id = max(0, min(speaker_id, NUM_SPEAKERS - 1))

        # If src_language not specified, use same as target
        if src_language is None:
            src_language = language

        # Validate inputs
        if not text or not text.strip():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="Text cannot be empty",
            )

        # Validate languages
        if language not in SUPPORTED_LANGUAGES:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Unsupported target language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
            )

        if src_language not in SUPPORTED_LANGUAGES:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Unsupported source language: {src_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
            )

        # Get seamless language codes
        tgt_lang_code = SUPPORTED_LANGUAGES[language][1]
        src_lang_code = SUPPORTED_LANGUAGES[src_language][1]

        try:
            model, processor = _load_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load SeamlessM4T v2 model: {e}",
            )

        # Get actual sample rate from model config
        actual_sample_rate = model.config.sampling_rate

        try:
            import torch

            # Check if text needs chunking
            if len(text) <= MAX_CHUNK_CHARS:
                # Process text
                with redirect_stdout_to_stderr():
                    text_inputs = processor(
                        text=text,
                        src_lang=src_lang_code,
                        return_tensors="pt",
                    )

                    # Move inputs to model device
                    device = next(model.parameters()).device
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                    # Generate audio
                    with torch.no_grad():
                        audio_output = model.generate(
                            **text_inputs,
                            tgt_lang=tgt_lang_code,
                            speaker_id=speaker_id,
                        )

                    # Extract audio array
                    audio = audio_output[0].cpu().numpy().squeeze()

                chunks_used = 1
            else:
                # Split into chunks for long text
                chunks = split_text_into_chunks(text, MAX_CHUNK_CHARS)
                print(
                    f"Splitting into {len(chunks)} chunks ({len(text)} chars)",
                    file=sys.stderr,
                    flush=True,
                )

                audio_chunks = []
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i + 1}/{len(chunks)}...", file=sys.stderr, flush=True)

                    with redirect_stdout_to_stderr():
                        text_inputs = processor(
                            text=chunk,
                            src_lang=src_lang_code,
                            return_tensors="pt",
                        )

                        device = next(model.parameters()).device
                        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                        with torch.no_grad():
                            audio_output = model.generate(
                                **text_inputs,
                                tgt_lang=tgt_lang_code,
                                speaker_id=speaker_id,
                            )

                        chunk_audio = audio_output[0].cpu().numpy().squeeze()
                        audio_chunks.append(chunk_audio)

                audio = _concatenate_audio_arrays(audio_chunks, actual_sample_rate, silence_ms=100)
                chunks_used = len(chunks)

            # Ensure audio is correct dtype
            audio = np.asarray(audio, dtype=np.float32)

            # Handle shape
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, actual_sample_rate)

            duration_ms = int(len(audio) / actual_sample_rate * 1000)

            memory_gb, memory_type = get_available_memory_gb()

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=actual_sample_rate,
                chunks_used=chunks_used,
                metadata={
                    "language": language,
                    "language_name": SUPPORTED_LANGUAGES[language][0],
                    "src_language": src_language,
                    "src_language_name": SUPPORTED_LANGUAGES[src_language][0],
                    "is_translation": language != src_language,
                    "speaker_id": speaker_id,
                    "memory_gb": memory_gb,
                    "memory_type": memory_type,
                },
            )

        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Generation failed: {e}",
            )
