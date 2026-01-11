# TTS Engines Reference

This document provides detailed reference for all TTS engines supported by Talky Talky.

## Maya1 (Voice Design)

Creates unique voices from natural language descriptions with inline emotion tags.

**Engine Type:** Text-prompted (voice description)

**Requirements:**
- Python 3.10+
- CUDA GPU with 16GB+ VRAM (best), or MPS (Apple Silicon), or CPU (slow)
- ~10GB disk space for model weights

**Emotion Tags:** `<laugh>`, `<sigh>`, `<gasp>`, `<whisper>`, `<angry>`, `<excited>`, etc.

**Voice Description Example:**
```
"Gruff male pirate, 50s, British accent, low pitch, gravelly, slow pacing"
```

**Built-in Voice Modulation:** Maya1 can achieve voice variations directly through descriptions:
- **Pitch**: "low pitch", "high pitch", "medium-low pitch"
- **Pacing**: "slow pacing", "fast", "measured", "energetic"
- **Timbre**: "gravelly", "smooth", "warm", "bright", "husky", "nasal", "resonant"
- **Character**: "authoritative", "gentle", "menacing", "cheerful"

---

## Chatterbox (Voice Cloning)

Clones voices from reference audio with emotion control.

**Engine Type:** Audio-prompted (voice cloning)

**Installation:**
```bash
pip install chatterbox-tts
```

**Parameters:**
- `exaggeration`: 0.0-1.0+, controls expressiveness (default 0.5)
- `cfg_weight`: 0.0-1.0, controls pacing (default 0.5)

**Emotion Tags:** `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`

---

## Chatterbox Turbo (Fast Voice Cloning)

Streamlined 350M parameter model optimized for low-latency voice cloning.

**Engine Type:** Audio-prompted (voice cloning)

**Installation:**
```bash
pip install chatterbox-tts  # Same package as Chatterbox
```

**Features:**
- Faster inference than standard Chatterbox
- Simpler API (no exaggeration/cfg_weight parameters)
- <200ms production latency
- Works on CUDA, MPS, and CPU

**Emotion Tags:** `[laugh]`, `[chuckle]`, `[cough]`

---

## MiraTTS (Fast Voice Cloning)

Fast voice cloning with high-quality 48kHz output.

**Engine Type:** Audio-prompted (voice cloning)

**Requirements:**
- NVIDIA GPU with CUDA (6GB+ VRAM)
- Does NOT support MPS or CPU

**Features:**
- 48kHz output (higher quality than most TTS)
- Over 100x realtime performance
- Works with only 6GB VRAM

---

## XTTS-v2 (Multilingual Voice Cloning)

Multilingual voice cloning from Coqui supporting 17 languages.

**Engine Type:** Audio-prompted (voice cloning)

**Installation:**
```bash
pip install TTS
```

**Features:**
- Only requires ~6 seconds of reference audio
- Cross-language cloning (clone voice in one language, output in another)
- Works on CUDA, MPS, and CPU

**Supported Languages:**
English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

**Parameters:**
- `language`: Target language code (default: "en")

---

## Kokoro (Voice Selection)

Lightweight 82M parameter TTS with 54 pre-built voices across 8 languages.

**Engine Type:** Voice selection (pre-built voices)

**Installation:**
```bash
pip install kokoro>=0.9.2
# System dependency required:
# Linux: apt-get install espeak-ng
# macOS: brew install espeak-ng
```

**Features:**
- 54 high-quality voices across 8 languages
- No voice cloning or description needed
- Very fast, runs on CPU/GPU/edge devices
- Apache 2.0 licensed

**Languages:** American English, British English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, Portuguese

**Voice ID Format:** `[lang][gender]_[name]` (e.g., `af_heart`, `bm_george`)

**Parameters:**
- `voice`: Voice ID (default: "af_heart")
- `speed`: Speech rate multiplier (default: 1.0)

---

## Soprano (Ultra-Fast TTS)

Ultra-lightweight 80M parameter model with 2000x realtime speed.

**Engine Type:** Single voice (no customization)

**Installation:**
```bash
pip install soprano-tts
```

**Requirements:**
- NVIDIA GPU with CUDA (required)
- Does NOT support MPS or CPU

**Features:**
- 2000x realtime (10 hours audio in <20 seconds)
- High-fidelity 32kHz output
- <15ms streaming latency
- Single built-in voice

**Parameters:**
- `temperature`: Sampling randomness (default: 0.3)
- `top_p`: Nucleus sampling (default: 0.95)
- `repetition_penalty`: Prevents repetition (default: 1.2)

---

## VibeVoice Realtime (Real-time TTS)

Microsoft's real-time TTS with ~300ms first-audio latency.

**Engine Type:** Voice selection (pre-built voices)

**Installation:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Features:**
- ~300ms to first audio (very low latency)
- Up to 10 minutes per generation
- Single speaker with multiple voice options
- Works on CUDA, MPS, and CPU

**Languages:** English (primary), other languages experimental

**Parameters:**
- `speaker_name`: Voice to use (default: "Carter")
- Available: Carter, Emily, Nova, Michael, Sarah

---

## VibeVoice Long-form (Multi-speaker TTS)

Microsoft's long-form TTS for podcasts and conversations.

**Engine Type:** Voice selection (multi-speaker)

**Installation:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Features:**
- Up to 90 minutes per generation
- Multi-speaker support (up to 4 speakers)
- Natural turn-taking for dialogues
- Works on CUDA, MPS, and CPU

**Languages:** English and Chinese

**Parameters:**
- `speaker_name`: Primary speaker (default: "Carter")
- `speakers`: List of speaker names for multi-speaker (max 4)

**Note:** VibeVoice has dependency conflicts with Chatterbox. Install in a separate environment if needed.

---

## CosyVoice3 (Multilingual Voice Cloning)

Alibaba's zero-shot voice cloning with instruction-based control.

**Engine Type:** Audio-prompted (voice cloning with instruction control)

**Installation:**
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt

# Install sox
# Ubuntu: sudo apt-get install sox libsox-dev
# macOS: brew install sox
```

**Features:**
- Zero-shot voice cloning from reference audio
- 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- 18+ Chinese dialects via instruction control
- Cross-lingual cloning
- Fine-grained control with `[breath]` tags

**Parameters:**
- `reference_audio_paths`: Reference audio for voice cloning
- `prompt_text`: Transcript of reference (improves quality)
- `instruction`: Natural language style control
- `language`: Target language code

**Instruction Examples:**
- `"请用广东话表达。"` - Speak in Cantonese
- `"请用尽可能快地语速说。"` - Speak as fast as possible

---

## SeamlessM4T v2 (Multilingual TTS with Translation)

Meta's 2.3B parameter multilingual model with 35 languages for speech output.

**Engine Type:** Multi-speaker with translation

**Installation:**
```bash
pip install transformers sentencepiece torch
# Or with talky-talky:
pip install talky-talky[seamlessm4t]
```

**Features:**
- 35 languages for speech output
- 200 different speaker voices (speaker_id 0-199)
- Translation + TTS in one step (set different src_language and language)
- High quality multilingual synthesis
- Works on CUDA, MPS, and CPU

**Languages (35):**
English, Spanish, French, German, Italian, Portuguese, Polish, Dutch, Russian,
Ukrainian, Turkish, Arabic, Chinese, Japanese, Korean, Hindi, Bengali, Thai,
Vietnamese, Indonesian, Malay, Tagalog, Swahili, Hebrew, Persian, Romanian,
Hungarian, Czech, Greek, Swedish, Danish, Finnish, Norwegian, Slovak, Bulgarian

**Parameters:**
- `language`: Target language code for speech output (default: "en")
- `src_language`: Source text language (for translation, default: same as language)
- `speaker_id`: Speaker voice index 0-199 (default: 0)

**Translation Example:**
```python
# Translate English to French speech
speak_seamlessm4t("Hello world", "bonjour.wav",
                  src_language="en", language="fr")
```

**License:** CC-BY-NC-4.0 (non-commercial use only)
