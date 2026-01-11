# Talky Talky

A comprehensive audio MCP (Model Context Protocol) server for AI agents. Generate speech, transcribe audio, clone voices, analyze speech quality, design soundscapes, and manage audio assets—all through a standardized interface that works with any MCP-compatible client.

## Quick Setup with AI Agents

Copy and paste this prompt to your AI agent (Claude Code, Cursor, Windsurf, etc.) to have it automatically configure Talky Talky:

```
Install and configure the Talky Talky MCP server for audio capabilities.

1. Clone the repo: git clone https://github.com/shawnrushefsky/talky-talky.git
2. Find the full path to uv: which uv (e.g., /Users/username/.local/bin/uv)
3. Add to my MCP configuration:

   For Claude Desktop (GUI app - MUST use full path to uv):
   {
     "mcpServers": {
       "talky-talky": {
         "command": "/full/path/to/uv",
         "args": ["run", "--directory", "<path-to-talky-talky>", "--extra", "macos-full", "talky-talky"]
       }
     }
   }

   For CLI tools (Claude Code, .mcp.json):
   {
     "mcpServers": {
       "talky-talky": {
         "command": "uv",
         "args": ["run", "--directory", "<path-to-talky-talky>", "--extra", "tts", "--extra", "transcription", "--extra", "analysis", "talky-talky"]
       }
     }
   }

4. Replace paths with actual values
5. Restart the application and verify by checking TTS availability

Platform extras:
- macOS: macos-full (TTS + transcription + analysis, excludes CUDA-only engines)
- Linux with CUDA: linux-cuda-full (all engines including CUDA-only)
- CPU only: cpu-full

Requirements: Python 3.11+, ffmpeg, GPU recommended for TTS/transcription engines.
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Available Tools](#available-tools)
  - [TTS Engine Tools](#tts-engine-tools)
  - [Speech Generation Tools](#speech-generation-tools)
  - [Transcription Tools](#transcription-tools)
  - [Audio Analysis Tools](#audio-analysis-tools)
  - [Audio Design Tools](#audio-design-tools)
  - [Voice Modulation Tools](#voice-modulation-tools)
  - [Audio Utility Tools](#audio-utility-tools)
  - [Audio Asset Management Tools](#audio-asset-management-tools)
- [TTS Engine Guide](#tts-engine-guide)
- [Transcription Engine Guide](#transcription-engine-guide)
- [Usage Examples](#usage-examples)
- [Development](#development)
- [License](#license)


## Features

### Text-to-Speech (11 Engines)

| Engine | Description | Languages | License |
|--------|-------------|-----------|---------|
| **Maya1** | Voice design from natural language descriptions | Multi | Apache-2.0 |
| **Chatterbox** | Voice cloning with emotion control | 23 | MIT |
| **Chatterbox Turbo** | Fast voice cloning for production | 23 | MIT |
| **MiraTTS** | Ultra-fast cloning at 100x realtime (CUDA) | Multi | MIT |
| **XTTS-v2** | Cross-language voice cloning | 17 | CPML |
| **Kokoro** | 54 pre-built voices, lightweight | 8 | Apache-2.0 |
| **Soprano** | 2000x realtime speed (CUDA) | EN | Apache-2.0 |
| **VibeVoice Realtime** | ~300ms latency real-time TTS | EN | MIT |
| **VibeVoice Long-form** | Multi-speaker up to 90 min | EN/ZH | MIT |
| **CosyVoice3** | Instruction-controlled cloning | 9 | Apache-2.0 |
| **SeamlessM4T v2** | Translation + TTS, 200 speakers | 35 | CC-BY-NC-4.0 |

### Speech-to-Text (2 Engines)

| Engine | Description | Languages | Speed |
|--------|-------------|-----------|-------|
| **Whisper** | OpenAI's robust ASR via transformers | 99+ | 1x |
| **Faster-Whisper** | CTranslate2-optimized Whisper | 99+ | 4x faster |

### Audio Analysis

- **Speech Quality**: MOS prediction, noisiness, discontinuity, coloration
- **Emotion Detection**: 9 emotions (angry, happy, sad, surprised, etc.)
- **Voice Similarity**: Speaker verification and comparison
- **TTS Verification**: Automated quality checks for generated audio

### SFX Analysis

- **Loudness**: Peak, RMS, LUFS, dynamic range, true peak
- **Clipping Detection**: Find digital distortion regions
- **Spectral Analysis**: Frequency content, brightness, energy distribution
- **Silence Detection**: Leading/trailing silence, gaps, content boundaries

### Audio Design

- **Mixing**: Layer multiple tracks with volume control
- **Effects**: EQ (lowpass, highpass, bass, treble), reverb, echo, speed
- **Fades**: Fade in/out with configurable duration
- **Overlays**: Position-based audio layering
- **Crossfades**: Smooth transitions between segments
- **Trimming**: Auto-detect content boundaries, remove silence
- **Silence Insertion**: Add controlled pauses between segments

### Voice Modulation

- **Pitch Shifting**: Change pitch without affecting speed (±12 semitones)
- **Time Stretching**: Change speed without affecting pitch
- **Voice Effects**: Robot, chorus, vibrato, flanger, telephone, and more
- **Formant Shifting**: Change voice character (masculine/feminine)

> **Note:** Maya1 can achieve pitch, pacing, and timbre variations directly through voice descriptions (e.g., "high pitch, fast pacing, gravelly timbre"). Use the voice modulation tools for post-processing or with engines that don't have built-in voice control.

### Audio Asset Management

- **Local Library**: Index and search local folders with SQLite FTS5
- **Freesound.org**: Search and download CC-licensed sounds
- **Jamendo**: 500k+ CC-licensed music tracks for free (non-commercial)
- **License Tracking**: CC0, CC-BY, CC-BY-NC, CC-BY-SA attribution
- **Tagging**: Manual and AI-powered auto-tagging

### Audio Utilities

- **Format Conversion**: WAV, MP3, M4A with ffmpeg
- **Concatenation**: Join files with optional gaps
- **Normalization**: Broadcast standard (-16 LUFS)
- **Playback**: System default audio player

### Cross-Platform Support

- **CUDA**: Full support with GPU acceleration
- **MPS**: Apple Silicon support for most engines
- **CPU**: Fallback for all engines (slower)


## Installation

### Prerequisites

- **Python 3.11+** (required for TTS library compatibility)
- **ffmpeg** (required for audio processing)
- **GPU** (recommended for TTS and transcription)

> **Don't have Python 3.11+?** Use [uv](https://docs.astral.sh/uv/) which auto-manages Python versions:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> uv run --extra tts talky-talky
> ```

### Platform-Specific Installation

```bash
git clone https://github.com/shawnrushefsky/talky-talky.git
cd talky-talky

# macOS (Apple Silicon or Intel)
pip install -e ".[macos-full]"     # TTS + transcription + analysis

# Linux with NVIDIA CUDA GPU
pip install -e ".[linux-cuda-full]" # All engines including CUDA-only

# CPU only (no GPU)
pip install -e ".[cpu-full]"        # Excludes CUDA-only engines
```

### Individual Components

```bash
# TTS engines
pip install -e ".[maya1]"           # Voice design
pip install -e ".[chatterbox]"      # Voice cloning (includes Turbo)
pip install -e ".[xtts]"            # Multilingual cloning
pip install -e ".[kokoro]"          # Pre-built voices
pip install -e ".[seamlessm4t]"     # Multilingual + translation
pip install -e ".[mira]"            # Fast cloning (CUDA only)
pip install -e ".[soprano]"         # Ultra-fast (CUDA only)

# Transcription
pip install -e ".[whisper]"         # OpenAI Whisper
pip install -e ".[faster-whisper]"  # 4x faster Whisper

# Analysis
pip install -e ".[emotion2vec]"     # Emotion detection
pip install -e ".[resemblyzer]"     # Voice similarity
pip install -e ".[nisqa]"           # Speech quality

# Combined
pip install -e ".[tts]"             # All TTS engines
pip install -e ".[transcription]"   # All transcription engines
pip install -e ".[analysis]"        # All analysis engines
```

### Using uv (Recommended)

```bash
git clone https://github.com/shawnrushefsky/talky-talky.git
cd talky-talky

# Run directly without installing
uv run --extra macos-full talky-talky
```

### Using Docker

```bash
# Basic image (audio utilities only)
docker pull ghcr.io/shawnrushefsky/talky-talky:latest
docker run -i ghcr.io/shawnrushefsky/talky-talky:latest

# With GPU access
docker run -i --gpus all talky-talky-cuda
```


## Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "/full/path/to/uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "macos-full", "talky-talky"]
    }
  }
}
```

> **macOS Note:** GUI apps don't inherit shell PATH. Use full path to uv (`which uv`).

### Claude Code / CLI

Add to `.mcp.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "tts", "--extra", "transcription", "talky-talky"]
    }
  }
}
```


## Available Tools

### TTS Engine Tools

| Tool | Description |
|------|-------------|
| `check_tts_availability` | Check available engines and device info |
| `get_tts_engines_info` | Get detailed engine info and parameters |
| `list_available_engines` | List installed engines |
| `get_tts_model_status` | Check Maya1 model download status |
| `download_tts_models` | Download Maya1 models (~10GB) |

### Speech Generation Tools

| Tool | Description |
|------|-------------|
| `speak_maya1` | Generate with voice description |
| `speak_chatterbox` | Voice cloning with emotion control |
| `speak_chatterbox_turbo` | Fast voice cloning |
| `speak_mira` | Fast cloning, 48kHz (CUDA) |
| `speak_xtts` | Multilingual cloning (17 languages) |
| `speak_kokoro` | Pre-built voices (54 voices) |
| `speak_soprano` | Ultra-fast TTS (CUDA) |
| `speak_vibevoice_realtime` | Real-time TTS |
| `speak_vibevoice_longform` | Long-form multi-speaker |
| `speak_cosyvoice` | Instruction-controlled cloning |
| `speak_seamlessm4t` | Multilingual TTS + translation |

### Transcription Tools

| Tool | Description |
|------|-------------|
| `check_transcription_availability` | Check transcription engine status |
| `get_transcription_engines_info` | Get engine details |
| `list_available_transcription_engines` | List installed engines |
| `transcribe_audio` | Transcribe audio to text |
| `transcribe_with_timestamps` | Transcribe with word-level timing |
| `verify_tts_output` | Verify TTS matches expected text |

### Audio Analysis Tools

#### Speech Analysis

| Tool | Description |
|------|-------------|
| `check_analysis_availability` | Check analysis engine status |
| `get_analysis_engines_info` | Get engine details |
| `analyze_emotion` | Detect emotion (9 categories) |
| `analyze_voice_similarity` | Compare voices for similarity |
| `extract_voice_embedding` | Get voice embedding vector |
| `analyze_speech_quality` | MOS score and quality dimensions |
| `verify_tts_comprehensive` | Combined quality verification |

#### SFX Analysis

| Tool | Description |
|------|-------------|
| `check_sfx_analysis_availability` | Check SFX tools availability |
| `analyze_audio_loudness` | Peak, RMS, LUFS, dynamic range |
| `detect_audio_clipping` | Find clipped samples/regions |
| `analyze_audio_spectrum` | Frequency and energy analysis |
| `detect_audio_silence` | Find silence regions and gaps |
| `validate_audio_format` | Validate against target specs |

### Audio Design Tools

| Tool | Description |
|------|-------------|
| `mix_audio_tracks` | Layer multiple tracks together |
| `adjust_audio_volume` | Volume control (multiplier or dB) |
| `apply_audio_fade` | Fade in/out effects |
| `apply_audio_effects` | EQ, reverb, echo, speed |
| `overlay_audio_track` | Position-based audio overlay |
| `crossfade_join_audio` | Smooth transitions between clips |
| `trim_audio_file` | Trim with auto silence detection |
| `insert_audio_silence` | Add controlled pauses |
| `batch_analyze_silence` | Batch silence detection |

### Voice Modulation Tools

| Tool | Description |
|------|-------------|
| `shift_audio_pitch` | Change pitch without affecting speed |
| `stretch_audio_time` | Change speed without affecting pitch |
| `apply_voice_effect_preset` | Apply voice effects (robot, chorus, etc.) |
| `list_voice_effects` | List available voice effect presets |
| `shift_voice_formant` | Change voice character (masculine/feminine) |

### Audio Utility Tools

| Tool | Description |
|------|-------------|
| `get_audio_file_info` | Get duration, format, size |
| `convert_audio_format` | Convert WAV/MP3/M4A |
| `join_audio_files` | Concatenate with optional gaps |
| `normalize_audio_levels` | Normalize to -16 LUFS |
| `check_ffmpeg_available` | Check ffmpeg installation |
| `play_audio` | Play with system player |
| `set_output_directory` | Set default output path |
| `get_output_directory` | Get current output path |

### Audio Asset Management Tools

| Tool | Description |
|------|-------------|
| `list_asset_sources` | List sources (local, Freesound, Jamendo) |
| `search_audio_assets` | Search SFX, music, ambience |
| `get_audio_asset` | Get asset details |
| `download_audio_asset` | Download to local storage |
| `import_audio_folder` | Import folder to library |
| `configure_freesound_api` | Set Freesound API key |
| `configure_jamendo_api` | Set Jamendo client ID for music |
| `add_asset_tags` | Add tags to asset |
| `remove_asset_tags` | Remove tags from asset |
| `list_all_asset_tags` | List all tags with counts |
| `auto_tag_audio_asset` | AI-powered auto-tagging |
| `check_autotag_availability` | Check auto-tag engines |


## TTS Engine Guide

### Maya1 (Voice Design)

Create unique voices from natural language descriptions with 20+ emotion tags.

```python
speak_maya1(
    text="I can't believe it! <laugh> We did it!",
    output_path="output.wav",
    voice_description="Excited young woman, American accent, energetic"
)
```

**Emotion Tags:** `<laugh>` `<chuckle>` `<sigh>` `<gasp>` `<whisper>` `<angry>` `<yell>` `<cry>` `<cough>`

### Chatterbox (Voice Cloning)

Zero-shot voice cloning with emotion control. 23 languages.

```python
speak_chatterbox(
    text="Hello! [chuckle] Nice to meet you.",
    output_path="output.wav",
    reference_audio_paths=["reference.wav"],
    exaggeration=0.6  # 0.0-1.0+ for expressiveness
)
```

### SeamlessM4T v2 (Multilingual + Translation)

Generate speech in 35 languages with optional translation.

```python
# Pure TTS
speak_seamlessm4t(
    text="Hello world",
    output_path="hello.wav",
    language="en"
)

# Translate English to French speech
speak_seamlessm4t(
    text="Hello world",
    output_path="bonjour.wav",
    src_language="en",
    language="fr",
    speaker_id=5  # 200 speaker options
)
```

### Kokoro (Pre-built Voices)

54 high-quality voices, 8 languages, lightweight (82M).

```python
speak_kokoro(
    text="Welcome!",
    output_path="output.wav",
    voice="af_heart",  # American female
    speed=1.0
)
```

**Voice Format:** `[lang][gender]_[name]` (e.g., `af_heart`, `bm_george`, `jf_alpha`)

See [CLAUDE.md](CLAUDE.md) for complete documentation on all engines.


## Transcription Engine Guide

### Faster-Whisper (Recommended)

4x faster than standard Whisper with same accuracy.

```python
transcribe_audio(
    audio_path="speech.wav",
    engine="faster_whisper",
    model_size="large-v3",  # tiny, base, small, medium, large-v3
    language="en"  # or None for auto-detection
)
```

### With Timestamps

```python
transcribe_with_timestamps(
    audio_path="speech.wav",
    word_level=True  # Get word-level timing
)
```

### TTS Verification

```python
verify_tts_output(
    audio_path="generated.wav",
    expected_text="Hello world",
    similarity_threshold=0.8
)
```


## Usage Examples

### Generate Narration with Custom Voice

```
Generate speech saying "Welcome to the future of AI" with a deep male narrator voice using Maya1
```

### Clone a Voice

```
Clone the voice from sample.wav using Chatterbox and say "This is voice cloning" with high expressiveness
```

### Transcribe Audio

```
Transcribe the audio file interview.wav with word-level timestamps
```

### Create a Podcast Intro

```
1. Generate intro music search for "podcast intro" sounds
2. Generate speech: "Welcome to Tech Talk" with a professional voice
3. Mix the music and speech together with music at 20% volume
4. Add fade in/out effects
```

### Analyze Speech Quality

```
Analyze the speech quality of generated.wav and check if it meets broadcast standards
```

### Translate and Speak

```
Use SeamlessM4T to translate "Hello, how are you?" from English to Japanese speech
```

### Voice Modulation

```
1. Generate speech with Maya1 saying "Hello there"
2. Create a robot version using the robot voice effect
3. Create a deeper version by shifting formants to 0.8
4. Create a chipmunk version by pitching up 6 semitones
```


## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check talky_talky
ruff format talky_talky
```

This project uses `uv` for package management.


## License

MIT

Individual TTS engines have their own licenses:
- **Apache-2.0**: Maya1, Kokoro, Soprano, CosyVoice3
- **MIT**: Chatterbox, MiraTTS, VibeVoice
- **CPML**: XTTS-v2 (Coqui Public Model License)
- **CC-BY-NC-4.0**: SeamlessM4T v2 (non-commercial only)


## Contributing

Contributions welcome! Please open an issue or submit a pull request.
