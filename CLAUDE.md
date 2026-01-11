# Talky Talky - Development Guide

MCP server providing Text-to-Speech, Speech-to-Text, and audio processing for AI agents.

## Quick Reference

| Documentation | Contents |
|--------------|----------|
| [docs/engines/tts.md](docs/engines/tts.md) | TTS engine reference (Maya1, Chatterbox, XTTS, etc.) |
| [docs/engines/songgen.md](docs/engines/songgen.md) | Song generation (LeVo/SongGeneration) |
| [docs/engines/transcription.md](docs/engines/transcription.md) | Whisper, Faster-Whisper reference |
| [docs/engines/analysis.md](docs/engines/analysis.md) | Emotion, voice similarity, quality analysis |
| [docs/adding-engines.md](docs/adding-engines.md) | How to add new TTS engines |
| [docs/tools-reference.md](docs/tools-reference.md) | Complete MCP tools list |
| [docs/audio-features.md](docs/audio-features.md) | Voice modulation, asset management |
| [docs/installation.md](docs/installation.md) | Installation and MCP client setup |

## Project Overview

**Engines:**
- **TTS**: Maya1 (voice design), Chatterbox/Turbo (voice cloning), MiraTTS, XTTS-v2, Kokoro, Soprano, VibeVoice, CosyVoice3, SeamlessM4T
- **Song Generation**: LeVo (complete songs from lyrics, CUDA-only)
- **Transcription**: Whisper, Faster-Whisper (4x faster)
- **Analysis**: Emotion2vec (emotion), Resemblyzer (voice similarity), NISQA (quality)
- **Assets**: Local indexing, Freesound.org, Jamendo (music)

**Audio utilities**: format conversion, concatenation, normalization, trimming, crossfade, mixing, effects, voice modulation (pitch/time/formant shifting)

## Architecture

### Technology Stack

- **Runtime**: Python 3.11+ (required for TTS library compatibility)
- **MCP SDK**: `mcp` with FastMCP
- **Audio Processing**: ffmpeg
- **Package Manager**: `uv`

### Directory Structure

```
talky_talky/
├── server.py             # MCP server entry point
├── tools/
│   ├── audio.py          # Audio utilities
│   ├── tts/              # TTS engines (base.py, utils.py, maya1.py, ...)
│   ├── songgen/          # Song generation (base.py, levo.py)
│   ├── transcription/    # STT engines (whisper.py, faster_whisper.py)
│   ├── analysis/         # Analysis engines (emotion2vec.py, resemblyzer.py, nisqa.py)
│   └── assets/           # Asset management (database.py, local.py, freesound.py)
└── utils/
    └── ffmpeg.py         # ffmpeg wrapper
```

### Engine Architecture

All engine modules use a pluggable architecture with abstract base classes:

```python
# TTS: base.py defines TTSEngine, TextPromptedEngine, AudioPromptedEngine, VoiceSelectionEngine
# SongGen: base.py defines SongGenEngine
# Transcription: base.py defines TranscriptionEngine
# Analysis: base.py defines EmotionEngine, VoiceSimilarityEngine, SpeechQualityEngine

# Registry pattern in each __init__.py
register_engine(MyEngine)
get_engine("engine_id")
```

## Development Conventions

### Using uv

This project uses `uv` for package management:

```bash
uv run talky-talky          # Run the server
uv run python -c "..."      # Run Python code
uv pip install -e ".[dev]"  # Install dev dependencies
uv lock                     # Update lock file after pyproject.toml changes
uvx ruff check .            # Lint
uvx ruff format .           # Format
```

### MCP Protocol Compatibility

**Critical:** MCP communicates via JSON-RPC over stdout. TTS libraries often print to stdout, breaking the protocol.

All engines must use `redirect_stdout_to_stderr()` when importing libraries, loading models, or generating:

```python
from .utils import redirect_stdout_to_stderr

def _load_model():
    with redirect_stdout_to_stderr():
        from some_tts_library import Model
        model = Model.from_pretrained("model-id")
    return model
```

### Logging

- Server logs to stderr (stdout reserved for MCP)
- Use `print(..., file=sys.stderr)` for debug output

### Adding New Engines

See [docs/adding-engines.md](docs/adding-engines.md) for the complete guide. Key steps:

1. Create `talky_talky/tools/tts/<engine>.py` extending base class
2. Register in `__init__.py`
3. Add MCP tool in `server.py`
4. Add dependencies to `pyproject.toml`
5. Run `uv lock`

### Device Compatibility

- Use `get_best_device()` from utils for CUDA/MPS/CPU detection
- CUDA-only engines: check `torch.cuda.is_available()` in `is_available()`
- PyTorch 2.6+: patch `torch.load` with `weights_only=False` for pickled weights

## Quick Start

```bash
# Install with all macOS-compatible engines
pip install -e ".[macos-full]"

# Run the server
uv run talky-talky
```

See [docs/installation.md](docs/installation.md) for platform-specific setup and MCP client configuration.
