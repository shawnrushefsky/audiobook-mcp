# Talky Talky - Development Guide

This document provides context for Claude Code and other AI assistants working on this codebase.

## Project Overview

Talky Talky is a Model Context Protocol (MCP) server that provides Text-to-Speech capabilities for AI agents. It features a pluggable engine architecture supporting multiple TTS backends:

- **Maya1**: Text-prompted voice design - create unique voices from natural language descriptions
- **Chatterbox**: Audio-prompted voice cloning - clone voices from reference audio samples

Plus audio utilities for format conversion, concatenation, and normalization.

## Architecture

### Technology Stack

- **Runtime**: Python 3.10+
- **MCP SDK**: `mcp` with FastMCP for server implementation
- **Audio Processing**: ffmpeg for format conversion and concatenation
- **TTS Engines**:
  - Maya1 (local, requires GPU) - voice design from text descriptions
  - Chatterbox (local) - voice cloning from reference audio

### Directory Structure

```
talky_talky/
├── __init__.py
├── server.py             # MCP server entry point, tool registrations
├── tools/
│   ├── __init__.py
│   ├── audio.py          # Audio utilities (convert, concat, info)
│   └── tts/
│       ├── __init__.py   # Public interface, engine registry
│       ├── base.py       # Abstract engine interfaces
│       ├── utils.py      # Shared utilities (chunking, tag conversion)
│       ├── maya1.py      # Maya1 engine implementation
│       └── chatterbox.py # Chatterbox engine implementation
└── utils/
    ├── __init__.py
    └── ffmpeg.py         # ffmpeg wrapper functions
```

### Engine Architecture

The TTS module uses a pluggable engine architecture:

```python
# Base classes in base.py
TTSEngine           # Abstract base for all engines
VoiceDesignEngine   # For text-prompted engines (Maya1)
VoiceCloningEngine  # For audio-prompted engines (Chatterbox)

# Registry in __init__.py
register_engine(MyEngine)  # Register new engines
get_engine("maya1")        # Get engine by ID
generate(text, output, engine="maya1", **kwargs)  # Unified generation
```

### Adding New TTS Engines

1. Create a new file in `talky_talky/tools/tts/` (e.g., `elevenlabs.py`)
2. Implement the appropriate base class:

```python
from .base import VoiceDesignEngine, TTSResult, EngineInfo

class ElevenLabsEngine(VoiceDesignEngine):
    @property
    def name(self) -> str:
        return "ElevenLabs"

    @property
    def engine_id(self) -> str:
        return "elevenlabs"

    def is_available(self) -> bool:
        # Check if dependencies are installed
        try:
            import elevenlabs
            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="voice_design",
            description="Cloud-based voice synthesis",
            # ... other fields
        )

    def generate(self, text, output_path, **kwargs) -> TTSResult:
        # Implementation
        pass
```

3. Register in `__init__.py`:
```python
from .elevenlabs import ElevenLabsEngine
register_engine(ElevenLabsEngine)
```

## MCP Tools

### TTS Engine Tools
- `check_tts_availability` - Check engine status and device info
- `get_tts_engines_info` - Get detailed info about all engines
- `list_available_engines` - List installed engines
- `get_tts_model_status` - Check Maya1 model download status
- `download_tts_models` - Download Maya1 models

### Speech Generation Tools
- `speak_maya1` - Generate speech with voice description
- `speak_chatterbox` - Generate speech with voice cloning

### Audio Utility Tools
- `get_audio_file_info` - Get audio file info (duration, format, size)
- `convert_audio_format` - Convert between formats (wav, mp3, m4a)
- `join_audio_files` - Concatenate multiple audio files
- `normalize_audio_levels` - Normalize to broadcast standard
- `check_ffmpeg_available` - Check ffmpeg installation

## TTS Engines

### Maya1 (Voice Design)

Creates unique voices from natural language descriptions with inline emotion tags.

**Requirements:**
- Python 3.10+
- CUDA GPU with 16GB+ VRAM (best), or MPS (Apple Silicon), or CPU (slow)
- ~10GB disk space for model weights

**Emotion Tags:** `<laugh>`, `<sigh>`, `<gasp>`, `<whisper>`, `<angry>`, `<excited>`, etc.

**Voice Description Example:**
```
"Gruff male pirate, 50s, British accent, low pitch, gravelly, slow pacing"
```

### Chatterbox (Voice Cloning)

Clones voices from reference audio with emotion control.

**Installation:**
```bash
pip install chatterbox-tts
```

**Parameters:**
- `exaggeration`: 0.0-1.0+, controls expressiveness (default 0.5)
- `cfg_weight`: 0.0-1.0, controls pacing (default 0.5)

**Emotion Tags:** `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`

## Installation & Setup

```bash
# Install base package
pip install -e .

# Install with Maya1 TTS support (requires CUDA GPU)
pip install -e ".[maya1]"

# Install with Chatterbox TTS support (voice cloning)
pip install -e ".[chatterbox]"

# Install all TTS engines
pip install -e ".[tts]"

# Install development dependencies
pip install -e ".[dev]"
```

## Running the Server

```bash
# Run the MCP server (communicates via stdio)
talky-talky

# Or run directly
python -m talky_talky.server
```

## Debugging

- The server logs to stderr (stdout is reserved for MCP protocol)
- Use `print(..., file=sys.stderr)` for debug logging
