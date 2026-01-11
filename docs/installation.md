# Installation & Setup

## Platform-Specific Installation (Recommended)

Use platform extras to install all compatible engines for your hardware:

```bash
# macOS (Apple Silicon or Intel) - excludes CUDA-only engines
pip install -e ".[macos-full]"        # TTS + transcription + analysis
pip install -e ".[macos]"             # TTS only
pip install -e ".[macos-transcription]"  # TTS + transcription
pip install -e ".[macos-analysis]"    # TTS + analysis

# Linux/Windows with NVIDIA CUDA GPU - includes all engines
pip install -e ".[linux-cuda-full]"   # TTS + transcription + analysis
pip install -e ".[linux-cuda]"        # TTS only
pip install -e ".[linux-cuda-transcription]"  # TTS + transcription
pip install -e ".[linux-cuda-analysis]"  # TTS + analysis

# CPU-only (no GPU) - same as macOS
pip install -e ".[cpu-full]"          # TTS + transcription + analysis
pip install -e ".[cpu]"               # TTS only
```

## Individual Engine Installation

```bash
# Install base package
pip install -e .

# TTS Engines
pip install -e ".[maya1]"      # Voice design from descriptions
pip install -e ".[chatterbox]" # Voice cloning (includes Turbo)
pip install -e ".[mira]"       # Fast voice cloning (CUDA only)
pip install -e ".[xtts]"       # Multilingual voice cloning
pip install -e ".[kokoro]"     # 54 pre-built voices (requires espeak-ng)
pip install -e ".[soprano]"    # Ultra-fast TTS (CUDA only)

# Transcription Engines
pip install -e ".[whisper]"         # OpenAI Whisper via transformers
pip install -e ".[faster-whisper]"  # 4x faster Whisper

# Analysis Engines (TTS self-verification)
pip install -e ".[emotion2vec]"  # Emotion detection
pip install -e ".[resemblyzer]"  # Voice similarity
pip install -e ".[nisqa]"        # Speech quality assessment

# Combined extras
pip install -e ".[tts]"           # All TTS engines (includes CUDA-only)
pip install -e ".[transcription]" # All transcription engines
pip install -e ".[analysis]"      # All analysis engines

# Development dependencies
pip install -e ".[dev]"
```

## Manual Installation (Dependency Conflicts)

VibeVoice and CosyVoice have dependency conflicts and require separate installation:

```bash
# VibeVoice:
git clone https://github.com/microsoft/VibeVoice.git && cd VibeVoice && pip install -e .

# CosyVoice:
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && cd CosyVoice && pip install -r requirements.txt
```

---

## Running the Server

```bash
# Run the MCP server (communicates via stdio)
uv run talky-talky

# Or run directly
uv run python -m talky_talky.server
```

---

## Configuring MCP Clients

### Claude Desktop (macOS)

**Important:** GUI applications on macOS don't inherit the shell PATH, so you must use the full path to `uv`.

1. Find the full path to uv:
   ```bash
   which uv
   # Example output: /Users/username/.local/bin/uv
   ```

2. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "talky-talky": {
         "command": "/Users/username/.local/bin/uv",
         "args": [
           "run",
           "--directory", "/path/to/talky-talky",
           "--extra", "macos-full",
           "talky-talky"
         ]
       }
     }
   }
   ```

   Available macOS extras:
   - `macos-full` - All TTS, transcription, and analysis engines
   - `macos` - TTS engines only
   - `macos-transcription` - TTS + transcription
   - `macos-analysis` - TTS + analysis

3. Replace paths with actual values and restart Claude Desktop.

### Claude Code / CLI Tools

CLI tools inherit the shell PATH, so you can use `uv` directly. Add to `.mcp.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "tts", "talky-talky"]
    }
  }
}
```

### Linux with CUDA

On Linux with CUDA GPU, you can use `--extra tts` to include all engines:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "tts", "talky-talky"]
    }
  }
}
```
