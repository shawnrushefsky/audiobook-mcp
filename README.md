# Talky Talky

An MCP (Model Context Protocol) server for orchestrating full-cast audiobook production with AI voice synthesis. Manage voice assignments, organize characters, import prose text, generate audio with built-in TTS engines, and stitch audio files into complete audiobooks—all through a standardized interface that works with any MCP-compatible client.

## Features

- **Project Management**: Create and manage audiobook projects with metadata
- **Character & Voice Management**: Define characters with voice configurations
- **Chapter & Segment Organization**: Structure your book into chapters and individual speech segments
- **Prose Import**: Import plain text with automatic dialogue detection and paragraph splitting
- **Built-in TTS Engines**:
  - **Maya1**: Voice design via natural language descriptions with 20+ emotion tags
  - **Chatterbox**: High-quality voice cloning with emotion control ([laugh], [cough], etc.)
- **Audio Stitching**: Combine segment audio files into chapters and final audiobooks with chapter markers
- **Async Job System**: Queue-based processing for long-running TTS operations

## Installation

### Prerequisites

- **Python** 3.10 or later
- **ffmpeg** (required for audio stitching features)
- **CUDA GPU** (recommended for TTS, but MPS and CPU also supported)

### Install from Source

```bash
git clone https://github.com/shawnrushefsky/talky-talky.git
cd talky-talky

# Basic installation
pip install -e .

# With Maya1 TTS support (voice design)
pip install -e ".[maya1]"

# With Chatterbox TTS support (voice cloning with emotion)
pip install -e ".[chatterbox]"

# All TTS engines
pip install -e ".[tts]"

# Development dependencies
pip install -e ".[dev]"
```

### Using uv (Recommended)

```bash
git clone https://github.com/shawnrushefsky/talky-talky.git
cd talky-talky

# Install with uv
uv pip install -e ".[tts]"

# Or run directly without installing
uv run --extra tts talky-talky
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "talky-talky"
    }
  }
}
```

Or with uv (no install required):

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "tts", "talky-talky"]
    }
  }
}
```

### Claude Code (CLI)

Add to your project's `.mcp.json`:

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

Or add to `~/.claude/settings.json` for global access:

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "talky-talky"
    }
  }
}
```

### Cursor

Add to Cursor's MCP configuration (Settings → MCP Servers):

```json
{
  "audiobook": {
    "command": "talky-talky"
  }
}
```

### Windsurf

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "talky-talky"
    }
  }
}
```

### Cline (VS Code Extension)

Add to Cline's MCP server settings in VS Code:

```json
{
  "cline.mcpServers": {
    "audiobook": {
      "command": "talky-talky"
    }
  }
}
```

### Continue.dev

Add to your Continue configuration (`~/.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "audiobook",
      "command": "talky-talky"
    }
  ]
}
```

### Zed Editor

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "language_models": {
    "mcp_servers": {
      "audiobook": {
        "command": "talky-talky"
      }
    }
  }
}
```

### Generic MCP Client (Python)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="talky-talky",
    args=[],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # List tools
        tools = await session.list_tools()

        # List resources
        resources = await session.list_resources()

        # Call a tool
        result = await session.call_tool(
            "init_audiobook_project",
            arguments={
                "path": "/path/to/my-audiobook",
                "title": "My Audiobook",
                "author": "Author Name",
            }
        )
```

## Available Tools

### Project Management

| Tool | Description |
|------|-------------|
| `get_suggested_project_path` | Get a suggested project path based on title |
| `init_audiobook_project` | Initialize a new audiobook project in a directory |
| `open_audiobook_project` | Open an existing audiobook project |
| `get_project` | Get project details and statistics |
| `update_audiobook_project` | Update project metadata (title, author, description) |

### Character Management

| Tool | Description |
|------|-------------|
| `create_character` | Add a new character with name and description |
| `get_characters` | List all characters with segment counts |
| `modify_character` | Update character details |
| `remove_character` | Remove a character |
| `set_character_voice` | Assign voice configuration (provider, voice_ref, settings) |
| `clear_character_voice` | Remove voice assignment from a character |

### Voice Sample Management

| Tool | Description |
|------|-------------|
| `create_voice_samples` | Generate voice samples using Maya1 for voice cloning |
| `add_external_voice_sample` | Add a voice sample from a local path or URL |
| `get_character_voice_samples` | List all voice samples for a character |
| `modify_voice_sample` | Update voice sample metadata |
| `remove_voice_sample` | Delete a specific voice sample |
| `remove_all_voice_samples` | Delete all voice samples for a character |
| `get_voice_samples_summary` | Get summary of voice samples for a character |

### Chapter Management

| Tool | Description |
|------|-------------|
| `create_chapter` | Add a new chapter |
| `get_chapters` | List all chapters with segment statistics |
| `modify_chapter` | Update chapter title |
| `remove_chapter` | Remove a chapter and all its segments |
| `reorder_book_chapters` | Change chapter order |

### Segment Management

| Tool | Description |
|------|-------------|
| `create_segment` | Add a text segment to a chapter |
| `get_chapter_segments` | List segments in a chapter with character info |
| `modify_segment` | Update segment text or assigned character |
| `remove_segment` | Remove a segment |
| `reorder_chapter_segments` | Change segment order within a chapter |
| `get_segments_without_audio` | Get all segments missing audio files |
| `bulk_create_segments` | Add multiple segments at once |

### Content Import

| Tool | Description |
|------|-------------|
| `import_text_to_chapter` | Import screenplay-format text with automatic splitting |
| `assign_dialogue_to_character` | Bulk assign character to segments matching a pattern |
| `get_character_lines` | Export all lines for a character |
| `detect_dialogue_in_chapter` | Analyze text and suggest character assignments |
| `get_character_line_distribution` | Get character line count statistics |

### TTS & Audio Generation

| Tool | Description |
|------|-------------|
| `check_tts_availability` | Check which TTS engines are available |
| `get_tts_info` | List available engines, emotion tags, and presets |
| `create_voice_description` | Build a Maya1 voice description from parameters |
| `get_tts_model_status` | Check if TTS models are downloaded |
| `download_tts_models` | Download Maya1 model weights |
| `generate_audio_for_segment` | Generate audio for a single segment |
| `generate_batch_segment_audio` | Generate audio for multiple segments |
| `get_job_status` | Check status of async audio generation job |
| `list_jobs` | List all async jobs |

### Audio Stitching

| Tool | Description |
|------|-------------|
| `register_audio_for_segment` | Link an external audio file to a segment |
| `get_audio_status_for_chapter` | Check which segments have/need audio |
| `stitch_chapter_audio` | Combine segment audio into chapter MP3 |
| `get_book_stitch_status` | Get overall book audio readiness status |
| `stitch_full_audiobook` | Create final audiobook with chapter markers |
| `clear_audio_from_segment` | Remove audio association from a segment |

## Available Resources

| Resource URI | Description |
|--------------|-------------|
| `audiobook://project` | Current project information and statistics |
| `audiobook://characters` | All characters with voice configurations |
| `audiobook://chapters` | All chapters with segment counts |
| `audiobook://tts/status` | TTS engine availability and configuration |

## Available Prompts

| Prompt | Description |
|--------|-------------|
| `audiobook_setup` | Guided workflow for setting up a new audiobook project |
| `voice_design` | Help designing a voice description for a character |
| `chapter_workflow` | Guide for processing a chapter end-to-end |

## Workflow Example

### 1. Initialize a Project

```
Use init_audiobook_project to create a new audiobook project at /path/to/my-book with title "The Great Adventure" by "Jane Author"
```

### 2. Add Characters

```
Add a character named "Narrator" and mark them as the narrator.
Add a character named "Alice" with description "The protagonist, a young adventurer"
Add a character named "Bob" with description "Alice's mentor, an elderly wizard"
```

### 3. Generate Voice Samples with Maya1

```
Create voice samples for Alice with this voice description:
"Realistic female voice in the 20s age with american accent. Medium-high pitch, bright timbre, energetic pacing, enthusiastic tone."

Use these sample texts:
1. "The forest stretched endlessly before me, ancient trees reaching toward a sky I could barely see through the canopy."
2. "I can't believe we actually made it! <laugh> After everything we've been through!"
3. "Look, I know you think I'm not ready, but I've trained my whole life for this moment."
```

### 4. Add Chapters and Import Text

```
Add a chapter titled "Chapter 1: The Beginning"
Import this screenplay-format text into Chapter 1:

NARRATOR: Alice walked through the forest, her heart pounding with anticipation.
ALICE: I can't believe I'm finally here.
NARRATOR: Bob appeared from behind an ancient oak tree.
BOB: You made it. I knew you would.
```

### 5. Generate Audio

```
Generate audio for all segments in Chapter 1 using the chatterbox engine.
```

### 6. Check Status and Stitch

```
Check the book stitch status to see if all segments have audio.
Stitch Chapter 1 into a single audio file.
When all chapters are ready, stitch the complete book with chapter markers.
```

## TTS Engine Guide

### Maya1 (Voice Design)

Maya1 creates unique voices from natural language descriptions. Best for generating voice samples.

**Voice Description Format:**
```
Realistic [gender] voice in the [age] age with [accent] accent.
[pitch] pitch, [timbre] timbre, [pacing] pacing, [tone] tone.
```

**Emotion Tags (inline):**
```
<laugh> <chuckle> <sigh> <gasp> <whisper> <angry> <yell> <cry> <cough>
```

**Example:**
```
"I can't believe it worked! <laugh> We actually did it!"
```

### Chatterbox (Voice Cloning - Recommended for Production)

Chatterbox clones voices from reference audio with emotion control. Supports paralinguistic tags.

**Tags:**
```
[laugh] [chuckle] [cough] [sigh] [gasp]
```

**Parameters:**
- `exaggeration`: 0.0-1.0, controls expressiveness (default 0.5)
- `cfg_weight`: Controls pacing, lower = slower (default 0.5)

## Project Storage Structure

Each project creates a `.audiobook` folder in your specified directory:

```
your-project/
└── .audiobook/
    ├── db.sqlite              # Project database
    ├── audio/
    │   ├── segments/          # Individual segment audio files
    │   └── voice_samples/     # Voice reference samples
    └── exports/
        ├── chapters/          # Stitched chapter audio
        └── book/              # Final audiobook output
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=talky_talky

# Lint and format
ruff check talky_talky
ruff format talky_talky
```

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
