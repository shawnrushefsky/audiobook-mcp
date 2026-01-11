# Song Generation Engines

AI-powered song generation from lyrics and style descriptions.

## LeVo (SongGeneration)

**Engine ID:** `levo`

Tencent's LeVo generates complete songs with vocals and accompaniment from structured lyrics. Based on the paper "LeVo: High-Quality Song Generation with Multi-Preference Alignment" ([arXiv:2506.07520](https://arxiv.org/abs/2506.07520)).

### Features

- **Complete songs**: Generates vocals and accompaniment together
- **Flexible output**: Mixed, separate tracks, vocals-only, or instrumental-only
- **Style control**: Natural language descriptions (e.g., "female, pop, happy")
- **Reference audio**: Clone style from ~10s audio sample
- **Multiple languages**: Chinese and English lyrics

### Model Variants

| Model | Max Duration | VRAM | Languages | Install |
|-------|-------------|------|-----------|---------|
| `base-new` | 2m30s | 10GB | zh, en | Default |
| `base-full` | 4m30s | 12GB | zh, en | Longer songs |
| `large` | 4m30s | 22GB | zh, en | Best quality |

### Requirements

- **CUDA GPU** with 10-28GB VRAM (model dependent)
- First run downloads ~10GB of model weights

### Installation

```bash
# Install dependencies
pip install "talky-talky[songgen]"

# Download models (first-time setup)
# Use the download_songgen_models tool or:
python -c "from talky_talky.tools.songgen.levo import download_models; download_models('base-new')"
```

### Lyrics Format

Lyrics use structure markers to define song sections:

| Marker | Purpose |
|--------|---------|
| `[intro]`, `[intro-short]` | Instrumental introduction |
| `[verse]` | Main verses |
| `[pre-chorus]` | Build-up before chorus |
| `[chorus]` | Main hook/chorus |
| `[bridge]` | Contrasting section |
| `[outro]`, `[outro-short]` | Ending section |
| `[interlude]` | Instrumental break |

**Formatting rules:**
- Separate sections with `;`
- Separate sentences within lyrics with `.`

### Example

```python
lyrics = """
[intro-short] ;
[verse] Hello world. I'm singing today ;
[chorus] This is the chorus. Sing along with me ;
[verse] Second verse now. Different words here ;
[chorus] This is the chorus. Sing along with me ;
[outro-short]
"""

# Using MCP tool
generate_song_levo(
    lyrics=lyrics,
    output_path="my_song.wav",
    description="female, pop, happy, upbeat",
    generate_type="mixed",  # or "separate", "vocal", "bgm"
)
```

### Style Descriptions

Use comma-separated attributes:

- **Voice**: female, male
- **Genre**: pop, rock, jazz, r&b, electronic, folk, classical, hip-hop
- **Mood**: happy, sad, angry, calm, energetic, romantic
- **Instruments**: piano, guitar, drums, strings, synth
- **Tempo**: slow, fast, upbeat, relaxed

Examples:
- `"female, pop, happy, piano, upbeat"`
- `"male, rock, aggressive, guitar, drums"`
- `"female, ballad, sad, piano, slow"`

### Output Types

| Type | Description | Files Created |
|------|-------------|---------------|
| `mixed` | Combined vocals and accompaniment | 1 file |
| `separate` | All three tracks | 3 files (mixed, vocal, bgm) |
| `vocal` | Vocals only (a cappella) | 1 file |
| `bgm` | Accompaniment only (instrumental) | 1 file |

### MCP Tools

| Tool | Description |
|------|-------------|
| `check_songgen_availability` | Check engine status and CUDA device |
| `get_songgen_engines_info` | Get detailed engine information |
| `get_songgen_model_status` | Check model download status |
| `download_songgen_models` | Download model weights |
| `get_songgen_lyrics_format` | Get lyrics format guide |
| `generate_song_levo` | Generate songs from lyrics |

### Low Memory Mode

For GPUs with limited VRAM, enable low memory mode:

```python
generate_song_levo(
    lyrics=lyrics,
    output_path="song.wav",
    description="female, pop, happy",
    low_mem=True,  # Uses more VRAM but slower
)
```

### Style Reference Audio

Clone a style from reference audio (~10 seconds recommended):

```python
generate_song_levo(
    lyrics=lyrics,
    output_path="song.wav",
    prompt_audio_path="/path/to/reference.wav",
)
```

Or use auto-select from built-in styles:

```python
generate_song_levo(
    lyrics=lyrics,
    output_path="song.wav",
    auto_prompt_style="Jazz",  # Pop, Rock, Jazz, R&B, Electronic, Folk, Classical, Hip-Hop, Auto
)
```

### Resources

- [Paper (arXiv)](https://arxiv.org/abs/2506.07520)
- [Demo](https://levo-demo.github.io/)
- [HuggingFace Space](https://huggingface.co/spaces/tencent/SongGeneration)
- [GitHub](https://github.com/tencent-ailab/songgeneration)

---

## ACE-Step

**Engine ID:** `acestep`

ACE-Step is a 3.5B parameter song generation foundation model that creates complete songs with vocals from text prompts and optional lyrics. It supports both Apple Silicon (MPS) and CUDA GPUs.

Based on: "ACE-Step: A Step Towards Music Generation Foundation Model"

### Features

- **Text-to-music**: Generate songs from style descriptions
- **Optional lyrics**: Add structured lyrics with section markers
- **Apple Silicon support**: Works on M1/M2/M3 Max/Ultra with 36GB+ unified memory
- **Up to 4 minutes**: Generate longer songs (max 240 seconds)
- **Multilingual**: Supports English, Chinese, Japanese, Korean, Spanish, French, German

### Requirements

| Platform | Requirements |
|----------|-------------|
| Apple Silicon (MPS) | M1/M2/M3 Max or Ultra, 36GB+ unified memory, macOS 12.3+ |
| CUDA GPU | 12GB+ VRAM (24GB+ recommended), RTX 3090/4090, A100, etc. |

### Installation

**Important:** ACE-Step conflicts with Chatterbox due to diffusers version mismatch. Install in a separate environment.

```bash
# Option 1: Use macos-songgen extra (excludes chatterbox)
pip install "talky-talky[macos-songgen]"
pip install git+https://github.com/ace-step/ACE-Step.git

# Option 2: Fresh environment with ACE-Step only
pip install git+https://github.com/ace-step/ACE-Step.git
pip install talky-talky
```

Models auto-download from HuggingFace (~7GB) on first use.

### Lyrics Format

ACE-Step uses structure markers to define song sections:

| Marker | Purpose |
|--------|---------|
| `[intro]` | Instrumental introduction |
| `[verse]` | Main verses |
| `[pre-chorus]` | Build-up before chorus |
| `[chorus]` | Main hook/chorus |
| `[bridge]` | Contrasting section |
| `[outro]` | Ending section |
| `[instrumental]` | Instrumental break |
| `[hook]` | Short catchy section |
| `[break]` | Musical break |

**Formatting rules:**
- Place markers on their own line before section text
- Separate lines with newlines

### Example

```python
# Using MCP tool - style prompt with optional lyrics
generate_song_acestep(
    prompt="female vocals, pop, upbeat, synth, drums",
    output_path="song.wav",
    lyrics="""
[verse]
Walking down the street today
Feeling like I'm on my way

[chorus]
This is my moment
Nothing can stop me now
""",
    audio_duration=60.0,  # 60 seconds
)

# Instrumental only (no lyrics)
generate_song_acestep(
    prompt="rock, electric guitar, drums, energetic, instrumental",
    output_path="rock_instrumental.wav",
    audio_duration=90.0,
)
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_duration` | 60.0 | Duration in seconds (max 240) |
| `infer_steps` | 27 | Inference steps (27=fast, 60=quality) |
| `guidance_scale` | 15.0 | CFG strength |
| `scheduler_type` | "euler" | Scheduler (euler, heun, pingpong) |
| `seed` | None | Random seed for reproducibility |
| `cpu_offload` | False | Offload to CPU (saves memory) |
| `quantized` | False | Use quantized model (lower quality) |

### Memory-Constrained Setups

For systems with limited memory:

```python
generate_song_acestep(
    prompt="pop, female, happy",
    output_path="song.wav",
    cpu_offload=True,   # Offload weights to CPU
    quantized=True,     # Use quantized model
)
```

This enables running on 8GB VRAM/RAM at reduced quality.

### MCP Tools

| Tool | Description |
|------|-------------|
| `check_songgen_availability` | Check engine status and device info |
| `get_songgen_engines_info` | Get detailed engine information |
| `get_acestep_model_status` | Check ACE-Step model download status |
| `download_acestep_models` | Pre-download ACE-Step models |
| `generate_song_acestep` | Generate songs with ACE-Step |

### Resources

- [GitHub](https://github.com/ace-step/ACE-Step)
- [HuggingFace Model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)

---

## Engine Comparison

| Feature | LeVo | ACE-Step |
|---------|------|----------|
| Model Size | 10-28GB | 3.5B (~7GB) |
| Max Duration | 4m30s | 4m |
| Apple Silicon | No (CUDA only) | Yes (36GB+ RAM) |
| Voice Cloning | Yes (reference audio) | No (text prompts) |
| Output Types | mixed/separate/vocal/bgm | mixed only |
| Languages | zh, en | en, zh, ja, ko, es, fr, de |
| Lyrics Format | Section markers with `;` | Section markers with newlines |
