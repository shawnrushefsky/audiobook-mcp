# Audio Features

This document covers voice modulation tools, audio processing best practices, and audio asset management.

## Sample Rate Handling

When working with TTS output and audio effects, sample rate mismatches can cause static or corruption. All Talky Talky audio tools now preserve sample rates and detect mismatches.

### Common Sample Rates

| Sample Rate | Typical Source |
|-------------|----------------|
| 24000 Hz | Chatterbox, Maya1, most TTS engines |
| 44100 Hz | CD quality, general audio |
| 48000 Hz | Video production, MiraTTS |
| 22050 Hz | Some legacy TTS, low-quality audio |

### Avoiding Sample Rate Issues

**When joining audio files:**
```python
from talky_talky.tools.audio import concatenate_audio, validate_audio_compatibility

# Check compatibility first
result = validate_audio_compatibility(["file1.wav", "file2.wav", "file3.wav"])
if not result.compatible:
    print("Issues:", result.issues)
    print("Fix:", result.recommendation)

# Option 1: Auto-resample when joining
result = concatenate_audio(
    audio_paths=["file1.wav", "file2.wav"],
    output_path="output.wav",
    resample=True,  # Auto-converts to first file's rate
)

# Option 2: Specify target sample rate
result = concatenate_audio(
    audio_paths=["file1.wav", "file2.wav"],
    output_path="output.wav",
    resample=True,
    target_sample_rate=24000,  # Match TTS output
)
```

**Resampling individual files:**
```python
from talky_talky.tools.audio import resample_audio

# Resample voice effect output to match TTS
result = resample_audio("effect_192k.wav", "effect_24k.wav", target_sample_rate=24000)
```

### Recommended Workflow for Production Audio

1. **Generate TTS segments** (outputs at engine's native rate, e.g., 24kHz)
2. **Apply voice effects** (now preserves sample rate)
3. **Check compatibility before joining**
4. **Join segments** with `resample=True` if needed, using WAV format
5. **Normalize final output** (preserves sample rate)

---

## Voice Modulation

Voice modulation tools allow you to transform audio in various ways without affecting other properties.

> **Note:** Maya1 can achieve many voice variations directly through voice descriptions without post-processing. Use these tools for post-processing existing audio, or with engines that don't have built-in voice control (Chatterbox, XTTS, Kokoro, etc.).

### Pitch Shifting

Change the pitch of audio without affecting its duration.

```python
from talky_talky.tools.audio import shift_pitch

# Raise pitch by 4 semitones (major third)
result = shift_pitch("voice.wav", "higher.wav", semitones=4)

# Lower pitch by 5 semitones (perfect fourth)
result = shift_pitch("voice.wav", "lower.wav", semitones=-5)

# Raise by one octave
result = shift_pitch("voice.wav", "octave_up.wav", semitones=12)
```

**Requirements:** `librosa>=0.10.0` (included in analysis extra)

### Time Stretching

Change the speed/duration of audio without affecting pitch.

```python
from talky_talky.tools.audio import stretch_time

# Slow down to 75% speed (longer duration)
result = stretch_time("voice.wav", "slow.wav", rate=0.75)

# Speed up to 125% (shorter duration)
result = stretch_time("voice.wav", "fast.wav", rate=1.25)
```

**Requirements:** `librosa>=0.10.0` (included in analysis extra)

### Voice Effects

Apply preset voice effects for creative transformation. All voice effects preserve the original sample rate of the input file.

```python
from talky_talky.tools.audio import apply_voice_effect

result = apply_voice_effect("voice.wav", "robot.wav", effect="robot")

# PA/intercom announcement
result = apply_voice_effect("voice.wav", "pa.wav", effect="megaphone", intensity=0.4)

# Subtle room ambience
result = apply_voice_effect("voice.wav", "room.wav", effect="cave", intensity=0.15)
```

**Available Effects:**
| Effect | Description | Recommended Intensity |
|--------|-------------|----------------------|
| `robot` | Robotic/synthetic voice using ring modulation | 0.4-0.6 |
| `chorus` | Choir/ensemble effect with multiple voices | 0.3-0.5 |
| `vibrato` | Pitch wobble effect | 0.3-0.5 |
| `flanger` | Sweeping phaser effect | 0.3-0.5 |
| `telephone` | Lo-fi telephone quality | 0.5-0.7 |
| `megaphone` | PA/bullhorn sound | **0.4-0.5** for PA systems |
| `deep` | Deeper voice with bass boost | 0.4-0.6 |
| `chipmunk` | Higher pitched, faster voice | 0.3-0.5 |
| `whisper` | Soft whisper effect | 0.4-0.6 |
| `cave` | Cavernous echo effect | **0.1-0.15** for subtle room ambience |

> **Warning:** The `cave` effect at default intensity (0.5) produces extreme echo unsuitable for PA/intercom voices. Use 0.1-0.15 for subtle room ambience. For PA/announcement systems, use `megaphone` at 0.4-0.5 instead.

**Requirements:** ffmpeg (no additional Python dependencies)

### Formant Shifting

Change voice character (masculine/feminine) without changing pitch.

```python
from talky_talky.tools.audio import shift_formant

# Make voice sound more feminine
result = shift_formant("male.wav", "feminine.wav", shift_ratio=1.2)

# Make voice sound more masculine
result = shift_formant("female.wav", "masculine.wav", shift_ratio=0.85)
```

**Parameters:**
- `shift_ratio < 1.0` = more masculine (deeper resonance)
- `shift_ratio > 1.0` = more feminine (higher resonance)
- Typical range: 0.7 to 1.4

**Requirements:**
- Best quality: `pyworld>=0.3.0` (WORLD vocoder)
- Fallback: `librosa>=0.10.0` (approximation)

### Installation

```bash
# Full voice modulation with pyworld
pip install -e ".[voice-modulation]"

# Voice modulation without pyworld (uses librosa fallback)
pip install -e ".[voice-modulation-lite]"

# Already included if using analysis extra
pip install -e ".[analysis]"
```

---

## Audio Asset Management

The asset management system provides unified access to sound effects, music, and ambience from multiple sources with license tracking.

### Asset Sources

**Local Source:**
- Indexes audio files from local folders
- Supports: wav, mp3, ogg, flac, m4a, aac, wma, aiff, opus
- Auto-extracts tags from filenames and folder structure
- Auto-detects asset type from path keywords (sfx, music, ambience)
- SQLite database with FTS5 full-text search

**Freesound.org Source:**
- Collaborative database of Creative Commons licensed sounds
- Requires free API credentials from https://freesound.org/apiv2/apply
- **Important:** Use the "Client secret/Api key" as the API token (NOT the "Client id")
- Token authentication (no OAuth required for basic access)
- Downloads high-quality MP3 previews (full quality requires OAuth2)

**Jamendo Source:**
- Platform for independent Creative Commons music with 500k+ tracks
- Requires client ID from https://developer.jamendo.com/v3.0
- Free for non-commercial use (contact Jamendo for commercial licensing)
- Full track downloads in MP3 format
- Search by genre, mood, tempo, vocal/instrumental
- Supports artist and album browsing

### Asset Types

- **sfx** - Sound effects (explosions, footsteps, UI sounds)
- **music** - Musical tracks and loops
- **ambience** - Environmental and atmospheric sounds

### License Tracking

All assets include license information for proper attribution:
- **CC0** - Public domain, no attribution required
- **CC-BY** - Attribution required
- **CC-BY-NC** - Attribution required, non-commercial only
- **CC-BY-SA** - Attribution required, share-alike
- **Sampling+** - Creative use allowed with attribution

### Architecture

```python
# Base classes in assets/base.py
Asset              # Core asset dataclass with metadata
AssetType          # Enum: sfx, music, ambience
LicenseInfo        # License type and attribution info
SearchResult       # Paginated search results
AssetSource        # Abstract base for all sources
LocalAssetSource   # Base for local file sources
RemoteAssetSource  # Base for API-based sources

# Sources
LocalSource        # Local filesystem indexer (assets/local.py)
FreesoundSource    # Freesound.org API (assets/freesound.py)

# High-level API in assets/__init__.py
search_assets()        # Unified search across all sources
get_asset()            # Get asset by ID
download_asset()       # Download to local storage
import_folder()        # Import local folder
configure_freesound()  # Set API key
```

### Example Usage

```python
from talky_talky.tools.assets import search_assets, download_asset, import_folder

# Import local sound effects folder
result = await import_folder("/path/to/sfx", asset_type="sfx")
print(f"Imported {result['assets_imported']} files")

# Search for explosion sounds
results = await search_assets("explosion", asset_type="sfx", max_duration_secs=3.0)
for asset in results.assets:
    print(f"{asset.name} ({asset.duration_ms}ms) - {asset.license.license_type}")

# Download a Freesound asset
download = await download_asset("freesound:12345")
print(f"Downloaded to: {download['local_path']}")
```

### Storage

The asset library uses SQLite for indexing with FTS5 full-text search:
- Default location: `~/Documents/talky-talky/assets/`
- Database: `assets.db`
- Downloads: `downloads/` subdirectory
- Configurable via `set_audio_library_path()`
