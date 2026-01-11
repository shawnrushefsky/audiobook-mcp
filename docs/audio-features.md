# Audio Features

This document covers voice modulation tools and audio asset management.

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

Apply preset voice effects for creative transformation.

```python
from talky_talky.tools.audio import apply_voice_effect

result = apply_voice_effect("voice.wav", "robot.wav", effect="robot")
result = apply_voice_effect("voice.wav", "chorus.wav", effect="chorus", intensity=0.7)
```

**Available Effects:**
| Effect | Description |
|--------|-------------|
| `robot` | Robotic/synthetic voice using ring modulation |
| `chorus` | Choir/ensemble effect with multiple voices |
| `vibrato` | Pitch wobble effect |
| `flanger` | Sweeping phaser effect |
| `telephone` | Lo-fi telephone quality |
| `megaphone` | PA/bullhorn sound |
| `deep` | Deeper voice with bass boost |
| `chipmunk` | Higher pitched, faster voice |
| `whisper` | Soft whisper effect |
| `cave` | Cavernous echo effect |

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
