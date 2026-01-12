"""MCP Resources - documentation and examples on demand."""


def register_resources(mcp):
    """Register MCP resources with the server."""

    @mcp.resource("docs://tts/engines")
    def tts_engines_doc() -> str:
        """Documentation for TTS engines."""
        return """# TTS Engines

## Voice Cloning Engines (require reference audio)

### chatterbox (recommended)
- Best expressiveness with [laugh], [sigh], [gasp], [chuckle], [hmm], [um] tags
- Parameters: exaggeration (0.6-0.7 recommended), cfg_weight
- 10+ seconds reference audio recommended

### chatterbox_turbo
- Faster than chatterbox, simpler API
- Same emotion tags as chatterbox

### mira
- Fast 48kHz output, 100x realtime
- No emotion tags, but high quality audio

### xtts
- 17 languages with cross-language cloning
- Only 6 seconds reference needed

### cosyvoice3
- 9 languages, dialect control via instruction
- Supports [breath] tag for natural pauses

## Voice Design Engine

### maya1
- Describe voice in natural language
- Tags: <laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited>, <sad>
- Example description: "Gruff male pirate, 50s, British accent"

## Preset Voice Engines

### kokoro
- 54 voices across 8 languages
- Voice format: [lang][gender]_[name] (e.g., "af_heart", "bm_george")
- Languages: a=American, b=British, j=Japanese, z=Mandarin

### vibevoice_realtime
- Microsoft real-time TTS, ~300ms latency
- Speakers: Carter, Emily, Nova, Michael, Sarah

### vibevoice_longform
- Up to 90 minutes, 4 speakers
- Good for podcasts and audiobooks

### soprano
- Ultra-fast CUDA only, 2000x realtime
- Best for batch processing

## Translation Engine

### seamlessm4t
- 35 languages, 200 speaker voices
- Can translate while synthesizing
"""

    @mcp.resource("docs://tts/emotion-tags")
    def emotion_tags_doc() -> str:
        """Documentation for emotion tags by engine."""
        return """# Emotion Tags by Engine

## Maya1 (angle bracket tags)
Place these in your text where emotion should occur:
- <laugh> or <chuckle> - laughter
- <sigh> - sighing
- <gasp> - surprise/gasping
- <whisper> - whispering
- <angry> - angry tone
- <excited> - excited tone
- <sad> - sad tone

Example: "Oh my! <gasp> I can't believe it! <laugh>"

## Chatterbox / Chatterbox Turbo (square bracket tags)
Documented tags:
- [laugh] - laughter
- [chuckle] - soft laughter
- [cough] - coughing
- [sigh] - sighing
- [gasp] - gasping
- [groan] - groaning
- [yawn] - yawning
- [sniff] - sniffing
- [clearing throat] - throat clearing

Experimental tags (try these!):
- [hmm], [uh], [um], [oh], [ah]
- [wow], [ooh], [eww], [huh], [mhm]

Example: "Well... [sigh] I suppose you're right. [hmm] Let me think."

## CosyVoice3
- [breath] - natural breathing between phrases

Example: "Hello everyone. [breath] Welcome to the presentation."

## Other Engines
MiraTTS, XTTS, Kokoro, Soprano, VibeVoice, SeamlessM4T do not support emotion tags.
"""

    @mcp.resource("docs://audio/effects")
    def audio_effects_doc() -> str:
        """Documentation for audio effects."""
        return """# Audio Effects

## Voice Effect Presets
Use apply_voice_effect_preset(input, effect, intensity):

- robot: Robotic/synthetic voice
- chorus: Multiple voices/ensemble effect
- vibrato: Pitch wobble/tremolo
- flanger: Sweeping phaser effect
- telephone: Lo-fi phone quality (lowpass + highpass)
- megaphone: PA/bullhorn sound (0.4-0.5 intensity)
- deep: Deeper voice with bass boost
- chipmunk: Higher pitched, cartoonish
- whisper: Soft whisper effect
- cave: Cavernous echo (0.1-0.15 for subtle ambience)

## apply_audio_effects Parameters
- lowpass_hz: Remove highs above frequency (3000 = telephone)
- highpass_hz: Remove lows below frequency (300 = remove rumble)
- bass_gain_db: Boost/cut bass (+6 = boost, -3 = cut)
- treble_gain_db: Boost/cut treble
- speed: Playback speed (affects pitch)
- reverb: Add room ambience
- echo_delay_ms: Echo delay (200 = subtle, 500 = dramatic)

## Pitch/Time Tools
- shift_audio_pitch: Change pitch without speed (+12 = octave up)
- stretch_audio_time: Change speed without pitch (0.5 = half speed)
- shift_voice_formant: Change voice character (<1 = masculine, >1 = feminine)
- autotune_vocals: Pitch correction to key/scale
"""

    @mcp.resource("docs://songgen/lyrics-format")
    def songgen_lyrics_doc() -> str:
        """Documentation for song lyrics format."""
        return """# Song Lyrics Format

## Structure Markers
- [intro], [intro-short]: Instrumental introduction
- [verse]: Main verses
- [pre-chorus]: Build-up before chorus
- [chorus]: Main hook
- [bridge]: Contrasting section
- [outro], [outro-short]: Ending
- [interlude]: Instrumental break

## Separators
- ; separates sections
- . separates sentences within sections

## Example Lyrics
```
[intro-short] ;
[verse] Walking down the street today. Feeling good in every way ;
[pre-chorus] Something's building up inside ;
[chorus] This is my moment. Nothing can stop me now ;
[verse] Second verse with different words. The story continues on ;
[chorus] This is my moment. Nothing can stop me now ;
[bridge] But sometimes I wonder. Where will this road lead ;
[chorus] This is my moment. Nothing can stop me now ;
[outro-short]
```

## Style Descriptions
Comma-separated attributes:
- Gender: male, female
- Genre: pop, rock, jazz, r&b, electronic, folk, classical, hip-hop
- Mood: happy, sad, energetic, calm, aggressive, romantic
- Instruments: piano, guitar, drums, synth, strings
- Tempo: fast, slow, mid-tempo
- Quality: epic, intimate, raw, polished

Example: "female, pop, happy, upbeat, piano, synth"
"""

    @mcp.resource("examples://audiobook-workflow")
    def audiobook_example() -> str:
        """Example audiobook workflow."""
        return """# Audiobook Production Workflow

## 1. Simple: Use create_audiobook_chapter
```python
result = create_audiobook_chapter(
    segments=[
        {"text": "Chapter 1. The Beginning."},
        {"text": "It was a dark and stormy night.", "gap_after_ms": 800},
        {"text": "Thunder rumbled in the distance."},
    ],
    output_path="chapter_01.wav",
    engine="chatterbox",
    reference_audio_paths=["narrator_sample.wav"],
)
```

## 2. Manual: Generate + Join + Normalize
```python
# Generate segments
for i, text in enumerate(texts):
    speak(text, f"seg_{i}.wav", engine="chatterbox",
          reference_audio_paths=["narrator.wav"])

# Trim silence from each
for i in range(len(texts)):
    trim_audio_file(f"seg_{i}.wav", padding_ms=30)

# Join with gaps
join_audio_files(
    [f"seg_{i}_trimmed.wav" for i in range(len(texts))],
    "chapter.wav",
    gap_ms=[400, 400, 800, 400]  # longer gap for scene break
)

# Normalize
normalize_audio_levels("chapter.wav")
```

## 3. With Background Music
```python
result = create_podcast_with_music(
    narration_segments=[...],
    output_path="episode.wav",
    intro_music_path="intro.wav",
    outro_music_path="outro.wav",
    background_music_path="ambient.wav",
    background_volume=0.15,
)
```
"""

    @mcp.resource("examples://voice-cloning")
    def voice_cloning_example() -> str:
        """Example voice cloning workflow."""
        return """# Voice Cloning Examples

## Chatterbox (best for expressive speech)
```python
speak(
    text="That's hilarious! [laugh] Tell me more about it.",
    output_path="response.wav",
    engine="chatterbox",
    reference_audio_paths=["voice_sample.wav"],
    exaggeration=0.65,  # more expressive than default
)
```

## MiraTTS (fast, high quality)
```python
speak(
    text="Welcome to our podcast.",
    output_path="intro.wav",
    engine="mira",
    reference_audio_paths=["host_voice.wav"],
)
```

## XTTS (multilingual)
```python
# Clone English voice, output in Japanese
speak(
    text="こんにちは、元気ですか?",
    output_path="japanese.wav",
    engine="xtts",
    reference_audio_paths=["english_voice.wav"],
    language="ja",
)
```

## CosyVoice (with breathing)
```python
speak(
    text="Let me tell you something. [breath] It's very important.",
    output_path="message.wav",
    engine="cosyvoice3",
    reference_audio_paths=["speaker.wav"],
)
```
"""

    @mcp.resource("examples://sound-design")
    def sound_design_example() -> str:
        """Example sound design workflow."""
        return """# Sound Design Examples

## Create a Scene with Ambient + Events
```python
result = create_sound_effect_scene(
    base_ambience_path="forest_ambient.wav",
    output_path="forest_scene.wav",
    duration_ms=30000,  # 30 seconds
    sound_events=[
        {"path": "owl_hoot.wav", "position_ms": 5000, "volume": 0.7},
        {"path": "branch_snap.wav", "position_ms": 12000, "volume": 0.9},
        {"path": "wolf_howl.wav", "position_ms": 25000, "volume": 0.6},
    ],
)
```

## Layer Multiple Audio Tracks
```python
mix_audio_tracks(
    audio_paths=["narration.wav", "music.wav", "rain_ambient.wav"],
    output_path="final_scene.wav",
    volumes=[1.0, 0.25, 0.4],  # full narration, 25% music, 40% rain
)
```

## Add Voice Effects for Character
```python
# Robot voice
apply_voice_effect_preset("dialogue.wav", effect="robot", intensity=0.7)

# Telephone call
apply_voice_effect_preset("phone_line.wav", effect="telephone", intensity=0.6)

# Mysterious cave echo
apply_voice_effect_preset("cave_speech.wav", effect="cave", intensity=0.15)
```

## Pitch Shift for Character Voices
```python
# Deeper villain voice
shift_audio_pitch("villain.wav", semitones=-3)
shift_voice_formant("villain_pitched.wav", shift_ratio=0.9)

# Higher fairy voice
shift_audio_pitch("fairy.wav", semitones=5)
shift_voice_formant("fairy_pitched.wav", shift_ratio=1.15)
```
"""
