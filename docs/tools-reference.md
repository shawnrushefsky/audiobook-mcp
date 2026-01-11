# MCP Tools Reference

Complete list of MCP tools provided by Talky Talky.

## TTS Engine Tools

| Tool | Description |
|------|-------------|
| `check_tts_availability` | Check engine status and device info |
| `get_tts_engines_info` | Get detailed info about all engines |
| `list_available_engines` | List installed engines |
| `get_tts_model_status` | Check Maya1 model download status |
| `download_tts_models` | Download Maya1 models |

## Speech Generation Tools

| Tool | Description |
|------|-------------|
| `speak_maya1` | Generate speech with voice description |
| `speak_chatterbox` | Generate speech with voice cloning |
| `speak_chatterbox_turbo` | Fast voice cloning for production |
| `speak_mira` | Fast voice cloning with 48kHz output |
| `speak_xtts` | Multilingual voice cloning (17 languages) |
| `speak_kokoro` | Use pre-built voices (54 voices, 8 languages) |
| `speak_soprano` | Ultra-fast TTS at 2000x realtime (CUDA only) |
| `speak_vibevoice_realtime` | Real-time TTS with ~300ms latency |
| `speak_vibevoice_longform` | Long-form multi-speaker TTS (up to 90 min) |
| `speak_cosyvoice` | Multilingual voice cloning with instruction control |
| `speak_seamlessm4t` | Multilingual TTS with translation (35 languages, 200 speakers) |

## Audio Utility Tools

| Tool | Description |
|------|-------------|
| `get_audio_file_info` | Get audio file info (duration, format, size) |
| `convert_audio_format` | Convert between formats (wav, mp3, m4a) |
| `join_audio_files` | Concatenate multiple audio files (supports gap_ms) |
| `normalize_audio_levels` | Normalize to broadcast standard |
| `check_ffmpeg_available` | Check ffmpeg installation |
| `play_audio` | Play audio file with system's default player |
| `set_output_directory` | Set default directory for saving audio files |
| `get_output_directory` | Get current default output directory |
| `trim_audio_file` | Trim audio with auto-detect mode for silence removal |
| `batch_analyze_silence` | Batch silence detection for multiple files |
| `insert_audio_silence` | Add controlled silence before/after audio |
| `crossfade_join_audio` | Concatenate with smooth crossfade transitions |

## Audio Design Tools

| Tool | Description |
|------|-------------|
| `mix_audio_tracks` | Layer/mix multiple audio tracks together |
| `adjust_audio_volume` | Adjust volume (multiplier or dB) |
| `apply_audio_fade` | Apply fade in/out effects |
| `apply_audio_effects` | Apply effects (EQ, reverb, echo, speed) |
| `overlay_audio_track` | Overlay audio at specific position |

## Voice Modulation Tools

| Tool | Description |
|------|-------------|
| `shift_audio_pitch` | Change pitch without affecting speed (±12 semitones) |
| `stretch_audio_time` | Change speed without affecting pitch (0.5x-2.0x) |
| `apply_voice_effect_preset` | Apply voice effects (robot, chorus, telephone, etc.) |
| `list_voice_effects` | List available voice effect presets |
| `shift_voice_formant` | Change voice character (masculine/feminine) |

## Transcription Tools

| Tool | Description |
|------|-------------|
| `check_transcription_availability` | Check transcription engine status and device info |
| `get_transcription_engines_info` | Get detailed info about all transcription engines |
| `list_available_transcription_engines` | List installed transcription engines |
| `transcribe_audio` | Transcribe audio file to text |
| `transcribe_with_timestamps` | Transcribe with word-level timestamps |
| `verify_tts_output` | Verify TTS audio matches expected text |

## Audio Analysis Tools (TTS Self-Verification)

| Tool | Description |
|------|-------------|
| `check_analysis_availability` | Check analysis engine status and device info |
| `get_analysis_engines_info` | Get detailed info about all analysis engines |
| `analyze_emotion` | Detect emotion in audio (angry, happy, sad, etc.) |
| `analyze_voice_similarity` | Compare two audio files for voice similarity |
| `extract_voice_embedding` | Get voice embedding vector for storage/comparison |
| `analyze_speech_quality` | Assess speech quality (MOS score, noisiness, etc.) |
| `verify_tts_comprehensive` | Combined verification (emotion, similarity, quality, transcription) |

## SFX Analysis Tools (Sound Effects)

| Tool | Description |
|------|-------------|
| `check_sfx_analysis_availability` | Check SFX analysis tools availability |
| `analyze_audio_loudness` | Measure peak, RMS, LUFS, dynamic range, true peak |
| `detect_audio_clipping` | Find clipped samples and regions of digital distortion |
| `analyze_audio_spectrum` | Analyze frequency content, brightness, energy distribution |
| `detect_audio_silence` | Find leading/trailing silence and gaps in audio |
| `validate_audio_format` | Validate sample rate, channels, bit depth against targets |

## Audio Asset Management Tools

| Tool | Description |
|------|-------------|
| `list_asset_sources` | List available asset sources (local, Freesound) |
| `search_audio_assets` | Search for sound effects, music, and ambience |
| `get_audio_asset` | Get detailed info about an asset |
| `download_audio_asset` | Download remote asset to local storage |
| `import_audio_folder` | Import local audio folder into asset library |
| `configure_freesound_api` | Configure Freesound.org API key |
| `configure_jamendo_api` | Configure Jamendo client ID for music |
| `set_audio_library_path` | Set custom library storage path |
| `get_audio_library_path` | Get current library path |
| `add_asset_tags` | Add tags to an asset |
| `remove_asset_tags` | Remove tags from an asset |
| `list_all_asset_tags` | List all tags with usage counts |
| `list_indexed_audio_folders` | List indexed local folders |
| `rescan_audio_folder` | Rescan folder for new files |
| `remove_indexed_audio_folder` | Remove folder from index |
