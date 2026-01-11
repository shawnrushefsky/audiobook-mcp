# Transcription Engines Reference

This document provides detailed reference for speech-to-text engines supported by Talky Talky.

## Whisper (via Transformers)

OpenAI's state-of-the-art speech recognition model via the transformers library.

**Installation:**
```bash
pip install transformers torch
# Or with talky-talky:
pip install talky-talky[whisper]
```

**Features:**
- 99+ languages with automatic detection
- Word-level timestamps
- Works on CUDA, MPS, and CPU
- Best accuracy among open-source models

**Model Sizes:**

| Model | Parameters | VRAM | Relative Speed |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| large-v3-turbo | 809M | ~6GB | 8x |

**Parameters:**
- `model_size`: Model size (default: "base")
- `language`: Language code or None for auto-detection
- `return_timestamps`: True for segments, "word" for word-level

**Recommended Models:**
- **Development/Testing**: base (fast, decent accuracy)
- **Production**: large-v3-turbo (best speed/accuracy balance)
- **Maximum Accuracy**: large-v3

---

## Faster-Whisper (CTranslate2)

CTranslate2-optimized Whisper implementation - 4x faster with same accuracy.

**Installation:**
```bash
pip install faster-whisper
# Or with talky-talky:
pip install talky-talky[faster-whisper]
```

**Features:**
- 4x faster than original Whisper
- Lower memory usage through quantization
- Word-level timestamps with VAD filtering
- Batched inference support

**Model Sizes:**

| Model | Parameters | VRAM | Speed vs large |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| large-v3-turbo | 809M | ~6GB | 8x |
| distil-large-v3 | 756M | ~4GB | 6x |

**Parameters:**
- `model_size`: Model size (default: "base")
- `language`: Language code or None for auto-detection
- `word_timestamps`: Enable word-level timestamps
- `vad_filter`: Filter silence using VAD (default: True)
- `beam_size`: Beam size for decoding (default: 5)

**English-Only Models:**
For English transcription, use .en models (e.g., "base.en") for slightly better accuracy.

**Hardware Support:**
- **NVIDIA GPU with CUDA**: Best performance (float16)
- **CPU**: Good performance with int8 quantization
- **Apple Silicon**: Uses CPU (MPS not directly supported by CTranslate2)
