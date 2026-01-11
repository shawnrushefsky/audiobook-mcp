# Audio Analysis Engines Reference

These engines enable agents to self-verify TTS output quality.

## Emotion2vec (Emotion Detection)

State-of-the-art speech emotion recognition using FunASR's emotion2vec_plus_large model.

**Installation:**
```bash
pip install funasr modelscope
# Or with talky-talky:
pip install talky-talky[emotion2vec]
```

**Features:**
- ~300M parameters, trained on 40k hours of speech emotion data
- 9 emotion categories with confidence scores
- Works on CUDA, MPS, and CPU
- ACL 2024 paper

**Supported Emotions:**
angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown

**Example Usage:**
```python
from talky_talky.tools.analysis import detect_emotion

result = detect_emotion("speech.wav")
if result.status == "success":
    print(f"Emotion: {result.primary_emotion} ({result.primary_score:.1%})")
```

---

## Resemblyzer (Voice Similarity)

Voice similarity comparison using 256-dimensional speaker embeddings.

**Installation:**
```bash
pip install resemblyzer
# Or with talky-talky:
pip install talky-talky[resemblyzer]
```

**Features:**
- ~1000x realtime on GPU
- 256-dimensional speaker embeddings
- Cosine similarity comparison
- Default threshold: 0.75 for same-speaker determination

**Example Usage:**
```python
from talky_talky.tools.analysis import compare_voices

result = compare_voices("reference.wav", "generated.wav")
if result.status == "success":
    print(f"Similarity: {result.similarity_score:.1%}")
    print(f"Same speaker: {result.is_same_speaker}")
```

---

## NISQA (Speech Quality Assessment)

Non-Intrusive Speech Quality Assessment predicting MOS scores without reference audio.

**Installation:**
```bash
pip install torchmetrics librosa requests
# Or with talky-talky:
pip install talky-talky[nisqa]
```

**Features:**
- Predicts MOS score (1-5 scale) without reference audio
- Quality dimensions: noisiness, discontinuity, coloration, loudness
- Deep CNN with self-attention architecture
- ~80MB model weights (downloaded on first use)

**Score Interpretation (MOS Scale):**
- 5.0: Excellent quality
- 4.0: Good quality
- 3.0: Fair quality
- 2.0: Poor quality
- 1.0: Bad quality

**Quality Dimensions:**
- **Noisiness**: Background noise level (higher = less noisy)
- **Discontinuity**: Audio dropouts (higher = more continuous)
- **Coloration**: Spectral distortion (higher = more natural)
- **Loudness**: Volume appropriateness (higher = better)

**Example Usage:**
```python
from talky_talky.tools.analysis import assess_quality

result = assess_quality("generated.wav")
if result.status == "success":
    print(f"Overall MOS: {result.overall_quality:.2f}/5.0")
    for dim in result.dimensions:
        print(f"  {dim.name}: {dim.score:.2f}")
```
