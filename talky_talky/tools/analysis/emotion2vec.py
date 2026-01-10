"""Emotion2vec Speech Emotion Recognition Engine.

Uses FunASR's emotion2vec_plus_large model for speech emotion detection.
Supports 9 emotion categories: angry, disgusted, fearful, happy, neutral,
other, sad, surprised, unknown.

Reference: https://github.com/ddlBoJack/emotion2vec
"""

import sys
import time
from pathlib import Path

from .base import EmotionEngine, EmotionEngineInfo, EmotionResult, EmotionScore

# Emotion labels in order (indices 0-8)
EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]

# Lazy-loaded model singleton
_model = None


def _redirect_stdout_to_stderr():
    """Context manager to redirect stdout to stderr during model loading."""
    import contextlib

    @contextlib.contextmanager
    def redirect():
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            yield
        finally:
            sys.stdout = old_stdout

    return redirect()


def _load_model():
    """Lazily load the emotion2vec model."""
    global _model
    if _model is not None:
        return _model

    with _redirect_stdout_to_stderr():
        from funasr import AutoModel

        print("Loading emotion2vec_plus_large model...", file=sys.stderr, flush=True)

        _model = AutoModel(
            model="iic/emotion2vec_plus_large",
            hub="hf",  # Use HuggingFace for overseas users
        )

        print("emotion2vec model loaded successfully", file=sys.stderr, flush=True)

    return _model


class Emotion2vecEngine(EmotionEngine):
    """Emotion detection using emotion2vec from FunASR."""

    @property
    def name(self) -> str:
        return "Emotion2vec"

    @property
    def engine_id(self) -> str:
        return "emotion2vec"

    def is_available(self) -> bool:
        """Check if funasr is installed."""
        try:
            import funasr  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EmotionEngineInfo:
        return EmotionEngineInfo(
            name=self.name,
            engine_id=self.engine_id,
            description=(
                "State-of-the-art speech emotion recognition using emotion2vec+ large model "
                "(~300M parameters). Trained on 40k hours of speech emotion data. "
                "Supports 9 emotion categories with high accuracy."
            ),
            requirements="funasr (pip install funasr modelscope)",
            supported_emotions=EMOTION_LABELS.copy(),
            extra_info={
                "model": "iic/emotion2vec_plus_large",
                "parameters": "~300M",
                "sample_rate": 16000,
                "license": "MIT",
                "paper": "ACL 2024",
            },
        )

    def get_setup_instructions(self) -> str:
        return """## Emotion2vec Setup Instructions

### Installation
```bash
pip install funasr modelscope
```

### Requirements
- Python 3.8+
- PyTorch 1.13+
- ~2GB disk space for model weights
- GPU recommended but not required

### First Run
The model will be downloaded automatically on first use (~1GB download).

### Supported Emotions
angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
"""

    def detect_emotion(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> EmotionResult:
        """Detect emotion in audio file.

        Args:
            audio_path: Path to audio file (WAV format, 16kHz recommended).

        Returns:
            EmotionResult with detected emotions and confidence scores.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return EmotionResult(
                status="error",
                error=f"Audio file not found: {audio_path}",
            )

        try:
            start_time = time.time()
            model = _load_model()

            # Run emotion detection
            with _redirect_stdout_to_stderr():
                result = model.generate(
                    str(audio_path),
                    granularity="utterance",
                    extract_embedding=False,
                )

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Parse results - result is a list with one dict per input
            if not result or len(result) == 0:
                return EmotionResult(
                    status="error",
                    error="No results returned from model",
                    processing_time_ms=processing_time_ms,
                )

            # Get scores from result
            # The result format is: [{'labels': [...], 'scores': [...], 'feats': [...]}]
            item = result[0]

            if "scores" in item and "labels" in item:
                scores = item["scores"]
                labels = item["labels"]

                # Build emotion scores list
                emotion_scores = []
                for label, score in zip(labels, scores):
                    # Labels are like "/m/angry" - extract just the emotion name
                    emotion_name = label.split("/")[-1] if "/" in label else label
                    emotion_scores.append(EmotionScore(emotion=emotion_name, score=float(score)))

                # Sort by score descending
                emotion_scores.sort(key=lambda x: x.score, reverse=True)

                primary = emotion_scores[0] if emotion_scores else None

                return EmotionResult(
                    status="success",
                    primary_emotion=primary.emotion if primary else None,
                    primary_score=primary.score if primary else None,
                    all_emotions=emotion_scores,
                    processing_time_ms=processing_time_ms,
                    metadata={
                        "model": "emotion2vec_plus_large",
                        "audio_file": str(audio_path),
                    },
                )
            else:
                # Fallback for different result format
                return EmotionResult(
                    status="error",
                    error=f"Unexpected result format: {list(item.keys())}",
                    processing_time_ms=processing_time_ms,
                    metadata={"raw_result": str(item)[:500]},
                )

        except Exception as e:
            return EmotionResult(
                status="error",
                error=str(e),
            )
