"""NISQA Speech Quality Assessment Engine.

Uses NISQA (Non-Intrusive Speech Quality Assessment) to predict speech quality
including overall MOS and quality dimensions: Noisiness, Discontinuity,
Coloration, and Loudness.

For TTS naturalness assessment, uses the NISQA-TTS model variant.

Reference: https://github.com/gabrielmittag/NISQA
"""

import sys
import time
from pathlib import Path

from .base import (
    QualityDimension,
    SpeechQualityEngine,
    SpeechQualityEngineInfo,
    SpeechQualityResult,
)

# Quality dimension names in order from TorchMetrics output
DIMENSION_NAMES = ["overall", "noisiness", "discontinuity", "coloration", "loudness"]

DIMENSION_DESCRIPTIONS = {
    "overall": "Overall Mean Opinion Score (MOS) prediction",
    "noisiness": "Perceived level of background noise",
    "discontinuity": "Presence of audio dropouts or glitches",
    "coloration": "Spectral distortion or tonal changes",
    "loudness": "Appropriateness of audio loudness level",
}

# Target sample rate for NISQA
SAMPLE_RATE = 16000


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


class NISQAEngine(SpeechQualityEngine):
    """Speech quality assessment using NISQA via TorchMetrics."""

    _metric = None

    @property
    def name(self) -> str:
        return "NISQA"

    @property
    def engine_id(self) -> str:
        return "nisqa"

    def is_available(self) -> bool:
        """Check if torchmetrics and required dependencies are installed."""
        try:
            from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> SpeechQualityEngineInfo:
        return SpeechQualityEngineInfo(
            name=self.name,
            engine_id=self.engine_id,
            description=(
                "Non-Intrusive Speech Quality Assessment using deep CNN with self-attention. "
                "Predicts overall MOS score plus quality dimensions (Noisiness, Discontinuity, "
                "Coloration, Loudness). Works without reference audio."
            ),
            requirements="torchmetrics librosa requests (pip install torchmetrics librosa requests)",
            quality_dimensions=DIMENSION_NAMES.copy(),
            score_range=(1.0, 5.0),  # MOS scale
            extra_info={
                "model": "NISQA v2.0",
                "sample_rate": SAMPLE_RATE,
                "paper": "NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction",
            },
        )

    def get_setup_instructions(self) -> str:
        return """## NISQA Setup Instructions

### Installation
```bash
pip install torchmetrics librosa requests
```

### Requirements
- Python 3.8+
- PyTorch
- librosa for audio loading
- requests for model download

### First Run
The NISQA model weights (~80MB) will be downloaded automatically on first use.

### Score Interpretation (MOS Scale 1-5)
- 5.0: Excellent quality
- 4.0: Good quality
- 3.0: Fair quality
- 2.0: Poor quality
- 1.0: Bad quality

### Quality Dimensions
- **Noisiness**: Background noise level (higher = less noisy)
- **Discontinuity**: Audio dropouts (higher = more continuous)
- **Coloration**: Spectral distortion (higher = more natural)
- **Loudness**: Volume appropriateness (higher = better)
"""

    def _get_metric(self):
        """Get or create the NISQA metric instance."""
        if self._metric is None:
            with _redirect_stdout_to_stderr():
                from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

                print("Loading NISQA model...", file=sys.stderr, flush=True)
                self._metric = NonIntrusiveSpeechQualityAssessment(SAMPLE_RATE)
                print("NISQA model loaded successfully", file=sys.stderr, flush=True)
        return self._metric

    def assess_quality(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> SpeechQualityResult:
        """Assess speech quality of audio file.

        Args:
            audio_path: Path to audio file to assess.

        Returns:
            SpeechQualityResult with MOS scores for overall quality and dimensions.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return SpeechQualityResult(
                status="error",
                error=f"Audio file not found: {audio_path}",
            )

        try:
            start_time = time.time()

            with _redirect_stdout_to_stderr():
                import librosa
                import torch

                # Load and resample audio to 16kHz
                audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
                audio_tensor = torch.from_numpy(audio).float()

                # Get NISQA metric
                metric = self._get_metric()

                # Run prediction
                scores = metric(audio_tensor)

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Parse scores - output is tensor of shape (5,)
            # [overall, noisiness, discontinuity, coloration, loudness]
            scores_list = scores.tolist()

            dimensions = []
            for i, (name, score) in enumerate(zip(DIMENSION_NAMES, scores_list)):
                dimensions.append(
                    QualityDimension(
                        name=name,
                        score=float(score),
                        description=DIMENSION_DESCRIPTIONS.get(name),
                    )
                )

            overall_score = scores_list[0] if scores_list else None

            return SpeechQualityResult(
                status="success",
                overall_quality=overall_score,
                dimensions=dimensions,
                processing_time_ms=processing_time_ms,
                metadata={
                    "audio_file": str(audio_path),
                    "sample_rate": SAMPLE_RATE,
                    "audio_duration_seconds": len(audio) / SAMPLE_RATE,
                },
            )

        except Exception as e:
            return SpeechQualityResult(
                status="error",
                error=str(e),
            )

    def assess_quality_batch(
        self,
        audio_paths: list[str | Path],
        **kwargs,
    ) -> list[SpeechQualityResult]:
        """Assess quality of multiple audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of SpeechQualityResult for each file.
        """
        return [self.assess_quality(path, **kwargs) for path in audio_paths]
