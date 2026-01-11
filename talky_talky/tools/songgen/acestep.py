"""ACE-Step Engine - AI Song Generation Foundation Model.

ACE-Step generates complete songs with vocals from text prompts and lyrics.
Supports Apple Silicon (MPS) with 36GB+ unified memory, as well as CUDA GPUs.

Based on: "ACE-Step: A Step Towards Music Generation Foundation Model"
GitHub: https://github.com/ace-step/ACE-Step
"""

import sys
from pathlib import Path
from typing import Optional

from .base import SongGenEngine, SongGenResult, EngineInfo, LyricsFormat
from ..tts.utils import redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

# Default checkpoint cache location
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ace-step" / "checkpoints"

# Sample rate for ACE-Step output
SAMPLE_RATE = 44100

# Maximum duration in seconds
MAX_DURATION_SECS = 240  # 4 minutes

# Lyrics structure markers
STRUCTURE_MARKERS = [
    "[intro]",
    "[verse]",
    "[pre-chorus]",
    "[chorus]",
    "[bridge]",
    "[outro]",
    "[instrumental]",
    "[hook]",
    "[break]",
]

# Default generation parameters
DEFAULT_INFER_STEPS = 27  # Fast mode (60 for higher quality)
DEFAULT_GUIDANCE_SCALE = 15.0
DEFAULT_SCHEDULER = "euler"


# ============================================================================
# Model Management
# ============================================================================

# Global pipeline instance
_pipeline = None
_pipeline_config = None


def _get_device_and_dtype():
    """Get the best device and dtype for the current platform."""
    try:
        import torch
    except ImportError:
        return "cpu", "float32"

    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS requires float32 for ACE-Step
        return "mps", "float32"
    else:
        return "cpu", "float32"


def check_models_downloaded() -> dict:
    """Check if ACE-Step models are downloaded."""
    checkpoint_dir = DEFAULT_CACHE_DIR

    # Check for key model files
    model_file = checkpoint_dir / "model.safetensors"
    checkpoint_dir / "config.json"

    if model_file.exists() or (
        checkpoint_dir.exists() and any(checkpoint_dir.glob("*.safetensors"))
    ):
        return {
            "downloaded": True,
            "checkpoint_dir": str(checkpoint_dir),
        }

    return {
        "downloaded": False,
        "checkpoint_dir": str(checkpoint_dir),
        "note": "Models will be auto-downloaded on first use from HuggingFace",
    }


def download_models(force: bool = False) -> dict:
    """Download ACE-Step models from HuggingFace.

    Note: ACE-Step auto-downloads models on first use, but this function
    can be used to pre-download them.

    Args:
        force: Re-download even if already cached.

    Returns:
        Dict with download status.
    """
    if not force:
        status = check_models_downloaded()
        if status["downloaded"]:
            return {
                "status": "already_downloaded",
                "message": "ACE-Step models are already downloaded",
                "checkpoint_dir": status["checkpoint_dir"],
            }

    try:
        from huggingface_hub import snapshot_download

        print("Downloading ACE-Step models...", file=sys.stderr, flush=True)

        # Download from HuggingFace
        checkpoint_path = snapshot_download(
            "ACE-Step/ACE-Step-v1-3.5B",
            local_dir=str(DEFAULT_CACHE_DIR),
            local_dir_use_symlinks=False,
        )

        return {
            "status": "success",
            "message": "ACE-Step models downloaded successfully",
            "checkpoint_dir": checkpoint_path,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "note": "Models will auto-download on first generation if this fails",
        }


def _load_pipeline(
    checkpoint_dir: Optional[str] = None,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    quantized: bool = False,
):
    """Load the ACE-Step pipeline."""
    global _pipeline, _pipeline_config

    device, dtype = _get_device_and_dtype()

    config = {
        "checkpoint_dir": checkpoint_dir,
        "device": device,
        "dtype": dtype,
        "torch_compile": torch_compile,
        "cpu_offload": cpu_offload,
        "quantized": quantized,
    }

    # Return cached pipeline if config matches
    if _pipeline is not None and _pipeline_config == config:
        return _pipeline

    print(f"Loading ACE-Step pipeline on {device} ({dtype})...", file=sys.stderr, flush=True)

    with redirect_stdout_to_stderr():
        from acestep.pipeline_ace_step import ACEStepPipeline

        _pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir,
            dtype=dtype,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            quantized=quantized,
        )

        # Load checkpoint
        if quantized:
            _pipeline.load_quantized_checkpoint()
        else:
            _pipeline.load_checkpoint()

    _pipeline_config = config
    print(f"ACE-Step loaded successfully on {device}", file=sys.stderr, flush=True)

    return _pipeline


# ============================================================================
# Generation
# ============================================================================


def _generate_song(
    prompt: str,
    output_path: Path,
    lyrics: Optional[str] = None,
    audio_duration: float = 60.0,
    infer_steps: int = DEFAULT_INFER_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    scheduler_type: str = DEFAULT_SCHEDULER,
    seed: Optional[int] = None,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    quantized: bool = False,
) -> SongGenResult:
    """Generate a song with ACE-Step.

    Args:
        prompt: Text description of desired music style/genre.
        output_path: Where to save the generated audio.
        lyrics: Optional lyrics with structure markers.
        audio_duration: Duration in seconds (max 240).
        infer_steps: Number of inference steps (27=fast, 60=quality).
        guidance_scale: Classifier-free guidance strength.
        scheduler_type: Scheduler type (euler, heun, pingpong).
        seed: Random seed for reproducibility.
        torch_compile: Enable torch.compile optimization.
        cpu_offload: Offload weights to CPU (saves VRAM).
        quantized: Use quantized model (lower quality, less memory).

    Returns:
        SongGenResult with status and metadata.
    """
    output_path = Path(output_path)

    # Validate inputs
    if not prompt or not prompt.strip():
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type="mixed",
            error="Prompt cannot be empty",
        )

    if audio_duration > MAX_DURATION_SECS:
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type="mixed",
            error=f"Duration exceeds maximum of {MAX_DURATION_SECS} seconds",
        )

    try:
        pipeline = _load_pipeline(
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            quantized=quantized,
        )
    except Exception as e:
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type="mixed",
            error=f"Failed to load ACE-Step: {e}",
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with redirect_stdout_to_stderr():
            # Prepare generation parameters
            gen_kwargs = {
                "prompt": prompt,
                "audio_duration": audio_duration,
                "infer_step": infer_steps,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "save_path": str(output_path.parent),
                "format": output_path.suffix.lstrip(".") or "wav",
            }

            if lyrics:
                gen_kwargs["lyrics"] = lyrics

            if seed is not None:
                gen_kwargs["manual_seeds"] = [seed]

            print(f"Generating {audio_duration}s song...", file=sys.stderr, flush=True)

            # Generate
            results = pipeline(**gen_kwargs)

            # Results is a list of paths + metadata dict
            if isinstance(results, (list, tuple)) and len(results) > 0:
                generated_path = results[0]
                if isinstance(generated_path, str) and Path(generated_path).exists():
                    # Move/rename to desired output path if needed
                    if str(generated_path) != str(output_path):
                        import shutil

                        shutil.move(generated_path, str(output_path))

        # Calculate duration
        try:
            import soundfile as sf

            info = sf.info(str(output_path))
            duration_ms = int(info.duration * 1000)
        except Exception:
            duration_ms = int(audio_duration * 1000)

        return SongGenResult(
            status="success",
            output_path=str(output_path),
            duration_ms=duration_ms,
            sample_rate=SAMPLE_RATE,
            generate_type="mixed",
            metadata={
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration,
                "infer_steps": infer_steps,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "seed": seed,
            },
        )

    except Exception as e:
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type="mixed",
            error=f"Generation failed: {e}",
        )


# ============================================================================
# Engine Implementation
# ============================================================================


class ACEStepEngine(SongGenEngine):
    """ACE-Step Engine - AI Song Generation Foundation Model.

    Generates complete songs with vocals from text prompts and optional lyrics.
    Supports Apple Silicon (MPS) and CUDA GPUs.

    Features:
    - Text-to-music generation with style prompts
    - Optional lyrics with structure markers
    - Up to 4 minutes of audio generation
    - MPS support for Apple Silicon (36GB+ unified memory)
    """

    @property
    def name(self) -> str:
        return "ACE-Step"

    @property
    def engine_id(self) -> str:
        return "acestep"

    def is_available(self) -> bool:
        """Check if ACE-Step is available."""
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            import transformers  # noqa: F401

            # Check for acestep package
            try:
                from acestep.pipeline_ace_step import ACEStepPipeline  # noqa: F401

                return True
            except ImportError:
                return False

        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        device, dtype = _get_device_and_dtype()

        return EngineInfo(
            name=self.name,
            description=(
                "AI song generation foundation model. Creates complete songs with vocals "
                "from text prompts and optional lyrics. Supports Apple Silicon (MPS) and CUDA. "
                "3.5B parameters, generates up to 4 minutes of music."
            ),
            requirements=f"torch, diffusers, transformers, acestep (~36GB unified memory for MPS, 12GB+ VRAM for CUDA). Current device: {device}",
            max_duration_secs=MAX_DURATION_SECS,
            sample_rate=SAMPLE_RATE,
            supported_languages=["en", "zh", "ja", "ko", "es", "fr", "de"],  # Multilingual
            gpu_memory_required_gb=36.0 if device == "mps" else 12.0,
            lyrics_format=LyricsFormat(
                overview=(
                    "Lyrics use structure markers to define song sections. "
                    "Markers are placed on their own lines before the section text."
                ),
                structure_markers=STRUCTURE_MARKERS,
                separator="\n",
                sentence_separator="\n",
                examples=[
                    {
                        "name": "Pop song with lyrics",
                        "prompt": "female vocals, pop, upbeat, synth, drums",
                        "lyrics": "[verse]\nWalking down the street today\nFeeling like I'm on my way\n\n[chorus]\nThis is my moment\nNothing can stop me now",
                    },
                    {
                        "name": "Instrumental rock",
                        "prompt": "rock, electric guitar, drums, energetic, instrumental",
                        "lyrics": None,  # No lyrics for instrumental
                    },
                ],
            ),
            supports_style_reference=False,  # ACE-Step uses text prompts
            supports_description=True,
            generate_types=["mixed"],  # ACE-Step outputs mixed audio
            extra_info={
                "github": "https://github.com/ace-step/ACE-Step",
                "huggingface": "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B",
                "model_size": "3.5B parameters",
                "mps_support": True,
                "cuda_support": True,
                "current_device": device,
                "current_dtype": dtype,
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## ACE-Step Setup

ACE-Step supports both CUDA GPUs and Apple Silicon (MPS).

### 1. Install dependencies:
```bash
pip install "talky-talky[acestep]"
```

Or install ACE-Step directly:
```bash
pip install git+https://github.com/ace-step/ACE-Step.git
```

### 2. Hardware Requirements:

**Apple Silicon (MPS):**
- macOS 12.3+
- M1/M2/M3 Max or Ultra with 36GB+ unified memory
- Use `--bf16 false` (float32 required)

**CUDA GPU:**
- 12GB+ VRAM (24GB+ recommended)
- RTX 3090/4090, A100, etc.

**Memory-constrained setups:**
- Use `cpu_offload=True` and `quantized=True`
- Enables running on 8GB VRAM/RAM

### 3. First run:
Models auto-download from HuggingFace (~7GB) on first use.
"""

    def generate(
        self,
        lyrics: str,
        output_path: Path,
        description: Optional[str] = None,
        generate_type: str = "mixed",
        audio_duration: float = 60.0,
        infer_steps: int = DEFAULT_INFER_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        scheduler_type: str = DEFAULT_SCHEDULER,
        seed: Optional[int] = None,
        torch_compile: bool = False,
        cpu_offload: bool = False,
        quantized: bool = False,
        **kwargs,
    ) -> SongGenResult:
        """Generate a song with ACE-Step.

        Note: ACE-Step uses 'prompt' for style description and 'lyrics' for song text.
        For compatibility with the base interface, 'lyrics' here is the style prompt
        and 'description' can contain actual lyrics.

        Args:
            lyrics: Style prompt (e.g., "female vocals, pop, upbeat"). Despite the name,
                   this is the style description for ACE-Step.
            output_path: Where to save the generated audio.
            description: Optional actual lyrics with structure markers.
            generate_type: Output type (only "mixed" supported).
            audio_duration: Duration in seconds (max 240).
            infer_steps: Inference steps (27=fast, 60=quality).
            guidance_scale: CFG strength (default 15.0).
            scheduler_type: Scheduler (euler, heun, pingpong).
            seed: Random seed for reproducibility.
            torch_compile: Enable torch.compile optimization.
            cpu_offload: Offload to CPU (saves memory).
            quantized: Use quantized model.

        Returns:
            SongGenResult with status and metadata.
        """
        # ACE-Step terminology:
        # - "prompt" = style description (tags like "pop, female, upbeat")
        # - "lyrics" = actual song lyrics
        #
        # Our interface uses:
        # - "lyrics" = the main input (style prompt for ACE-Step)
        # - "description" = optional actual lyrics

        return _generate_song(
            prompt=lyrics,  # Style prompt
            output_path=output_path,
            lyrics=description,  # Actual lyrics (optional)
            audio_duration=audio_duration,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            seed=seed,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            quantized=quantized,
        )
