"""LeVo/SongGeneration Engine - AI Song Generation.

LeVo (Tencent) generates complete songs from structured lyrics and style descriptions.
Supports vocals, accompaniment, and mixed outputs up to 4.5 minutes.

Based on: "LeVo: High-Quality Song Generation with Multi-Preference Alignment"
Paper: https://arxiv.org/abs/2506.07520
"""

import os
import sys
import tempfile  # noqa: F401
from pathlib import Path
from typing import Optional

from .base import SongGenEngine, SongGenResult, EngineInfo, LyricsFormat
from ..tts.utils import redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

# Model configurations
MODELS = {
    "base-new": {
        "repo_id": "lglg666/SongGeneration-base-new",
        "max_duration_secs": 150,  # 2m30s
        "gpu_memory_gb": 10,
        "gpu_memory_low_mem_gb": 16,
        "languages": ["zh", "en"],
        "description": "Base model with Chinese and English support (2m30s max)",
    },
    "base-full": {
        "repo_id": "lglg666/SongGeneration-base-full",
        "max_duration_secs": 270,  # 4m30s
        "gpu_memory_gb": 12,
        "gpu_memory_low_mem_gb": 18,
        "languages": ["zh", "en"],
        "description": "Full-length model with Chinese and English support (4m30s max)",
    },
    "large": {
        "repo_id": "lglg666/SongGeneration-large",
        "max_duration_secs": 270,  # 4m30s
        "gpu_memory_gb": 22,
        "gpu_memory_low_mem_gb": 28,
        "languages": ["zh", "en"],
        "description": "Large model with best quality (4m30s max, 22GB VRAM)",
    },
}

RUNTIME_REPO_ID = "lglg666/SongGeneration-Runtime"
DEFAULT_MODEL = "base-new"
SAMPLE_RATE = 44100  # LeVo outputs 44.1kHz

# Lyrics structure markers
STRUCTURE_MARKERS = [
    "[intro]",
    "[intro-short]",
    "[verse]",
    "[pre-chorus]",
    "[chorus]",
    "[bridge]",
    "[outro]",
    "[outro-short]",
    "[interlude]",
]

# Style prompts for auto-reference
STYLE_PROMPTS = ["Pop", "Rock", "Jazz", "R&B", "Electronic", "Folk", "Classical", "Hip-Hop", "Auto"]


# ============================================================================
# Model Management
# ============================================================================

# Global model instances
_model = None
_model_name = None
_model_config = None


def _get_cache_dir() -> Path:
    """Get the cache directory for models."""
    cache_dir = os.environ.get("SONGGEN_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    # Default to HuggingFace cache
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(hf_home) / "songgeneration"


def _check_runtime_downloaded() -> dict:
    """Check if the runtime dependencies are downloaded."""
    cache_dir = _get_cache_dir()
    ckpt_dir = cache_dir / "ckpt"
    third_party_dir = cache_dir / "third_party"

    return {
        "downloaded": ckpt_dir.exists() and third_party_dir.exists(),
        "ckpt_path": str(ckpt_dir) if ckpt_dir.exists() else None,
        "third_party_path": str(third_party_dir) if third_party_dir.exists() else None,
        "cache_dir": str(cache_dir),
    }


def _check_model_downloaded(model_name: str) -> dict:
    """Check if a specific model is downloaded."""
    if model_name not in MODELS:
        return {"error": f"Unknown model: {model_name}"}

    cache_dir = _get_cache_dir()
    model_dir = cache_dir / f"songgeneration_{model_name.replace('-', '_')}"
    config_path = model_dir / "config.yaml"
    model_path = model_dir / "model.pt"

    return {
        "downloaded": config_path.exists() and model_path.exists(),
        "model_dir": str(model_dir) if model_dir.exists() else None,
        "model_name": model_name,
        "repo_id": MODELS[model_name]["repo_id"],
    }


def check_models_downloaded(model_name: str = DEFAULT_MODEL) -> dict:
    """Check download status for runtime and model."""
    runtime_status = _check_runtime_downloaded()
    model_status = _check_model_downloaded(model_name)

    return {
        "runtime": runtime_status,
        "model": model_status,
        "all_downloaded": runtime_status["downloaded"] and model_status["downloaded"],
        "model_name": model_name,
    }


def download_models(model_name: str = DEFAULT_MODEL, force: bool = False) -> dict:
    """Download SongGeneration models from HuggingFace.

    Args:
        model_name: Model variant to download ("base-new", "base-full", "large").
        force: Re-download even if already cached.

    Returns:
        Dict with download status.
    """
    if model_name not in MODELS:
        return {
            "status": "error",
            "error": f"Unknown model: {model_name}. Available: {list(MODELS.keys())}",
        }

    if not force:
        status = check_models_downloaded(model_name)
        if status["all_downloaded"]:
            return {
                "status": "already_downloaded",
                "message": f"SongGeneration {model_name} is already downloaded",
                "models": status,
            }

    results = {"runtime": {"status": "pending"}, "model": {"status": "pending"}}

    try:
        from huggingface_hub import snapshot_download

        cache_dir = _get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download runtime
        print("Downloading SongGeneration runtime...", file=sys.stderr, flush=True)
        runtime_path = snapshot_download(
            RUNTIME_REPO_ID,
            local_dir=cache_dir / "runtime_temp",
            local_dir_use_symlinks=False,
        )

        # Move runtime components to expected locations
        runtime_temp = Path(runtime_path)
        if (runtime_temp / "ckpt").exists():
            import shutil

            shutil.move(str(runtime_temp / "ckpt"), str(cache_dir / "ckpt"))
        if (runtime_temp / "third_party").exists():
            import shutil

            shutil.move(str(runtime_temp / "third_party"), str(cache_dir / "third_party"))

        results["runtime"]["status"] = "downloaded"
        results["runtime"]["path"] = str(cache_dir)

    except Exception as e:
        results["runtime"]["status"] = "error"
        results["runtime"]["error"] = str(e)
        # Don't continue if runtime failed
        return {
            "status": "error",
            "message": "Failed to download runtime",
            "models": results,
        }

    try:
        from huggingface_hub import snapshot_download

        # Download model
        print(f"Downloading SongGeneration {model_name} model...", file=sys.stderr, flush=True)
        model_dir_name = f"songgeneration_{model_name.replace('-', '_')}"
        model_path = snapshot_download(
            MODELS[model_name]["repo_id"],
            local_dir=cache_dir / model_dir_name,
            local_dir_use_symlinks=False,
        )
        results["model"]["status"] = "downloaded"
        results["model"]["path"] = str(model_path)

    except Exception as e:
        results["model"]["status"] = "error"
        results["model"]["error"] = str(e)

    all_success = all(r["status"] in ("downloaded", "cached") for r in results.values())
    return {
        "status": "success" if all_success else "error",
        "message": f"SongGeneration {model_name} downloaded"
        if all_success
        else "Some downloads failed",
        "models": results,
    }


def _load_model(
    model_name: str = DEFAULT_MODEL, low_mem: bool = False, use_flash_attn: bool = True
):
    """Load the SongGeneration model."""
    global _model, _model_name, _model_config

    if _model is not None and _model_name == model_name:
        return _model, _model_config

    # Check if models are downloaded
    status = check_models_downloaded(model_name)
    if not status["all_downloaded"]:
        raise RuntimeError(
            f"Models not downloaded. Run download_models('{model_name}') first.\n"
            f"Runtime: {status['runtime']['downloaded']}, Model: {status['model']['downloaded']}"
        )

    cache_dir = _get_cache_dir()
    model_dir = cache_dir / f"songgeneration_{model_name.replace('-', '_')}"

    # Add paths to Python path for imports
    third_party_path = cache_dir / "third_party"
    if str(third_party_path) not in sys.path:
        sys.path.insert(0, str(third_party_path))

    print(f"Loading SongGeneration {model_name} model...", file=sys.stderr, flush=True)

    with redirect_stdout_to_stderr():
        import torch
        import yaml

        # Load config
        config_path = model_dir / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        _model_config = config

        # Import model builders (these come from the third_party directory)
        try:
            from levo import builders
            from levo.inference import CodecLM
        except ImportError as e:
            raise ImportError(
                f"Failed to import LeVo modules. Ensure runtime is properly installed: {e}"
            ) from e

        # Load audio tokenizer
        audio_tokenizer = builders.get_audio_tokenizer_model(
            config["audio_tokenizer_config"],
            ckpt_path=str(cache_dir / config["audio_tokenizer_ckpt"]),
        )

        # Load language model
        lm = builders.get_lm_model(config)

        # Load checkpoint
        ckpt_path = model_dir / "model.pt"
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Filter and load state dict
        lm_state = {
            k.replace("audiolm.", ""): v for k, v in state_dict.items() if k.startswith("audiolm.")
        }
        lm.load_state_dict(lm_state, strict=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("SongGeneration requires CUDA GPU. No CUDA device found.")

        lm = lm.to(device)
        audio_tokenizer = audio_tokenizer.to(device)

        # Wrap in CodecLM for generation
        model = CodecLM(
            lm=lm,
            audio_tokenizer=audio_tokenizer,
            use_flash_attn=use_flash_attn,
        )

        # Load separation tokenizer if available
        if "sep_tokenizer_config" in config:
            sep_tokenizer = builders.get_audio_tokenizer_model(
                config["sep_tokenizer_config"],
                ckpt_path=str(cache_dir / config.get("sep_tokenizer_ckpt", "")),
            )
            sep_tokenizer = sep_tokenizer.to(device)
            model.sep_tokenizer = sep_tokenizer

    _model = model
    _model_name = model_name

    print(f"SongGeneration {model_name} loaded on CUDA", file=sys.stderr, flush=True)
    return _model, _model_config


# ============================================================================
# Generation
# ============================================================================


def _generate_song(
    lyrics: str,
    output_path: Path,
    description: Optional[str] = None,
    generate_type: str = "mixed",
    prompt_audio_path: Optional[str] = None,
    auto_prompt_style: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    low_mem: bool = False,
    use_flash_attn: bool = True,
) -> SongGenResult:
    """Generate a song from lyrics.

    Args:
        lyrics: Structured lyrics with section markers.
        output_path: Where to save the output.
        description: Musical style description (e.g., "female, pop, sad, piano").
        generate_type: Output type - "mixed", "separate", "vocal", or "bgm".
        prompt_audio_path: Optional path to 10s reference audio for style.
        auto_prompt_style: Auto-select reference style (Pop, Jazz, Rock, etc.).
        model_name: Model variant to use.
        low_mem: Enable low-memory mode.
        use_flash_attn: Use flash attention (faster but requires compatible GPU).

    Returns:
        SongGenResult with status and paths.
    """
    import soundfile as sf

    output_path = Path(output_path)

    # Validate inputs
    if not lyrics or not lyrics.strip():
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type=generate_type,
            error="Lyrics cannot be empty",
        )

    if generate_type not in ["mixed", "separate", "vocal", "bgm"]:
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type=generate_type,
            error=f"Invalid generate_type: {generate_type}. Must be mixed, separate, vocal, or bgm.",
        )

    try:
        model, config = _load_model(model_name, low_mem, use_flash_attn)
    except Exception as e:
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type=generate_type,
            error=f"Failed to load model: {e}",
        )

    try:
        with redirect_stdout_to_stderr():
            import torch

            # Prepare generation inputs
            gen_kwargs = {
                "lyrics": lyrics,
                "descriptions": description or "",
                "return_tokens": True,
            }

            if prompt_audio_path:
                gen_kwargs["prompt_audio_path"] = prompt_audio_path
            elif auto_prompt_style:
                gen_kwargs["auto_prompt_audio_type"] = auto_prompt_style

            # Generate tokens
            print(f"Generating song ({generate_type} mode)...", file=sys.stderr, flush=True)

            with torch.no_grad():
                tokens = model.generate(**gen_kwargs)

                # Generate audio based on type
                if generate_type == "mixed":
                    audio = model.generate_audio(tokens, mode="mixed")
                    audio_np = audio.cpu().numpy()
                elif generate_type == "vocal":
                    audio = model.generate_audio(tokens, mode="vocal")
                    audio_np = audio.cpu().numpy()
                elif generate_type == "bgm":
                    audio = model.generate_audio(tokens, mode="bgm")
                    audio_np = audio.cpu().numpy()
                elif generate_type == "separate":
                    vocal_audio = model.generate_audio(tokens, mode="vocal")
                    bgm_audio = model.generate_audio(tokens, mode="bgm")
                    mixed_audio = model.generate_audio(tokens, mode="mixed")
                    audio_np = mixed_audio.cpu().numpy()

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle stereo output (may be [2, samples] or [samples, 2])
            if len(audio_np.shape) > 1:
                if audio_np.shape[0] == 2:
                    audio_np = audio_np.T  # Convert to [samples, 2]
                elif audio_np.shape[1] != 2:
                    audio_np = audio_np.squeeze()  # Mono

            sf.write(str(output_path), audio_np, SAMPLE_RATE)

            duration_ms = int(len(audio_np) / SAMPLE_RATE * 1000)

            result = SongGenResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=SAMPLE_RATE,
                generate_type=generate_type,
                metadata={
                    "model": model_name,
                    "description": description,
                    "lyrics_length": len(lyrics),
                },
            )

            # Handle separate mode outputs
            if generate_type == "separate":
                vocal_path = output_path.with_stem(output_path.stem + "_vocal")
                bgm_path = output_path.with_stem(output_path.stem + "_bgm")

                vocal_np = vocal_audio.cpu().numpy()
                bgm_np = bgm_audio.cpu().numpy()

                if len(vocal_np.shape) > 1 and vocal_np.shape[0] == 2:
                    vocal_np = vocal_np.T
                if len(bgm_np.shape) > 1 and bgm_np.shape[0] == 2:
                    bgm_np = bgm_np.T

                sf.write(str(vocal_path), vocal_np, SAMPLE_RATE)
                sf.write(str(bgm_path), bgm_np, SAMPLE_RATE)

                result.vocal_path = str(vocal_path)
                result.bgm_path = str(bgm_path)

            return result

    except Exception as e:
        return SongGenResult(
            status="error",
            output_path=str(output_path),
            duration_ms=0,
            sample_rate=SAMPLE_RATE,
            generate_type=generate_type,
            error=f"Generation failed: {e}",
        )


# ============================================================================
# Engine Implementation
# ============================================================================


class LeVoEngine(SongGenEngine):
    """LeVo/SongGeneration Engine - AI Song Generation.

    Generates complete songs from structured lyrics and style descriptions.
    Supports vocals, accompaniment (BGM), and mixed outputs.

    Based on Tencent's LeVo paper: "High-Quality Song Generation with Multi-Preference Alignment"
    """

    @property
    def name(self) -> str:
        return "LeVo (SongGeneration)"

    @property
    def engine_id(self) -> str:
        return "levo"

    def is_available(self) -> bool:
        """Check if LeVo is available (CUDA + dependencies)."""
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            # Check core dependencies
            import yaml  # noqa: F401
            import soundfile  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            description=(
                "AI song generation from lyrics and style descriptions. "
                "Creates complete songs with vocals and accompaniment. "
                "Supports Chinese and English lyrics up to 4.5 minutes."
            ),
            requirements="CUDA GPU (10-28GB VRAM), torch, transformers (~10GB download)",
            max_duration_secs=MODELS[DEFAULT_MODEL]["max_duration_secs"],
            sample_rate=SAMPLE_RATE,
            supported_languages=["zh", "en"],
            gpu_memory_required_gb=MODELS[DEFAULT_MODEL]["gpu_memory_gb"],
            lyrics_format=LyricsFormat(
                overview=(
                    "Lyrics use structure markers to define song sections. "
                    "Separate sections with ';' and sentences within sections with '.'"
                ),
                structure_markers=STRUCTURE_MARKERS,
                separator=";",
                sentence_separator=".",
                examples=[
                    {
                        "name": "Simple pop song",
                        "lyrics": "[intro-short] ; [verse] Hello world. I'm singing today ; [chorus] This is the chorus. Sing along with me ; [outro-short]",
                        "description": "female, pop, happy, upbeat",
                    },
                    {
                        "name": "Ballad",
                        "lyrics": "[verse] Memories fade away. Like leaves in autumn wind ; [chorus] But I still remember you. Your voice echoes in my mind ; [bridge] Time moves on. Hearts grow cold ; [chorus] But I still remember you ; [outro]",
                        "description": "male, ballad, sad, piano, slow",
                    },
                ],
            ),
            supports_style_reference=True,
            supports_description=True,
            generate_types=["mixed", "separate", "vocal", "bgm"],
            extra_info={
                "paper": "https://arxiv.org/abs/2506.07520",
                "demo": "https://levo-demo.github.io/",
                "models": MODELS,
                "style_prompts": STYLE_PROMPTS,
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## LeVo (SongGeneration) Setup

LeVo requires a CUDA GPU with 10-28GB VRAM depending on model size.

### 1. Install dependencies:
```bash
pip install "talky-talky[songgen]"
```

### 2. Download models (first-time setup):
Use the `download_songgen_models` tool or:
```python
from talky_talky.tools.songgen.levo import download_models
download_models("base-new")  # 10GB VRAM required
```

### Model Options:
- `base-new`: 2m30s max, 10GB VRAM, Chinese/English (default)
- `base-full`: 4m30s max, 12GB VRAM, Chinese/English
- `large`: 4m30s max, 22GB VRAM, best quality

### Hardware Requirements:
- NVIDIA GPU with CUDA support
- 10-28GB VRAM depending on model
- ~10GB disk space for model weights
"""

    def generate(
        self,
        lyrics: str,
        output_path: Path,
        description: Optional[str] = None,
        generate_type: str = "mixed",
        prompt_audio_path: Optional[str] = None,
        auto_prompt_style: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        low_mem: bool = False,
        use_flash_attn: bool = True,
        **kwargs,
    ) -> SongGenResult:
        """Generate a song from lyrics.

        Args:
            lyrics: Structured lyrics with section markers like [verse], [chorus].
                   Separate sections with ';' and sentences with '.'.
            output_path: Where to save the generated audio.
            description: Musical style description (e.g., "female, pop, sad, piano").
            generate_type: Output type:
                - "mixed": Combined vocals and accompaniment (default)
                - "separate": Outputs 3 files (mixed, vocals, bgm)
                - "vocal": Vocals only (a cappella)
                - "bgm": Accompaniment only (instrumental)
            prompt_audio_path: Optional path to ~10s reference audio for style.
            auto_prompt_style: Auto-select reference style: Pop, Rock, Jazz, etc.
            model_name: Model variant: "base-new", "base-full", "large".
            low_mem: Enable low-memory mode (slower, uses less VRAM).
            use_flash_attn: Use flash attention (faster, requires compatible GPU).

        Returns:
            SongGenResult with status, paths, and metadata.
        """
        return _generate_song(
            lyrics=lyrics,
            output_path=output_path,
            description=description,
            generate_type=generate_type,
            prompt_audio_path=prompt_audio_path,
            auto_prompt_style=auto_prompt_style,
            model_name=model_name,
            low_mem=low_mem,
            use_flash_attn=use_flash_attn,
        )
