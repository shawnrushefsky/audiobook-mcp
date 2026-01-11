"""Song Generation module for talky-talky.

This module provides AI-powered song generation from lyrics and descriptions.

Engines:
- LeVo (Tencent/SongGeneration): Text-to-song generation from structured lyrics (CUDA only)
- ACE-Step: Song generation foundation model (MPS + CUDA support)

Usage:
    from talky_talky.tools.songgen import generate, get_info, is_available

    result = generate(
        lyrics="[verse] Hello world...",
        output_path="song.wav",
        description="female, pop, happy",
    )
"""

from .base import (
    SongGenResult,
    SongGenEngine,
    EngineInfo,
)

# Engine registry
_engines: dict[str, SongGenEngine] = {}


def register_engine(engine: SongGenEngine) -> None:
    """Register a song generation engine."""
    _engines[engine.engine_id] = engine


def get_engine(engine_id: str) -> SongGenEngine:
    """Get a registered engine by ID."""
    if engine_id not in _engines:
        raise ValueError(f"Unknown engine: {engine_id}. Available: {list(_engines.keys())}")
    return _engines[engine_id]


def list_engines() -> list[str]:
    """List all registered engine IDs."""
    return list(_engines.keys())


def get_available_engines() -> dict[str, dict]:
    """Get information about available (installed) engines."""
    available = {}
    for engine_id, engine in _engines.items():
        if engine.is_available():
            available[engine_id] = {
                "name": engine.name,
                "available": True,
            }
    return available


def check_songgen() -> dict:
    """Check song generation engine availability and device info."""
    from ..tts.utils import get_best_device

    device, device_name, vram_gb = get_best_device()

    results = {
        "device": device,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "engines": {},
    }

    for engine_id, engine in _engines.items():
        is_avail = engine.is_available()
        results["engines"][engine_id] = {
            "name": engine.name,
            "available": is_avail,
            "setup_instructions": engine.get_setup_instructions() if not is_avail else None,
        }

    return results


def get_info(engine_id: str) -> EngineInfo:
    """Get detailed information about an engine."""
    engine = get_engine(engine_id)
    return engine.get_info()


def generate(
    engine_id: str,
    lyrics: str,
    output_path: str,
    **kwargs,
) -> SongGenResult:
    """Generate a song using the specified engine."""
    from pathlib import Path

    engine = get_engine(engine_id)
    return engine.generate(
        lyrics=lyrics,
        output_path=Path(output_path),
        **kwargs,
    )


# Register engines
try:
    from .levo import LeVoEngine

    register_engine(LeVoEngine())
except ImportError:
    pass  # Engine dependencies not installed

try:
    from .acestep import ACEStepEngine

    register_engine(ACEStepEngine())
except ImportError:
    pass  # Engine dependencies not installed
