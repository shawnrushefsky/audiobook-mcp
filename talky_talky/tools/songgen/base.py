"""Base classes and interfaces for song generation engines.

This module defines the abstract interface that all song generation engines must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SongGenResult:
    """Result from a song generation."""

    status: str  # "success" or "error"
    output_path: str  # Path to main output (mixed or specified type)
    duration_ms: int
    sample_rate: int
    generate_type: str  # "mixed", "separate", "vocal", "bgm"
    error: Optional[str] = None
    # For separate mode, paths to individual tracks
    vocal_path: Optional[str] = None
    bgm_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LyricsFormat:
    """Documentation for lyrics formatting."""

    overview: str
    structure_markers: list[str]  # [verse], [chorus], etc.
    separator: str  # Usually ";"
    sentence_separator: str  # Usually "."
    examples: list[dict] = field(default_factory=list)


@dataclass
class EngineInfo:
    """Information about a song generation engine."""

    name: str
    description: str
    requirements: str
    max_duration_secs: int
    sample_rate: int
    supported_languages: list[str]
    gpu_memory_required_gb: float
    lyrics_format: Optional[LyricsFormat] = None
    supports_style_reference: bool = False
    supports_description: bool = True
    generate_types: list[str] = field(default_factory=lambda: ["mixed", "separate", "vocal", "bgm"])
    extra_info: dict = field(default_factory=dict)


class SongGenEngine(ABC):
    """Abstract base class for song generation engines.

    All song generation engines must implement this interface to be usable in the system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this engine."""
        pass

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine (e.g., 'levo', 'suno')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available (dependencies installed, etc.)."""
        pass

    @abstractmethod
    def get_info(self) -> EngineInfo:
        """Get detailed information about this engine."""
        pass

    @abstractmethod
    def generate(
        self,
        lyrics: str,
        output_path: Path,
        description: Optional[str] = None,
        generate_type: str = "mixed",
        **kwargs,
    ) -> SongGenResult:
        """Generate a song from lyrics.

        Args:
            lyrics: Structured lyrics with section markers (e.g., [verse], [chorus]).
            output_path: Where to save the generated audio.
            description: Musical style description (e.g., "female, pop, happy, piano").
            generate_type: Output type - "mixed" (default), "separate", "vocal", or "bgm".
            **kwargs: Engine-specific parameters.

        Returns:
            SongGenResult with status and metadata.
        """
        pass

    def get_setup_instructions(self) -> str:
        """Get setup instructions for this engine."""
        return f"No setup instructions available for {self.name}."
