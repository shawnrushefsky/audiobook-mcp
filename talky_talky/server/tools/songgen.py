"""Song generation tools."""

from typing import Optional

from ...tools.songgen import generate as generate_song
from ...tools.songgen.levo import (
    check_models_downloaded as check_levo_models,
    download_models as download_levo_models,
    MODELS as LEVO_MODELS,
    STRUCTURE_MARKERS,
    STYLE_PROMPTS,
)
from ...tools.songgen.acestep import (
    check_models_downloaded as check_acestep_models,
    download_models as download_acestep_models_impl,
)
from ..config import to_dict, resolve_output_path


def register_songgen_tools(mcp):
    """Register song generation tools with the MCP server."""

    @mcp.tool()
    def generate_song_levo(
        lyrics: str,
        output_path: str,
        description: str = "female, pop, happy",
        generate_type: str = "mixed",
        prompt_audio_path: Optional[str] = None,
        auto_prompt_style: Optional[str] = None,
        model_name: str = "base-new",
        low_mem: bool = False,
    ) -> dict:
        """Generate a song with LeVo (Tencent SongGeneration).

        Args:
            lyrics: Structured lyrics with section markers:
                [intro], [verse], [chorus], [bridge], [outro]
                Separate sections with ';', sentences with '.'.
                Example: "[intro] ; [verse] Hello world. Singing today ; [chorus] ..."
            output_path: Where to save the song.
            description: Style tags: "female, pop, happy, piano".
            generate_type: mixed, separate, vocal, or bgm.
            prompt_audio_path: ~10s reference audio for style.
            auto_prompt_style: Pop, Rock, Jazz, R&B, Electronic, Folk, etc.
            model_name: base-new (2m30s), base-full (4m30s), large (best quality).
            low_mem: Use less VRAM (slower).

        Returns status, output_path, duration_ms.
        Note: Requires CUDA GPU with 10-28GB VRAM.
        """
        result = generate_song(
            engine_id="levo",
            lyrics=lyrics,
            output_path=resolve_output_path(output_path),
            description=description,
            generate_type=generate_type,
            prompt_audio_path=prompt_audio_path,
            auto_prompt_style=auto_prompt_style,
            model_name=model_name,
            low_mem=low_mem,
        )
        return to_dict(result)

    @mcp.tool()
    def generate_song_acestep(
        prompt: str,
        output_path: str,
        lyrics: Optional[str] = None,
        audio_duration: float = 60.0,
        infer_steps: int = 27,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        seed: Optional[int] = None,
        cpu_offload: bool = False,
        quantized: bool = False,
    ) -> dict:
        """Generate a song with ACE-Step (MPS + CUDA).

        Args:
            prompt: Style tags: "female vocals, pop, upbeat, synth".
            output_path: Where to save the song.
            lyrics: Optional lyrics with [verse], [chorus], etc.
            audio_duration: Duration in seconds (max 240).
            infer_steps: 27 (fast) or 60 (quality).
            guidance_scale: Prompt adherence strength.
            scheduler_type: euler, heun, or pingpong.
            seed: Random seed for reproducibility.
            cpu_offload: Reduce VRAM usage.
            quantized: Use 8-bit model.

        Returns status, output_path, duration_ms.
        Note: Apple Silicon needs 36GB+ unified memory.
        """
        result = generate_song(
            engine_id="acestep",
            lyrics=prompt,
            output_path=resolve_output_path(output_path),
            description=lyrics,
            audio_duration=audio_duration,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            seed=seed,
            cpu_offload=cpu_offload,
            quantized=quantized,
        )
        return to_dict(result)

    @mcp.tool()
    def get_songgen_model_status(model_name: str = "base-new") -> dict:
        """Get download status of song generation models."""
        return check_levo_models(model_name)

    @mcp.tool()
    def download_songgen_models(model_name: str = "base-new", force: bool = False) -> dict:
        """Download LeVo song generation models (~10GB).

        Args:
            model_name: base-new, base-full, or large.
            force: Re-download if exists.
        """
        return download_levo_models(model_name, force=force)

    @mcp.tool()
    def get_acestep_model_status() -> dict:
        """Get download status of ACE-Step models."""
        return check_acestep_models()

    @mcp.tool()
    def download_acestep_models(force: bool = False) -> dict:
        """Download ACE-Step models (~7GB)."""
        return download_acestep_models_impl(force=force)

    @mcp.tool()
    def get_songgen_lyrics_format() -> dict:
        """Get lyrics format guide for song generation."""
        return {
            "markers": STRUCTURE_MARKERS,
            "section_sep": ";",
            "sentence_sep": ".",
            "style_prompts": STYLE_PROMPTS,
            "models": {k: v["description"] for k, v in LEVO_MODELS.items()},
            "example": (
                "[intro-short] ; [verse] Hello world. Singing today ; "
                "[chorus] This is the hook. Sing along ; [outro-short]"
            ),
        }
