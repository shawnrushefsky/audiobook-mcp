"""MCP tool modules for talky-talky server.

This package contains modular tool definitions organized by category:
- discovery: System capabilities and availability checks
- tts: Text-to-speech generation (consolidated speak() tool)
- songgen: Song generation tools
- audio: Audio processing utilities
- transcription: Speech-to-text tools
- analysis: Audio analysis and verification
- assets: Asset library management
- workflows: High-level workflow tools
"""

from .discovery import register_discovery_tools
from .tts import register_tts_tools
from .songgen import register_songgen_tools
from .audio import register_audio_tools
from .transcription import register_transcription_tools
from .analysis import register_analysis_tools
from .assets import register_asset_tools
from .workflows import register_workflow_tools
from .resources import register_resources


def register_all_tools(mcp):
    """Register all tool modules with the MCP server."""
    register_discovery_tools(mcp)
    register_tts_tools(mcp)
    register_songgen_tools(mcp)
    register_audio_tools(mcp)
    register_transcription_tools(mcp)
    register_analysis_tools(mcp)
    register_asset_tools(mcp)
    register_workflow_tools(mcp)
    register_resources(mcp)


__all__ = ["register_all_tools"]
