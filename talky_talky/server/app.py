#!/usr/bin/env python3
"""Talky Talky - TTS, STT, and Audio Processing MCP Server.

A modular MCP server providing:
- TTS: 11 engines (voice cloning, design, preset voices)
- Transcription: Whisper, Faster-Whisper
- Song generation: LeVo, ACE-Step
- Audio processing: format conversion, editing, effects, modulation
- Asset management: local indexing, Freesound, Jamendo

Use capabilities() to discover available features.
Use get_engines_info(subsystem) for detailed documentation.
"""

from mcp.server.fastmcp import FastMCP

from .tools import register_all_tools


# Initialize MCP server
mcp = FastMCP("talky-talky")

# Register all tools from modular tool files
register_all_tools(mcp)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
