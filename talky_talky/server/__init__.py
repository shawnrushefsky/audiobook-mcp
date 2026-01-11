"""Server package - configuration and utilities for the MCP server."""

from .config import (
    VERSION,
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_OUTPUT_DIR,
    _load_config,
    _save_config,
    get_output_dir,
    resolve_output_path,
    to_dict,
)

__all__ = [
    "VERSION",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "DEFAULT_OUTPUT_DIR",
    "_load_config",
    "_save_config",
    "get_output_dir",
    "resolve_output_path",
    "to_dict",
]
