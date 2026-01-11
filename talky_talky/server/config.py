"""Server configuration utilities."""

import json
from dataclasses import asdict
from pathlib import Path

# Server version
VERSION = "0.2.0"

# Configuration paths
CONFIG_DIR = Path.home() / ".config" / "talky-talky"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "talky-talky"


def _load_config() -> dict:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_config(config: dict) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_output_dir() -> Path:
    """Get the configured output directory, creating it if needed."""
    config = _load_config()
    output_dir = Path(config.get("output_directory", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_output_path(output_path: str) -> str:
    """Resolve output path, using default directory if path is just a filename."""
    path = Path(output_path)
    # If it's just a filename (no directory components), use the configured output dir
    if path.parent == Path(".") or str(path.parent) == "":
        return str(get_output_dir() / path.name)
    # Otherwise, ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def to_dict(obj) -> dict:
    """Convert dataclass to dict, handling nested objects."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    elif isinstance(obj, dict):
        return obj
    else:
        return {"value": obj}
