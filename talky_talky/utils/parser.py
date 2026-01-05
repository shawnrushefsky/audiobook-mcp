"""Text parsing utilities for screenplay-format scripts.

Expected format:
    CHARACTER NAME: Dialogue text here.

    ANOTHER CHARACTER: More dialogue.
    This continues on the next line.

    NARRATOR: Description of the scene.

Rules:
- Character name is everything before the first `: ` (colon-space)
- Dialogue/narration is everything after
- Blank lines separate segments
- Multi-line text continues until the next CHARACTER: line or blank line
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedSegment:
    """A parsed segment from screenplay-format text."""

    text: str
    character_name: Optional[str] = None
    is_narration: bool = False


def parse_screenplay(text: str) -> list[ParsedSegment]:
    """Parse screenplay-format text into segments.

    Format: CHARACTER NAME: dialogue text

    Args:
        text: The screenplay-format text to parse.

    Returns:
        List of ParsedSegment objects with character names and text.
    """
    segments: list[ParsedSegment] = []
    lines = text.strip().split("\n")

    current_character: Optional[str] = None
    current_text_lines: list[str] = []

    def flush_segment():
        """Save the current segment if there's content."""
        nonlocal current_character, current_text_lines
        if current_text_lines:
            text_content = " ".join(current_text_lines).strip()
            if text_content:
                is_narration = current_character and current_character.upper() == "NARRATOR"
                segments.append(
                    ParsedSegment(
                        text=text_content,
                        character_name=current_character,
                        is_narration=is_narration or False,
                    )
                )
        current_text_lines = []

    # Pattern to match "CHARACTER NAME: text" at the start of a line
    # Character names can include letters, numbers, spaces, and common punctuation
    character_pattern = re.compile(r"^([A-Z][A-Za-z0-9 '\-_.]+?):\s*(.*)$")

    for line in lines:
        line = line.strip()

        # Blank line ends current segment
        if not line:
            flush_segment()
            current_character = None
            continue

        # Check if this line starts a new character's dialogue
        match = character_pattern.match(line)
        if match:
            # Save previous segment before starting new one
            flush_segment()

            current_character = match.group(1).strip()
            dialogue = match.group(2).strip()
            if dialogue:
                current_text_lines.append(dialogue)
        else:
            # Continuation of current segment
            current_text_lines.append(line)

    # Don't forget the last segment
    flush_segment()

    return segments


def parse_text(text: str) -> list[ParsedSegment]:
    """Parse text into segments.

    This is an alias for parse_screenplay for backward compatibility.
    """
    return parse_screenplay(text)


def extract_character_names(text: str) -> list[str]:
    """Extract character names from screenplay-format text.

    Args:
        text: The screenplay-format text to analyze.

    Returns:
        Sorted list of unique character names found.
    """
    names: set[str] = set()

    # Pattern to match "CHARACTER NAME:" at the start of a line
    character_pattern = re.compile(r"^([A-Z][A-Za-z0-9 '\-_.]+?):\s*", re.MULTILINE)

    for match in character_pattern.finditer(text):
        name = match.group(1).strip()
        if name:
            names.add(name)

    return sorted(names)


def clean_for_tts(text: str) -> str:
    """Clean text for TTS (remove excessive whitespace, normalize quotes, etc.)."""
    result = text
    result = re.sub(r"\s+", " ", result)  # Normalize whitespace
    result = result.replace('"', '"').replace('"', '"')  # Normalize curly double quotes
    result = result.replace("'", "'").replace("'", "'")  # Normalize curly single quotes
    result = result.replace("«", '"').replace("»", '"')  # Convert guillemets
    result = result.replace("—", " - ")  # Em dash with spaces
    result = result.replace("–", " - ")  # En dash with spaces
    result = result.replace("...", "…")  # Normalize ellipsis
    return result.strip()
