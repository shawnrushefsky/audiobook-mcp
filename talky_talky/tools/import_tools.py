"""Import tools for parsing and importing screenplay-format text into chapters.

Expected screenplay format:
    CHARACTER NAME: Dialogue or narration text here.

    NARRATOR: The scene description goes here.

    CAPTAIN BLACKBEARD: Arr, me hearties!
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..db.connection import get_database
from .chapters import get_chapter
from .characters import get_character, list_characters
from .segments import bulk_add_segments, list_segments, update_segment
from ..utils.parser import parse_screenplay, extract_character_names, clean_for_tts


@dataclass
class ImportResult:
    segments_created: int
    assigned_segments: int
    unassigned_segments: int
    character_names_in_script: list[str]
    matched_characters: list[str]
    unmatched_characters: list[str]


@dataclass
class AssignResult:
    updated_count: int
    updated_segments: list[str]


@dataclass
class CharacterLine:
    segment_id: str
    chapter_title: str
    text: str
    has_audio: bool


@dataclass
class CharacterLinesExport:
    character_name: str
    total_lines: int
    total_characters: int
    lines: list[CharacterLine]


@dataclass
class DialogueSuggestion:
    segment_id: str
    text_preview: str
    potential_speaker: Optional[str]


@dataclass
class DialogueDetection:
    total_segments: int
    unassigned_segments: int
    detected_names: list[str]
    suggestions: list[DialogueSuggestion]


@dataclass
class CharacterDistribution:
    character_id: str
    character_name: str
    is_narrator: bool
    segment_count: int
    total_characters: int
    has_voice: bool


@dataclass
class LineDistribution:
    total_segments: int
    assigned_segments: int
    unassigned_segments: int
    by_character: list[CharacterDistribution]


def import_chapter_text(
    chapter_id: str,
    text: str,
    default_character_id: Optional[str] = None,
) -> ImportResult:
    """Import screenplay-format text into a chapter.

    Expected format:
        CHARACTER NAME: Dialogue or narration text.

        NARRATOR: Scene description here.

    Character names in the script are matched (case-insensitive) to existing
    characters in the project. Unmatched character names result in unassigned
    segments that can be assigned later.

    Args:
        chapter_id: The chapter to import into.
        text: Screenplay-format text to import.
        default_character_id: Optional character to assign to segments without
            a recognized character name.

    Returns:
        ImportResult with statistics about the import.
    """
    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    # Verify default character exists if provided
    if default_character_id:
        character = get_character(default_character_id)
        if not character:
            raise ValueError(f"Character not found: {default_character_id}")

    # Get existing characters for name matching
    existing_characters = list_characters()
    # Build a case-insensitive lookup: lowercase name -> character
    name_to_character = {c.name.lower(): c for c in existing_characters}

    # Parse the screenplay-format text
    parsed = parse_screenplay(text)
    script_names = extract_character_names(text)

    # Track which names matched and which didn't
    matched_names: set[str] = set()
    unmatched_names: set[str] = set()

    # Clean and prepare segments for insertion
    segments_to_add = []
    assigned_count = 0
    unassigned_count = 0

    for seg in parsed:
        character_id = None

        if seg.character_name:
            # Try to match the character name (case-insensitive)
            lookup_name = seg.character_name.lower()
            if lookup_name in name_to_character:
                character_id = name_to_character[lookup_name].id
                matched_names.add(seg.character_name)
            else:
                unmatched_names.add(seg.character_name)
                # Use default character if provided
                character_id = default_character_id

        if character_id:
            assigned_count += 1
        else:
            unassigned_count += 1

        segments_to_add.append(
            {
                "text_content": clean_for_tts(seg.text),
                "character_id": character_id,
            }
        )

    # Bulk add segments
    bulk_add_segments(chapter_id, segments_to_add)

    return ImportResult(
        segments_created=len(parsed),
        assigned_segments=assigned_count,
        unassigned_segments=unassigned_count,
        character_names_in_script=script_names,
        matched_characters=sorted(matched_names),
        unmatched_characters=sorted(unmatched_names),
    )


def assign_dialogue(
    chapter_id: str,
    pattern: str,
    character_id: str,
) -> AssignResult:
    """Assign a character to all dialogue segments matching a pattern."""
    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    # Verify character exists
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    # Get all segments in the chapter
    segments = list_segments(chapter_id)

    # Find segments matching the pattern (case-insensitive)
    regex = re.compile(pattern, re.IGNORECASE)
    matching_segments = [s for s in segments if regex.search(s.text_content)]

    # Update matching segments
    updated_ids: list[str] = []
    for segment in matching_segments:
        update_segment(segment.id, character_id=character_id)
        updated_ids.append(segment.id)

    return AssignResult(
        updated_count=len(updated_ids),
        updated_segments=updated_ids,
    )


def export_character_lines(character_id: str) -> CharacterLinesExport:
    """Export all lines for a specific character (for batch voice generation)."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    # Get all segments for this character with chapter info
    cursor.execute(
        """
        SELECT s.*, ch.title as chapter_title
        FROM segments s
        JOIN chapters ch ON s.chapter_id = ch.id
        WHERE s.character_id = ?
        ORDER BY ch.sort_order ASC, s.sort_order ASC
    """,
        (character_id,),
    )

    rows = cursor.fetchall()

    lines = [
        CharacterLine(
            segment_id=row["id"],
            chapter_title=row["chapter_title"],
            text=row["text_content"],
            has_audio=row["audio_path"] is not None,
        )
        for row in rows
    ]

    total_characters = sum(len(line.text) for line in lines)

    return CharacterLinesExport(
        character_name=character.name,
        total_lines=len(lines),
        total_characters=total_characters,
        lines=lines,
    )


def detect_dialogue(chapter_id: str) -> DialogueDetection:
    """Detect potential dialogue and suggest character assignments."""
    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    segments = list_segments(chapter_id)
    full_text = " ".join(s.text_content for s in segments)
    detected_names = extract_character_names(full_text)

    # Get existing characters for matching
    characters = list_characters()
    character_names = {c.name.lower() for c in characters}

    # Find unassigned segments and try to suggest speakers
    unassigned = [s for s in segments if not s.character_id]
    suggestions = []

    for seg in unassigned[:20]:  # Limit to first 20
        # Try to find a speaker from detected names or existing characters
        potential_speaker: Optional[str] = None

        for name in detected_names:
            if name.lower() in character_names:
                potential_speaker = name
                break

        text_preview = (
            seg.text_content[:80] + "..." if len(seg.text_content) > 80 else seg.text_content
        )

        suggestions.append(
            DialogueSuggestion(
                segment_id=seg.id,
                text_preview=text_preview,
                potential_speaker=potential_speaker,
            )
        )

    return DialogueDetection(
        total_segments=len(segments),
        unassigned_segments=len(unassigned),
        detected_names=detected_names,
        suggestions=suggestions,
    )


def get_line_distribution() -> LineDistribution:
    """Get a summary of character line distribution across the project."""
    db = get_database()
    cursor = db.cursor()

    # Total counts
    cursor.execute("SELECT COUNT(*) as total FROM segments")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as assigned FROM segments WHERE character_id IS NOT NULL")
    assigned = cursor.fetchone()["assigned"]

    # By character breakdown
    cursor.execute("""
        SELECT
            c.id as character_id,
            c.name as character_name,
            c.is_narrator,
            c.voice_config,
            COUNT(s.id) as segment_count,
            COALESCE(SUM(LENGTH(s.text_content)), 0) as total_characters
        FROM characters c
        LEFT JOIN segments s ON s.character_id = c.id
        GROUP BY c.id
        ORDER BY segment_count DESC
    """)

    rows = cursor.fetchall()

    by_character = [
        CharacterDistribution(
            character_id=row["character_id"],
            character_name=row["character_name"],
            is_narrator=bool(row["is_narrator"]),
            segment_count=row["segment_count"],
            total_characters=row["total_characters"],
            has_voice=row["voice_config"] is not None,
        )
        for row in rows
    ]

    return LineDistribution(
        total_segments=total,
        assigned_segments=assigned,
        unassigned_segments=total - assigned,
        by_character=by_character,
    )
