"""Audio registration and stitching tools."""

import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..db.connection import get_database, get_audiobook_dir, get_current_project_path
from .chapters import get_chapter, list_chapters, get_chapters_with_stats
from .segments import list_segments, get_segment
from .projects import get_project_info
from ..utils.ffmpeg import (
    check_ffmpeg,
    validate_audio_file,
    get_audio_duration,
    concatenate_audio_files,
    create_audiobook_with_chapters,
    convert_audio_format,
    ChapterMarker,
)


@dataclass
class RegisterResult:
    segment_id: str
    audio_path: str
    duration_ms: int
    copied_to: str


@dataclass
class MissingSegment:
    id: str
    sort_order: int
    text_preview: str
    character_name: Optional[str]


@dataclass
class ChapterAudioStatus:
    chapter_id: str
    chapter_title: str
    total_segments: int
    segments_with_audio: int
    segments_missing_audio: int
    total_duration_ms: int
    ready_to_stitch: bool
    missing_segments: list[MissingSegment]


@dataclass
class ChapterStatus:
    id: str
    title: str
    sort_order: int
    segment_count: int
    segments_with_audio: int
    duration_ms: int
    ready: bool


@dataclass
class StitchStatus:
    total_chapters: int
    chapters_ready: int
    total_segments: int
    segments_with_audio: int
    total_duration_ms: int
    ready_to_stitch_book: bool
    chapters: list[ChapterStatus]


@dataclass
class StitchChapterResult:
    chapter_id: str
    chapter_title: str
    output_path: str
    segment_count: int
    total_duration_ms: int


@dataclass
class ChapterTimestamp:
    title: str
    start_ms: int


@dataclass
class StitchBookResult:
    output_path: str
    chapter_count: int
    total_duration_ms: int
    chapters: list[ChapterTimestamp]


def register_segment_audio(
    segment_id: str,
    audio_path: str,
    duration_ms: Optional[int] = None,
) -> RegisterResult:
    """Register an audio file for a segment."""
    db = get_database()
    cursor = db.cursor()
    project_path = get_current_project_path()

    if not project_path:
        raise RuntimeError("No project is currently open")

    # Verify segment exists
    segment = get_segment(segment_id)
    if not segment:
        raise ValueError(f"Segment not found: {segment_id}")

    # Validate audio file
    validation = validate_audio_file(audio_path)
    if not validation.valid:
        raise ValueError(f"Invalid audio file: {validation.error}")

    # Use provided duration or detected duration
    actual_duration = duration_ms if duration_ms is not None else validation.duration_ms

    # Copy audio to project's audio directory
    audiobook_dir = get_audiobook_dir(project_path)
    segments_dir = Path(audiobook_dir) / "audio" / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(audio_path).suffix or ".mp3"
    dest_path = segments_dir / f"{segment_id}{ext}"
    shutil.copy2(audio_path, dest_path)

    # Store relative path in database
    relative_path = f"audio/segments/{segment_id}{ext}"

    # Update segment with audio info
    cursor.execute(
        "UPDATE segments SET audio_path = ?, duration_ms = ? WHERE id = ?",
        (relative_path, actual_duration, segment_id),
    )
    db.commit()

    return RegisterResult(
        segment_id=segment_id,
        audio_path=relative_path,
        duration_ms=actual_duration or 0,
        copied_to=str(dest_path),
    )


def get_chapter_audio_status(chapter_id: str) -> ChapterAudioStatus:
    """Get the status of audio for a chapter."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    segments = list_segments(chapter_id)
    with_audio = [s for s in segments if s.audio_path]
    missing_audio = [s for s in segments if not s.audio_path]

    # Get character names for missing segments
    missing_with_names: list[MissingSegment] = []
    for s in missing_audio:
        character_name: Optional[str] = None
        if s.character_id:
            cursor.execute("SELECT name FROM characters WHERE id = ?", (s.character_id,))
            row = cursor.fetchone()
            if row:
                character_name = row["name"]

        text_preview = s.text_content[:50] + "..." if len(s.text_content) > 50 else s.text_content

        missing_with_names.append(
            MissingSegment(
                id=s.id,
                sort_order=s.sort_order,
                text_preview=text_preview,
                character_name=character_name,
            )
        )

    total_duration = sum(s.duration_ms or 0 for s in with_audio)

    return ChapterAudioStatus(
        chapter_id=chapter_id,
        chapter_title=chapter.title,
        total_segments=len(segments),
        segments_with_audio=len(with_audio),
        segments_missing_audio=len(missing_audio),
        total_duration_ms=total_duration,
        ready_to_stitch=len(missing_audio) == 0 and len(segments) > 0,
        missing_segments=missing_with_names,
    )


def stitch_chapter(
    chapter_id: str,
    output_filename: Optional[str] = None,
) -> StitchChapterResult:
    """Stitch all segments in a chapter into a single audio file."""
    project_path = get_current_project_path()

    if not project_path:
        raise RuntimeError("No project is currently open")

    # Check ffmpeg availability
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    # Verify chapter exists and has all audio
    status = get_chapter_audio_status(chapter_id)
    if not status.ready_to_stitch:
        raise ValueError(
            f"Chapter is not ready to stitch. Missing {status.segments_missing_audio} audio files."
        )

    audiobook_dir = get_audiobook_dir(project_path)
    segments = list_segments(chapter_id)

    # Get absolute paths for all audio files
    input_files = [str(Path(audiobook_dir) / s.audio_path) for s in segments if s.audio_path]

    # Determine output path
    import re

    safe_title = re.sub(r"[^a-zA-Z0-9]", "_", status.chapter_title)
    filename = output_filename or f"{safe_title}.mp3"
    output_dir = Path(audiobook_dir) / "exports" / "chapters"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Concatenate files
    concatenate_audio_files(input_files, str(output_path), "mp3")

    # Get actual duration of output
    duration = get_audio_duration(str(output_path))

    return StitchChapterResult(
        chapter_id=chapter_id,
        chapter_title=status.chapter_title,
        output_path=str(output_path),
        segment_count=len(segments),
        total_duration_ms=duration,
    )


def get_stitch_status() -> StitchStatus:
    """Get the overall stitch status for the book."""
    chapters = get_chapters_with_stats()

    chapter_statuses: list[ChapterStatus] = []
    for ch in chapters:
        status = get_chapter_audio_status(ch.id)
        chapter_statuses.append(
            ChapterStatus(
                id=ch.id,
                title=ch.title,
                sort_order=ch.sort_order,
                segment_count=ch.segment_count,
                segments_with_audio=status.segments_with_audio,
                duration_ms=status.total_duration_ms,
                ready=status.ready_to_stitch,
            )
        )

    total_segments = sum(ch.segment_count for ch in chapter_statuses)
    total_with_audio = sum(ch.segments_with_audio for ch in chapter_statuses)
    total_duration = sum(ch.duration_ms for ch in chapter_statuses)
    chapters_ready = sum(1 for ch in chapter_statuses if ch.ready)

    return StitchStatus(
        total_chapters=len(chapters),
        chapters_ready=chapters_ready,
        total_segments=total_segments,
        segments_with_audio=total_with_audio,
        total_duration_ms=total_duration,
        ready_to_stitch_book=chapters_ready == len(chapters) and len(chapters) > 0,
        chapters=chapter_statuses,
    )


def stitch_book(
    output_filename: Optional[str] = None,
    include_chapter_markers: bool = True,
) -> StitchBookResult:
    """Stitch all chapters into a complete audiobook."""
    project_path = get_current_project_path()

    if not project_path:
        raise RuntimeError("No project is currently open")

    # Check ffmpeg availability
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    # Check overall status
    status = get_stitch_status()
    if not status.ready_to_stitch_book:
        not_ready = [ch.title for ch in status.chapters if not ch.ready]
        raise ValueError(f"Book is not ready to stitch. Chapters not ready: {', '.join(not_ready)}")

    audiobook_dir = get_audiobook_dir(project_path)
    info = get_project_info()
    chapters = list_chapters()

    # First, stitch each chapter and collect the chapter files
    chapter_files: list[str] = []
    chapter_markers: list[ChapterMarker] = []
    current_ms = 0

    for chapter in chapters:
        # Stitch the chapter
        result = stitch_chapter(chapter.id)
        chapter_files.append(result.output_path)

        # Record chapter marker
        chapter_markers.append(
            ChapterMarker(
                title=chapter.title,
                start_ms=current_ms,
            )
        )

        current_ms += result.total_duration_ms

    # Determine output path
    import re

    safe_title = re.sub(r"[^a-zA-Z0-9]", "_", info.project.title)
    filename = output_filename or f"{safe_title}.mp3"
    output_dir = Path(audiobook_dir) / "exports" / "book"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Create final audiobook
    if include_chapter_markers:
        metadata = {
            "title": info.project.title,
            "artist": info.project.author,
            "album": info.project.title,
        }
        create_audiobook_with_chapters(chapter_files, str(output_path), chapter_markers, metadata)
    else:
        concatenate_audio_files(chapter_files, str(output_path), "mp3")

    # Get actual duration
    duration = get_audio_duration(str(output_path))

    return StitchBookResult(
        output_path=str(output_path),
        chapter_count=len(chapters),
        total_duration_ms=duration,
        chapters=[ChapterTimestamp(title=cm.title, start_ms=cm.start_ms) for cm in chapter_markers],
    )


def clear_segment_audio(segment_id: str) -> None:
    """Clear audio from a segment."""
    db = get_database()
    cursor = db.cursor()

    # Verify segment exists
    segment = get_segment(segment_id)
    if not segment:
        raise ValueError(f"Segment not found: {segment_id}")

    # Clear audio path and duration
    cursor.execute(
        "UPDATE segments SET audio_path = NULL, duration_ms = NULL WHERE id = ?", (segment_id,)
    )
    db.commit()


@dataclass
class ConvertResult:
    original_path: str
    converted_path: str
    original_format: str
    target_format: str
    original_size_bytes: int
    converted_size_bytes: int
    compression_ratio: float
    duration_ms: int


def convert_segment_audio(
    segment_id: str,
    target_format: str = "mp3",
) -> ConvertResult:
    """Convert a segment's audio to a more compact format.

    Args:
        segment_id: The segment whose audio to convert
        target_format: Output format (mp3, m4a). Default: mp3

    Returns:
        ConvertResult with paths and size comparison
    """
    project_path = get_current_project_path()
    if not project_path:
        raise RuntimeError("No project is currently open")

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if target_format not in ("mp3", "m4a"):
        raise ValueError("target_format must be 'mp3' or 'm4a'")

    segment = get_segment(segment_id)
    if not segment:
        raise ValueError(f"Segment not found: {segment_id}")

    if not segment.audio_path:
        raise ValueError(f"Segment has no audio: {segment_id}")

    audiobook_dir = get_audiobook_dir(project_path)
    original_path = Path(audiobook_dir) / segment.audio_path

    if not original_path.exists():
        raise FileNotFoundError(f"Audio file not found: {original_path}")

    original_format = original_path.suffix.lstrip(".")
    original_size = original_path.stat().st_size

    # Create converted path
    converted_path = original_path.with_suffix(f".{target_format}")

    # Convert
    convert_audio_format(str(original_path), str(converted_path), target_format)

    converted_size = converted_path.stat().st_size
    duration = get_audio_duration(str(converted_path))

    return ConvertResult(
        original_path=str(original_path),
        converted_path=str(converted_path),
        original_format=original_format,
        target_format=target_format,
        original_size_bytes=original_size,
        converted_size_bytes=converted_size,
        compression_ratio=round(original_size / converted_size, 2) if converted_size else 0,
        duration_ms=duration,
    )


def convert_voice_sample(
    sample_id: str,
    target_format: str = "mp3",
) -> ConvertResult:
    """Convert a voice sample to a more compact format.

    Args:
        sample_id: The voice sample to convert
        target_format: Output format (mp3, m4a). Default: mp3

    Returns:
        ConvertResult with paths and size comparison
    """
    from .voice_samples import get_voice_sample

    project_path = get_current_project_path()
    if not project_path:
        raise RuntimeError("No project is currently open")

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if target_format not in ("mp3", "m4a"):
        raise ValueError("target_format must be 'mp3' or 'm4a'")

    sample = get_voice_sample(sample_id)
    if not sample:
        raise ValueError(f"Voice sample not found: {sample_id}")

    audiobook_dir = get_audiobook_dir(project_path)
    original_path = Path(audiobook_dir) / sample.sample_path

    if not original_path.exists():
        raise FileNotFoundError(f"Audio file not found: {original_path}")

    original_format = original_path.suffix.lstrip(".")
    original_size = original_path.stat().st_size

    # Create converted path
    converted_path = original_path.with_suffix(f".{target_format}")

    # Convert
    convert_audio_format(str(original_path), str(converted_path), target_format)

    converted_size = converted_path.stat().st_size
    duration = get_audio_duration(str(converted_path))

    return ConvertResult(
        original_path=str(original_path),
        converted_path=str(converted_path),
        original_format=original_format,
        target_format=target_format,
        original_size_bytes=original_size,
        converted_size_bytes=converted_size,
        compression_ratio=round(original_size / converted_size, 2) if converted_size else 0,
        duration_ms=duration,
    )


def get_audio_file_path(
    segment_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    format: str = "original",
) -> dict:
    """Get the file path for a segment or voice sample audio.

    Args:
        segment_id: Get audio for this segment
        sample_id: Get audio for this voice sample
        format: 'original' for the source file, or 'mp3'/'m4a' if converted version exists

    Returns:
        Dict with path, exists, format, and size_bytes
    """
    from .voice_samples import get_voice_sample

    project_path = get_current_project_path()
    if not project_path:
        raise RuntimeError("No project is currently open")

    if not segment_id and not sample_id:
        raise ValueError("Either segment_id or sample_id is required")

    audiobook_dir = get_audiobook_dir(project_path)

    if segment_id:
        segment = get_segment(segment_id)
        if not segment:
            raise ValueError(f"Segment not found: {segment_id}")
        if not segment.audio_path:
            raise ValueError(f"Segment has no audio: {segment_id}")
        base_path = Path(audiobook_dir) / segment.audio_path
    else:
        sample = get_voice_sample(sample_id)
        if not sample:
            raise ValueError(f"Voice sample not found: {sample_id}")
        base_path = Path(audiobook_dir) / sample.sample_path

    # Determine which file to return
    if format == "original":
        target_path = base_path
    else:
        target_path = base_path.with_suffix(f".{format}")

    exists = target_path.exists()
    size_bytes = target_path.stat().st_size if exists else 0
    actual_format = target_path.suffix.lstrip(".") if exists else None

    return {
        "path": str(target_path),
        "exists": exists,
        "format": actual_format,
        "size_bytes": size_bytes,
    }
