"""Tests for input validation on MCP tools."""

import pytest


class TestCharacterValidation:
    """Tests for character input validation."""

    def test_empty_character_name_rejected(self, initialized_project):
        """Test that empty character name is rejected."""
        from talky_talky.server import create_character

        with pytest.raises(ValueError, match="cannot be empty"):
            create_character("")

        with pytest.raises(ValueError, match="cannot be empty"):
            create_character("   ")

    def test_long_character_name_rejected(self, initialized_project):
        """Test that overly long character name is rejected."""
        from talky_talky.server import create_character

        long_name = "A" * 201
        with pytest.raises(ValueError, match="cannot exceed 200"):
            create_character(long_name)


class TestChapterValidation:
    """Tests for chapter input validation."""

    def test_empty_chapter_title_rejected(self, initialized_project):
        """Test that empty chapter title is rejected."""
        from talky_talky.server import create_chapter

        with pytest.raises(ValueError, match="cannot be empty"):
            create_chapter("")

        with pytest.raises(ValueError, match="cannot be empty"):
            create_chapter("   ")

    def test_long_chapter_title_rejected(self, initialized_project):
        """Test that overly long chapter title is rejected."""
        from talky_talky.server import create_chapter

        long_title = "A" * 501
        with pytest.raises(ValueError, match="cannot exceed 500"):
            create_chapter(long_title)


class TestSegmentValidation:
    """Tests for segment input validation."""

    def test_empty_segment_text_rejected(self, project_with_chapters):
        """Test that empty segment text is rejected."""
        from talky_talky.server import create_segment

        chapter_id = project_with_chapters["chapters"]["chapter1"].id

        with pytest.raises(ValueError, match="cannot be empty"):
            create_segment(chapter_id, "")

        with pytest.raises(ValueError, match="cannot be empty"):
            create_segment(chapter_id, "   ")

    def test_empty_chapter_id_rejected(self, initialized_project):
        """Test that empty chapter ID is rejected."""
        from talky_talky.server import create_segment

        with pytest.raises(ValueError, match="chapter_id is required"):
            create_segment("", "Some text")

    def test_long_segment_text_rejected(self, project_with_chapters):
        """Test that overly long segment text is rejected."""
        from talky_talky.server import create_segment

        chapter_id = project_with_chapters["chapters"]["chapter1"].id
        long_text = "A" * 50001

        with pytest.raises(ValueError, match="cannot exceed 50,000"):
            create_segment(chapter_id, long_text)


class TestBulkSegmentValidation:
    """Tests for bulk segment creation validation."""

    def test_empty_segments_list_rejected(self, project_with_chapters):
        """Test that empty segments list is rejected."""
        from talky_talky.server import bulk_create_segments

        chapter_id = project_with_chapters["chapters"]["chapter1"].id

        with pytest.raises(ValueError, match="cannot be empty"):
            bulk_create_segments(chapter_id, [])

    def test_too_many_segments_rejected(self, project_with_chapters):
        """Test that too many segments in one call is rejected."""
        from talky_talky.server import bulk_create_segments

        chapter_id = project_with_chapters["chapters"]["chapter1"].id
        segments = [{"text_content": f"Segment {i}"} for i in range(1001)]

        with pytest.raises(ValueError, match="Cannot add more than 1000"):
            bulk_create_segments(chapter_id, segments)

    def test_segment_without_text_rejected(self, project_with_chapters):
        """Test that segment without text_content is rejected."""
        from talky_talky.server import bulk_create_segments

        chapter_id = project_with_chapters["chapters"]["chapter1"].id
        segments = [{"text_content": "Valid"}, {"character_id": "some-id"}]  # Missing text_content

        with pytest.raises(ValueError, match="missing required 'text_content'"):
            bulk_create_segments(chapter_id, segments)


class TestImportValidation:
    """Tests for text import validation."""

    def test_empty_import_text_rejected(self, project_with_chapters):
        """Test that empty import text is rejected."""
        from talky_talky.server import import_text_to_chapter

        chapter_id = project_with_chapters["chapters"]["chapter1"].id

        with pytest.raises(ValueError, match="cannot be empty"):
            import_text_to_chapter(chapter_id, "")

    def test_long_import_text_rejected(self, project_with_chapters):
        """Test that overly long import text is rejected."""
        from talky_talky.server import import_text_to_chapter

        chapter_id = project_with_chapters["chapters"]["chapter1"].id
        long_text = "NARRATOR: " + "A" * 500001

        with pytest.raises(ValueError, match="cannot exceed 500,000"):
            import_text_to_chapter(chapter_id, long_text)
