"""Tests for project management."""

from pathlib import Path

import pytest

from audiobook_mcp.tools.projects import init_project, get_project_info, update_project


class TestProjectInitialization:
    """Tests for project initialization."""

    def test_init_project_creates_directory_structure(self, temp_project_dir):
        """Test that init_project creates the expected directory structure."""
        init_project(temp_project_dir, "Test Book", "Test Author")

        audiobook_dir = Path(temp_project_dir) / ".audiobook"
        assert audiobook_dir.exists()
        assert (audiobook_dir / "db.sqlite").exists()
        assert (audiobook_dir / "audio" / "segments").exists()
        assert (audiobook_dir / "exports" / "chapters").exists()
        assert (audiobook_dir / "exports" / "book").exists()

    def test_init_project_sets_metadata(self, temp_project_dir):
        """Test that init_project sets project metadata correctly."""
        project = init_project(temp_project_dir, "My Audiobook", "John Doe", "A test audiobook")

        assert project.title == "My Audiobook"
        assert project.author == "John Doe"
        assert project.description == "A test audiobook"

    def test_init_project_fails_on_existing_project(self, initialized_project):
        """Test that init_project fails if project already exists."""
        with pytest.raises(ValueError, match="already initialized"):
            init_project(initialized_project["path"], "Another Book", "Another Author")


class TestProjectInfo:
    """Tests for getting project info."""

    def test_get_project_info_returns_stats(self, project_with_chapters):
        """Test that get_project_info returns statistics."""
        info = get_project_info()

        assert info.project.title == "Test Project"
        assert info.stats.chapter_count == 2
        assert info.stats.segment_count == 3
        assert info.stats.character_count == 3


class TestProjectUpdate:
    """Tests for updating project metadata."""

    def test_update_project_title(self, initialized_project):
        """Test updating project title."""
        project = update_project(title="New Title")
        assert project.title == "New Title"

    def test_update_project_author(self, initialized_project):
        """Test updating project author."""
        project = update_project(author="New Author")
        assert project.author == "New Author"

    def test_update_project_description(self, initialized_project):
        """Test updating project description."""
        project = update_project(description="New description")
        assert project.description == "New description"
