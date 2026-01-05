"""Shared fixtures for talky-talky tests."""

import shutil
import tempfile

import pytest

from talky_talky.db.connection import close_database


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for a test project."""
    temp_dir = tempfile.mkdtemp(prefix="audiobook_test_")
    yield temp_dir
    # Cleanup
    close_database()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def initialized_project(temp_project_dir):
    """Create a temporary initialized project."""
    from talky_talky.tools.projects import init_project

    project = init_project(temp_project_dir, "Test Project", "Test Author", "Test Description")
    yield {"path": temp_project_dir, "project": project}
    close_database()


@pytest.fixture
def project_with_characters(initialized_project):
    """Create a project with some characters."""
    from talky_talky.tools.characters import add_character

    narrator = add_character("Narrator", "The story narrator", is_narrator=True)
    alice = add_character("Alice", "The protagonist")
    bob = add_character("Bob", "Alice's friend")

    yield {
        **initialized_project,
        "characters": {"narrator": narrator, "alice": alice, "bob": bob},
    }


@pytest.fixture
def project_with_chapters(project_with_characters):
    """Create a project with chapters and segments."""
    from talky_talky.tools.chapters import add_chapter
    from talky_talky.tools.segments import add_segment

    chapter1 = add_chapter("Chapter 1: The Beginning")
    chapter2 = add_chapter("Chapter 2: The Middle")

    # Add segments to chapter 1
    seg1 = add_segment(
        chapter1.id,
        "Once upon a time, in a land far away...",
        project_with_characters["characters"]["narrator"].id,
    )
    seg2 = add_segment(
        chapter1.id,
        "Hello, is anyone there?",
        project_with_characters["characters"]["alice"].id,
    )
    seg3 = add_segment(
        chapter1.id,
        "Yes, I'm here!",
        project_with_characters["characters"]["bob"].id,
    )

    yield {
        **project_with_characters,
        "chapters": {"chapter1": chapter1, "chapter2": chapter2},
        "segments": {"seg1": seg1, "seg2": seg2, "seg3": seg3},
    }
