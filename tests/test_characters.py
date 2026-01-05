"""Tests for character management."""

import json

import pytest

from talky_talky.tools.characters import (
    add_character,
    get_character,
    get_characters_with_stats,
    update_character,
    delete_character,
    set_voice,
    clear_voice,
)


class TestCharacterCreation:
    """Tests for creating characters."""

    def test_add_character(self, initialized_project):
        """Test adding a basic character."""
        character = add_character("Hero", "The main protagonist")

        assert character.name == "Hero"
        assert character.description == "The main protagonist"
        assert character.is_narrator is False
        assert character.voice_config is None

    def test_add_narrator_character(self, initialized_project):
        """Test adding a narrator character."""
        character = add_character("Narrator", "Story narrator", is_narrator=True)

        assert character.name == "Narrator"
        assert character.is_narrator is True

    def test_add_duplicate_character_fails(self, initialized_project):
        """Test that adding a duplicate character fails."""
        add_character("Unique Name", "First character")

        with pytest.raises(Exception):  # IntegrityError wrapped
            add_character("Unique Name", "Duplicate")


class TestCharacterRetrieval:
    """Tests for retrieving characters."""

    def test_get_character_by_id(self, project_with_characters):
        """Test getting a character by ID."""
        alice_id = project_with_characters["characters"]["alice"].id
        character = get_character(alice_id)

        assert character is not None
        assert character.name == "Alice"

    def test_get_nonexistent_character_returns_none(self, initialized_project):
        """Test getting a nonexistent character."""
        character = get_character("nonexistent-id")
        assert character is None

    def test_get_characters_with_stats(self, project_with_chapters):
        """Test listing characters with segment counts."""
        characters = get_characters_with_stats()

        assert len(characters) == 3
        # Each character should have segment_count attribute
        for char in characters:
            assert hasattr(char, "segment_count")


class TestCharacterUpdate:
    """Tests for updating characters."""

    def test_update_character_name(self, project_with_characters):
        """Test updating character name."""
        alice_id = project_with_characters["characters"]["alice"].id
        character = update_character(alice_id, name="Alicia")

        assert character.name == "Alicia"

    def test_update_character_description(self, project_with_characters):
        """Test updating character description."""
        alice_id = project_with_characters["characters"]["alice"].id
        character = update_character(alice_id, description="New description")

        assert character.description == "New description"


class TestCharacterDeletion:
    """Tests for deleting characters."""

    def test_delete_character(self, project_with_characters):
        """Test deleting a character."""
        alice_id = project_with_characters["characters"]["alice"].id
        delete_character(alice_id)

        character = get_character(alice_id)
        assert character is None

    def test_delete_nonexistent_character_fails(self, initialized_project):
        """Test deleting a nonexistent character."""
        with pytest.raises(ValueError, match="not found"):
            delete_character("nonexistent-id")


class TestVoiceConfiguration:
    """Tests for voice configuration."""

    def test_set_voice_maya1(self, project_with_characters):
        """Test setting Maya1 voice configuration."""
        alice_id = project_with_characters["characters"]["alice"].id
        character = set_voice(
            alice_id,
            provider="maya1",
            voice_ref="Realistic female voice in the 20s age with american accent.",
        )

        assert character.voice_config is not None
        config = json.loads(character.voice_config)
        assert config["provider"] == "maya1"
        assert "20s" in config["voice_ref"]

    def test_set_voice_chatterbox(self, project_with_characters):
        """Test setting Chatterbox voice configuration."""
        bob_id = project_with_characters["characters"]["bob"].id
        character = set_voice(
            bob_id,
            provider="chatterbox",
            voice_ref="bob-voice",
            settings={"exaggeration": 0.7, "cfg_weight": 0.4},
        )

        assert character.voice_config is not None
        config = json.loads(character.voice_config)
        assert config["provider"] == "chatterbox"
        assert config["settings"]["exaggeration"] == 0.7

    def test_clear_voice(self, project_with_characters):
        """Test clearing voice configuration."""
        alice_id = project_with_characters["characters"]["alice"].id

        # First set a voice
        set_voice(alice_id, provider="maya1", voice_ref="test voice")

        # Then clear it
        character = clear_voice(alice_id)
        assert character.voice_config is None
