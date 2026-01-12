"""Asset management tools - local indexing, Freesound, Jamendo."""

from typing import Optional

from ...tools.assets import (
    search_assets as search_assets_async,
    get_asset as get_asset_async,
    download_asset as download_asset_async,
    import_folder as import_folder_async,
    list_sources,
    configure_freesound,
    configure_jamendo,
    set_asset_library_path as set_library_path,
    get_asset_library_path as get_library_path,
    add_tags,
    remove_tags,
    list_tags,
    list_indexed_folders as list_indexed_folders_async,
    rescan_folder as rescan_folder_async,
    remove_indexed_folder as remove_indexed_folder_async,
    auto_tag_asset as auto_tag_asset_async,
    get_autotag_capabilities,
)


def register_asset_tools(mcp):
    """Register asset management tools with the MCP server."""

    @mcp.tool()
    def list_asset_sources() -> dict:
        """List available audio asset sources (local, freesound, jamendo)."""
        return list_sources()

    @mcp.tool()
    def search_audio_assets(
        query: str,
        asset_type: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_duration_secs: Optional[float] = None,
        max_duration_secs: Optional[float] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """Search audio assets across sources.

        Args:
            query: Search query (e.g., "explosion", "forest ambience").
            asset_type: Filter by sfx, music, or ambience.
            source: Limit to local or freesound.
            tags: Filter by tags.
        """
        import asyncio

        return asyncio.run(
            search_assets_async(
                query,
                asset_type,
                source,
                tags,
                min_duration_secs,
                max_duration_secs,
                page,
                page_size,
            )
        )

    @mcp.tool()
    def get_audio_asset(asset_id: str) -> dict:
        """Get detailed info about an asset (format: source:id)."""
        import asyncio

        return asyncio.run(get_asset_async(asset_id))

    @mcp.tool()
    def download_audio_asset(
        asset_id: str,
        output_path: Optional[str] = None,
    ) -> dict:
        """Download an asset to local storage."""
        import asyncio

        return asyncio.run(download_asset_async(asset_id, output_path))

    @mcp.tool()
    def import_audio_folder(
        folder_path: str,
        asset_type: Optional[str] = None,
        recursive: bool = True,
    ) -> dict:
        """Import audio files from folder into asset library.

        Args:
            asset_type: Default type (sfx, music, ambience) or auto-detect.
            recursive: Include subdirectories.
        """
        import asyncio

        return asyncio.run(import_folder_async(folder_path, asset_type, recursive))

    @mcp.tool()
    def configure_freesound_api(api_key: str) -> dict:
        """Configure Freesound.org API key.

        Get credentials at https://freesound.org/apiv2/apply
        Use the "Client secret" as api_key.
        """
        return configure_freesound(api_key)

    @mcp.tool()
    def configure_jamendo_api(client_id: str) -> dict:
        """Configure Jamendo API client ID.

        Register at https://developer.jamendo.com/v3.0
        """
        return configure_jamendo(client_id)

    @mcp.tool()
    def set_audio_library_path(path: str) -> dict:
        """Set asset library location (or "default" to reset)."""
        return set_library_path(path)

    @mcp.tool()
    def get_audio_library_path() -> dict:
        """Get current asset library path."""
        return get_library_path()

    @mcp.tool()
    def add_asset_tags(asset_id: str, tags: list[str]) -> dict:
        """Add tags to an asset for organization."""
        return add_tags(asset_id, tags)

    @mcp.tool()
    def remove_asset_tags(asset_id: str, tags: list[str]) -> dict:
        """Remove tags from an asset."""
        return remove_tags(asset_id, tags)

    @mcp.tool()
    def list_all_asset_tags() -> dict:
        """List all tags in library with usage counts."""
        return list_tags()

    @mcp.tool()
    def list_indexed_audio_folders() -> dict:
        """List folders that have been indexed for assets."""
        import asyncio

        return asyncio.run(list_indexed_folders_async())

    @mcp.tool()
    def rescan_audio_folder(folder_path: str) -> dict:
        """Rescan indexed folder for new/modified files."""
        import asyncio

        return asyncio.run(rescan_folder_async(folder_path))

    @mcp.tool()
    def remove_indexed_audio_folder(folder_path: str) -> dict:
        """Remove folder from index (doesn't delete files)."""
        import asyncio

        return asyncio.run(remove_indexed_folder_async(folder_path))

    @mcp.tool()
    def check_autotag_availability() -> dict:
        """Check available AI auto-tagging capabilities."""
        return get_autotag_capabilities()

    @mcp.tool()
    def auto_tag_audio_asset(
        asset_id: str,
        use_transcription: bool = True,
        use_emotion: bool = True,
        use_quality: bool = True,
        transcription_model: str = "base",
        max_keywords: int = 8,
    ) -> dict:
        """Auto-tag asset using AI analysis (transcription, emotion, quality)."""
        import asyncio

        return asyncio.run(
            auto_tag_asset_async(
                asset_id,
                use_transcription,
                use_emotion,
                use_quality,
                transcription_model,
                max_keywords,
            )
        )
