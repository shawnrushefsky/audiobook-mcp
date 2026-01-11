"""Audio asset management - search, index, and download sound effects, music, and ambience.

This module provides unified access to audio assets from multiple sources:
- Local files: Index and search your own audio collections
- Freesound: Search and download Creative Commons licensed sounds

Example usage:
    from talky_talky.tools.assets import (
        search_assets,
        import_folder,
        configure_freesound,
        download_asset,
    )

    # Import a local folder
    count = await import_folder("/path/to/sounds", asset_type="sfx")

    # Search across all sources
    results = await search_assets("explosion", asset_type="sfx")

    # Configure Freesound API
    configure_freesound("your-api-key")

    # Search Freesound
    results = await search_assets("thunder", source="freesound")

    # Download an asset
    path = await download_asset(results.assets[0], "/path/to/output.mp3")
"""

import asyncio
from pathlib import Path
from typing import Any

from .base import (
    Asset,
    AssetSource,
    AssetSourceInfo,
    AssetType,
    LicenseInfo,
    LicenseType,
    LocalAssetSource,
    RemoteAssetSource,
    SearchResult,
)
from .database import (
    get_database_path,
    set_database_path,
    get_asset as db_get_asset,
    add_tags_to_asset,
    remove_tags_from_asset,
    get_all_tags,
)
from .local import get_local_source, LocalSource
from .freesound import get_freesound_source, FreesoundSource
from .jamendo import get_jamendo_source, JamendoSource
from .autotag import (
    auto_tag_audio,
    get_autotag_capabilities,
    AutoTagResult,
)

__all__ = [
    # Data models
    "Asset",
    "AssetType",
    "AssetSource",
    "AssetSourceInfo",
    "LicenseInfo",
    "LicenseType",
    "LocalAssetSource",
    "RemoteAssetSource",
    "SearchResult",
    # Sources
    "LocalSource",
    "FreesoundSource",
    "JamendoSource",
    "get_local_source",
    "get_freesound_source",
    "get_jamendo_source",
    # High-level API
    "search_assets",
    "get_asset",
    "download_asset",
    "import_folder",
    "list_sources",
    "configure_freesound",
    "configure_jamendo",
    "set_asset_library_path",
    "get_asset_library_path",
    # Tag management
    "add_tags",
    "remove_tags",
    "list_tags",
    # Folder management
    "list_indexed_folders",
    "rescan_folder",
    "remove_indexed_folder",
    # Auto-tagging
    "auto_tag_audio",
    "auto_tag_asset",
    "get_autotag_capabilities",
    "AutoTagResult",
]


# Source registry
_sources: dict[str, AssetSource] = {}


def _init_sources() -> None:
    """Initialize source registry."""
    global _sources
    if not _sources:
        _sources["local"] = get_local_source()
        _sources["freesound"] = get_freesound_source()
        _sources["jamendo"] = get_jamendo_source()


def list_sources() -> dict[str, AssetSourceInfo]:
    """List all available asset sources.

    Returns:
        Dict mapping source_id to AssetSourceInfo
    """
    _init_sources()
    return {source_id: source.get_info() for source_id, source in _sources.items()}


def get_source(source_id: str) -> AssetSource | None:
    """Get a specific source by ID."""
    _init_sources()
    return _sources.get(source_id)


# Configuration


def set_asset_library_path(path: str) -> dict:
    """Set the asset library path (where database and downloads are stored).

    Args:
        path: Directory path, or "default" to reset

    Returns:
        Dict with configured path and status
    """
    db_path = set_database_path(
        path if path == "default" else str(Path(path) / "assets.db")
    )

    return {
        "status": "success",
        "database_path": str(db_path),
        "library_path": str(db_path.parent),
    }


def get_asset_library_path() -> dict:
    """Get the current asset library path.

    Returns:
        Dict with current path configuration
    """
    db_path = get_database_path()
    return {
        "database_path": str(db_path),
        "library_path": str(db_path.parent),
        "exists": db_path.exists(),
    }


def configure_freesound(api_key: str) -> dict:
    """Configure Freesound API key.

    To get an API key:
    1. Create account at https://freesound.org
    2. Apply at https://freesound.org/apiv2/apply

    Args:
        api_key: Your Freesound API key

    Returns:
        Dict with configuration status
    """
    source = get_freesound_source()
    source.configure_api_key(api_key)

    return {
        "status": "success",
        "source": "freesound",
        "configured": source.is_api_key_configured(),
    }


def configure_jamendo(client_id: str) -> dict:
    """Configure Jamendo client ID.

    To get a client ID:
    1. Create account at https://www.jamendo.com
    2. Register app at https://developer.jamendo.com/v3.0

    Note: Free for non-commercial use. Contact Jamendo for commercial licensing.

    Args:
        client_id: Your Jamendo client ID

    Returns:
        Dict with configuration status
    """
    source = get_jamendo_source()
    source.configure_api_key(client_id)

    return {
        "status": "success",
        "source": "jamendo",
        "configured": source.is_api_key_configured(),
    }


# Search and retrieval


async def search_assets(
    query: str,
    asset_type: AssetType | str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    min_duration_ms: int | None = None,
    max_duration_ms: int | None = None,
    min_duration_secs: float | None = None,
    max_duration_secs: float | None = None,
    page: int = 1,
    page_size: int = 20,
    **kwargs: Any,
) -> SearchResult:
    """Search for assets across all sources or a specific source.

    Args:
        query: Search query string
        asset_type: Filter by type ("sfx", "music", "ambience")
        source: Limit to specific source ("local", "freesound")
        tags: Filter by tags
        min_duration_ms: Minimum duration in milliseconds
        max_duration_ms: Maximum duration in milliseconds
        min_duration_secs: Minimum duration in seconds (convenience)
        max_duration_secs: Maximum duration in seconds (convenience)
        page: Page number (1-indexed)
        page_size: Results per page
        **kwargs: Source-specific parameters

    Returns:
        SearchResult with matching assets
    """
    _init_sources()

    # Convert string asset_type to enum
    if isinstance(asset_type, str):
        asset_type = AssetType(asset_type.lower())

    # Convert seconds to milliseconds
    if min_duration_secs is not None:
        min_duration_ms = int(min_duration_secs * 1000)
    if max_duration_secs is not None:
        max_duration_ms = int(max_duration_secs * 1000)

    # Search specific source
    if source:
        src = _sources.get(source)
        if not src:
            return SearchResult(
                assets=[],
                total_count=0,
                page=page,
                page_size=page_size,
                source=source,
            )

        if not src.is_available():
            return SearchResult(
                assets=[],
                total_count=0,
                page=page,
                page_size=page_size,
                source=source,
            )

        return await src.search(
            query=query,
            asset_type=asset_type,
            tags=tags,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            page=page,
            page_size=page_size,
            **kwargs,
        )

    # Search all available sources
    all_assets: list[Asset] = []
    total_count = 0

    # Calculate per-source page size for unified results
    per_source_size = max(page_size // len(_sources), 10)

    tasks = []
    available_sources = []

    for source_id, src in _sources.items():
        if src.is_available():
            available_sources.append(source_id)
            tasks.append(
                src.search(
                    query=query,
                    asset_type=asset_type,
                    tags=tags,
                    min_duration_ms=min_duration_ms,
                    max_duration_ms=max_duration_ms,
                    page=page,
                    page_size=per_source_size,
                    **kwargs,
                )
            )

    if not tasks:
        return SearchResult(
            assets=[],
            total_count=0,
            page=page,
            page_size=page_size,
        )

    # Run searches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error searching {available_sources[i]}: {result}")
            continue
        if isinstance(result, SearchResult):
            all_assets.extend(result.assets)
            total_count += result.total_count

    # Sort by relevance (local first, then by name)
    all_assets.sort(key=lambda a: (0 if a.source == "local" else 1, a.name.lower()))

    # Paginate combined results
    start = 0
    end = page_size
    paginated = all_assets[start:end]

    return SearchResult(
        assets=paginated,
        total_count=total_count,
        page=page,
        page_size=page_size,
    )


async def get_asset(asset_id: str) -> Asset | None:
    """Get a specific asset by ID.

    Args:
        asset_id: Asset ID (format: "source:id", e.g., "freesound:12345")

    Returns:
        Asset if found, None otherwise
    """
    # Try database first
    asset = db_get_asset(asset_id)
    if asset:
        return asset

    # Parse source from ID
    if ":" in asset_id:
        source_id, _ = asset_id.split(":", 1)
    else:
        source_id = "local"

    _init_sources()
    source = _sources.get(source_id)

    if source and source.is_available():
        return await source.get_asset(asset_id)

    return None


async def download_asset(
    asset: Asset | str,
    output_path: str | None = None,
) -> dict:
    """Download an asset to local storage.

    Args:
        asset: Asset object or asset ID
        output_path: Output path (auto-generated if not provided)

    Returns:
        Dict with download status and local path
    """
    # Resolve asset if string ID provided
    if isinstance(asset, str):
        resolved = await get_asset(asset)
        if not resolved:
            return {"status": "error", "error": f"Asset not found: {asset}"}
        asset = resolved

    # If already local, just return the path
    if asset.local_path and Path(asset.local_path).exists():
        return {
            "status": "success",
            "local_path": asset.local_path,
            "already_local": True,
        }

    # Generate output path if not provided
    if not output_path:
        library_path = get_database_path().parent
        downloads_dir = library_path / "downloads" / asset.source
        downloads_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        # For Freesound, previews are always MP3 regardless of original format
        if asset.source == "freesound" and asset.preview_url:
            # Extract extension from preview URL (e.g., preview-hq-mp3 -> mp3)
            if "mp3" in asset.preview_url:
                ext = "mp3"
            elif "ogg" in asset.preview_url:
                ext = "ogg"
            else:
                ext = "mp3"  # Default for Freesound previews
        elif asset.source == "jamendo":
            # Jamendo defaults to MP3
            ext = "mp3"
        else:
            # Use original format or default to mp3
            ext = asset.format or "mp3"

        filename = f"{asset.id.replace(':', '_')}.{ext}"
        output_path = str(downloads_dir / filename)

    # Get source and download
    _init_sources()
    source = _sources.get(asset.source)

    if not source:
        return {"status": "error", "error": f"Unknown source: {asset.source}"}

    if not source.is_available():
        return {"status": "error", "error": f"Source not available: {asset.source}"}

    try:
        local_path = await source.download(asset, output_path)
        return {
            "status": "success",
            "local_path": local_path,
            "already_local": False,
            "asset_id": asset.id,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Local folder management


async def import_folder(
    folder_path: str,
    asset_type: AssetType | str | None = None,
    recursive: bool = True,
    auto_tag: bool = False,
) -> dict:
    """Import audio files from a folder.

    Args:
        folder_path: Path to folder to import
        asset_type: Default type ("sfx", "music", "ambience") - auto-detected if None
        recursive: Scan subdirectories
        auto_tag: Use AI for additional tagging (slower)

    Returns:
        Dict with import results
    """
    if isinstance(asset_type, str):
        asset_type = AssetType(asset_type.lower())

    source = get_local_source()

    try:
        count = await source.import_folder(
            folder_path=folder_path,
            asset_type=asset_type,
            recursive=recursive,
            auto_tag=auto_tag,
        )
        return {
            "status": "success",
            "folder": folder_path,
            "assets_imported": count,
            "recursive": recursive,
            "auto_tagged": auto_tag,
        }
    except Exception as e:
        return {
            "status": "error",
            "folder": folder_path,
            "error": str(e),
        }


async def list_indexed_folders() -> list[dict]:
    """List all indexed folders.

    Returns:
        List of folder info dicts
    """
    source = get_local_source()
    return await source.list_indexed_folders()


async def rescan_folder(folder_path: str) -> dict:
    """Rescan an indexed folder for changes.

    Args:
        folder_path: Path to folder to rescan

    Returns:
        Dict with rescan results
    """
    source = get_local_source()

    try:
        count = await source.rescan_folder(folder_path)
        return {
            "status": "success",
            "folder": folder_path,
            "assets_updated": count,
        }
    except Exception as e:
        return {
            "status": "error",
            "folder": folder_path,
            "error": str(e),
        }


async def remove_indexed_folder(folder_path: str) -> dict:
    """Remove an indexed folder and its assets from the database.

    Args:
        folder_path: Path to folder to remove

    Returns:
        Dict with removal status
    """
    source = get_local_source()

    try:
        success = await source.remove_indexed_folder(folder_path)
        return {
            "status": "success" if success else "not_found",
            "folder": folder_path,
        }
    except Exception as e:
        return {
            "status": "error",
            "folder": folder_path,
            "error": str(e),
        }


# Tag management


def add_tags(asset_id: str, tags: list[str], source: str = "manual") -> dict:
    """Add tags to an asset.

    Args:
        asset_id: Asset ID
        tags: Tags to add
        source: Tag source ("manual", "ai", "api")

    Returns:
        Dict with status
    """
    success = add_tags_to_asset(asset_id, tags, source)
    return {
        "status": "success" if success else "not_found",
        "asset_id": asset_id,
        "tags_added": tags,
    }


def remove_tags(asset_id: str, tags: list[str]) -> dict:
    """Remove tags from an asset.

    Args:
        asset_id: Asset ID
        tags: Tags to remove

    Returns:
        Dict with status
    """
    success = remove_tags_from_asset(asset_id, tags)
    return {
        "status": "success" if success else "not_found",
        "asset_id": asset_id,
        "tags_removed": tags,
    }


def list_tags() -> list[dict]:
    """List all tags with usage counts.

    Returns:
        List of tag info dicts with name, source, and count
    """
    return get_all_tags()


# Sync wrappers for non-async contexts


def search_assets_sync(
    query: str,
    **kwargs: Any,
) -> SearchResult:
    """Synchronous wrapper for search_assets."""
    return asyncio.run(search_assets(query, **kwargs))


def get_asset_sync(asset_id: str) -> Asset | None:
    """Synchronous wrapper for get_asset."""
    return asyncio.run(get_asset(asset_id))


def download_asset_sync(asset: Asset | str, output_path: str | None = None) -> dict:
    """Synchronous wrapper for download_asset."""
    return asyncio.run(download_asset(asset, output_path))


def import_folder_sync(folder_path: str, **kwargs: Any) -> dict:
    """Synchronous wrapper for import_folder."""
    return asyncio.run(import_folder(folder_path, **kwargs))


# Auto-tagging


async def auto_tag_asset(
    asset_id: str,
    use_transcription: bool = True,
    use_emotion: bool = True,
    use_quality: bool = True,
    transcription_model: str = "base",
    max_keywords: int = 8,
) -> dict:
    """Auto-tag an asset using AI analysis.

    Generates semantic tags by analyzing the audio:
    - Transcription: Extracts keywords from speech content
    - Emotion: Detects emotional tone (happy, sad, angry, etc.)
    - Quality: Assesses audio quality level

    Args:
        asset_id: Asset ID to tag
        use_transcription: Extract keywords from speech
        use_emotion: Detect and tag emotions
        use_quality: Assess and tag quality level
        transcription_model: Whisper model size ("tiny", "base", "small", "medium", "large-v3")
        max_keywords: Maximum keywords to extract from transcription

    Returns:
        Dict with generated tags and analysis details
    """
    # Get asset
    asset = await get_asset(asset_id)
    if not asset:
        return {"status": "error", "error": f"Asset not found: {asset_id}"}

    if not asset.local_path:
        return {"status": "error", "error": f"Asset has no local path: {asset_id}"}

    # Run auto-tagging
    result = auto_tag_audio(
        audio_path=asset.local_path,
        use_transcription=use_transcription,
        use_emotion=use_emotion,
        use_quality=use_quality,
        transcription_model=transcription_model,
        max_keywords=max_keywords,
    )

    if result.status != "success":
        return result.to_dict()

    # Save tags to database
    if result.tags:
        for tag in result.tags:
            source = result.tag_sources.get(tag, "ai")
            add_tags_to_asset(asset_id, [tag], source)

    return {
        "status": "success",
        "asset_id": asset_id,
        "tags_added": result.tags,
        "tag_sources": result.tag_sources,
        "transcription": result.transcription,
        "emotion": result.emotion,
        "emotion_confidence": result.emotion_confidence,
        "quality_score": result.quality_score,
        "processing_time_ms": result.processing_time_ms,
    }
