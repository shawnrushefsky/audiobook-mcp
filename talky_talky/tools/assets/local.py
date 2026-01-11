"""Local file system asset source."""

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import (
    Asset,
    AssetSourceInfo,
    AssetType,
    LicenseInfo,
    LicenseType,
    LocalAssetSource,
    SearchResult,
)
from .database import (
    delete_indexed_folder,
    get_indexed_folders,
    save_asset,
    save_indexed_folder,
    search_assets as db_search,
    get_asset as db_get_asset,
)

# Supported audio formats
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".opus",
}

# Keywords for auto-detecting asset type from path
SFX_KEYWORDS = {
    "sfx",
    "sound",
    "effect",
    "fx",
    "foley",
    "impact",
    "whoosh",
    "click",
    "beep",
    "explosion",
    "footstep",
    "ui",
    "interface",
}
MUSIC_KEYWORDS = {
    "music",
    "song",
    "track",
    "soundtrack",
    "ost",
    "bgm",
    "score",
    "melody",
    "loop",
}
AMBIENCE_KEYWORDS = {
    "ambience",
    "ambient",
    "atmosphere",
    "background",
    "environment",
    "nature",
    "room",
    "tone",
}


def _get_audio_info(file_path: Path) -> dict:
    """Get audio file information using mutagen or soundfile."""
    info = {
        "duration_ms": 0,
        "sample_rate": 0,
        "channels": 0,
        "format": file_path.suffix.lower().lstrip("."),
        "file_size_bytes": file_path.stat().st_size,
    }

    # Try mutagen first (handles more formats)
    try:
        import mutagen

        audio = mutagen.File(str(file_path))
        if audio is not None:
            info["duration_ms"] = int((audio.info.length or 0) * 1000)
            info["sample_rate"] = getattr(audio.info, "sample_rate", 0)
            info["channels"] = getattr(audio.info, "channels", 0)
            return info
    except Exception:
        pass

    # Fallback to soundfile for wav/flac
    try:
        import soundfile as sf

        with sf.SoundFile(str(file_path)) as f:
            info["duration_ms"] = int(len(f) / f.samplerate * 1000)
            info["sample_rate"] = f.samplerate
            info["channels"] = f.channels
    except Exception:
        pass

    return info


def _detect_asset_type(file_path: Path) -> AssetType:
    """Detect asset type from file path keywords."""
    path_lower = str(file_path).lower()

    # Check for keywords in path
    for keyword in AMBIENCE_KEYWORDS:
        if keyword in path_lower:
            return AssetType.AMBIENCE

    for keyword in MUSIC_KEYWORDS:
        if keyword in path_lower:
            return AssetType.MUSIC

    for keyword in SFX_KEYWORDS:
        if keyword in path_lower:
            return AssetType.SFX

    # Default to SFX
    return AssetType.SFX


def _extract_tags_from_path(file_path: Path) -> list[str]:
    """Extract potential tags from file path."""
    tags = set()

    # Get filename without extension
    name = file_path.stem.lower()

    # Split on common separators
    parts = []
    for sep in ["_", "-", " ", ".", "(", ")", "[", "]"]:
        name = name.replace(sep, "|")
    parts = [p.strip() for p in name.split("|") if p.strip()]

    # Filter out numbers and very short strings
    for part in parts:
        if len(part) >= 3 and not part.isdigit():
            tags.add(part)

    # Add parent folder names as potential tags (up to 2 levels)
    for parent in list(file_path.parents)[:2]:
        if parent.name and len(parent.name) >= 3:
            tags.add(parent.name.lower())

    return list(tags)


def _generate_local_id(file_path: Path) -> str:
    """Generate a unique ID for a local file."""
    # Use path hash for stability
    path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:12]
    return f"local:{path_hash}"


class LocalSource(LocalAssetSource):
    """Local file system asset source."""

    @property
    def source_id(self) -> str:
        return "local"

    @property
    def name(self) -> str:
        return "Local Files"

    def get_info(self) -> AssetSourceInfo:
        folders = get_indexed_folders()
        return AssetSourceInfo(
            name=self.name,
            source_id=self.source_id,
            description=f"Local audio files from {len(folders)} indexed folder(s)",
            requires_api_key=False,
            api_key_configured=True,
            is_available=True,
            supported_types=[AssetType.SFX, AssetType.MUSIC, AssetType.AMBIENCE],
            attribution_required=False,
        )

    def is_available(self) -> bool:
        return True

    async def search(
        self,
        query: str,
        asset_type: AssetType | None = None,
        tags: list[str] | None = None,
        min_duration_ms: int | None = None,
        max_duration_ms: int | None = None,
        page: int = 1,
        page_size: int = 20,
        **kwargs: Any,
    ) -> SearchResult:
        """Search local assets."""
        return db_search(
            query=query,
            asset_type=asset_type,
            source="local",
            tags=tags,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            local_only=True,
            page=page,
            page_size=page_size,
        )

    async def get_asset(self, asset_id: str) -> Asset | None:
        """Get a local asset by ID."""
        full_id = asset_id if asset_id.startswith("local:") else f"local:{asset_id}"
        return db_get_asset(full_id)

    async def download(self, asset: Asset, output_path: str) -> str:
        """For local assets, just return the existing path."""
        if asset.local_path:
            return asset.local_path
        raise ValueError("Asset has no local path")

    async def import_folder(
        self,
        folder_path: str,
        asset_type: AssetType | None = None,
        recursive: bool = True,
        auto_tag: bool = False,
    ) -> int:
        """Import assets from a folder.

        Args:
            folder_path: Path to folder to import
            asset_type: Default asset type (auto-detected if None)
            recursive: Scan subdirectories
            auto_tag: Use AI for additional tagging

        Returns:
            Number of assets imported
        """
        folder = Path(folder_path).resolve()

        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")

        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        print(f"Scanning folder: {folder}", file=sys.stderr)

        # Find all audio files
        pattern = "**/*" if recursive else "*"
        audio_files = []

        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(folder.glob(f"{pattern}{ext}"))
            audio_files.extend(folder.glob(f"{pattern}{ext.upper()}"))

        # Remove duplicates
        audio_files = list(set(audio_files))
        print(f"Found {len(audio_files)} audio files", file=sys.stderr)

        imported = 0
        for i, file_path in enumerate(audio_files):
            try:
                # Get audio info
                audio_info = _get_audio_info(file_path)

                # Determine asset type
                file_type = asset_type or _detect_asset_type(file_path)

                # Extract tags from path
                tags = _extract_tags_from_path(file_path)

                # Create asset
                asset = Asset(
                    id=_generate_local_id(file_path),
                    name=file_path.stem,
                    asset_type=file_type,
                    source="local",
                    duration_ms=audio_info["duration_ms"],
                    format=audio_info["format"],
                    sample_rate=audio_info["sample_rate"],
                    channels=audio_info["channels"],
                    file_size_bytes=audio_info["file_size_bytes"],
                    tags=tags,
                    description="",
                    license=LicenseInfo(LicenseType.UNKNOWN),
                    local_path=str(file_path),
                    metadata={
                        "original_path": str(file_path),
                        "folder": str(folder),
                    },
                    created_at=datetime.fromtimestamp(
                        file_path.stat().st_ctime
                    ).isoformat(),
                    updated_at=datetime.now().isoformat(),
                )

                # Save to database
                save_asset(asset)
                imported += 1

                # Progress logging
                if (i + 1) % 100 == 0:
                    print(
                        f"Imported {i + 1}/{len(audio_files)} files...",
                        file=sys.stderr,
                    )

            except Exception as e:
                print(f"Error importing {file_path}: {e}", file=sys.stderr)
                continue

        # Save folder record
        save_indexed_folder(
            path=str(folder),
            asset_type=asset_type,
            recursive=recursive,
            file_count=imported,
        )

        print(f"Successfully imported {imported} assets", file=sys.stderr)

        # Auto-tag if requested (done after import for efficiency)
        if auto_tag and imported > 0:
            print("Auto-tagging will be run separately...", file=sys.stderr)

        return imported

    async def rescan_folder(self, folder_path: str) -> int:
        """Rescan an imported folder for changes."""
        folders = get_indexed_folders()
        folder_info = None

        for f in folders:
            if f["path"] == folder_path:
                folder_info = f
                break

        if not folder_info:
            raise ValueError(f"Folder not indexed: {folder_path}")

        # Re-import with same settings
        asset_type = AssetType(folder_info["asset_type"]) if folder_info["asset_type"] else None
        recursive = bool(folder_info["recursive"])

        return await self.import_folder(
            folder_path=folder_path,
            asset_type=asset_type,
            recursive=recursive,
            auto_tag=False,
        )

    async def list_indexed_folders(self) -> list[dict]:
        """List all indexed folders."""
        return get_indexed_folders()

    async def remove_indexed_folder(self, folder_path: str) -> bool:
        """Remove an indexed folder and its assets."""
        return delete_indexed_folder(folder_path)


# Singleton instance
_local_source: LocalSource | None = None


def get_local_source() -> LocalSource:
    """Get the local source singleton."""
    global _local_source
    if _local_source is None:
        _local_source = LocalSource()
    return _local_source
