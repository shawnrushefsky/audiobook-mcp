"""Base classes and data models for asset management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssetType(str, Enum):
    """Types of audio assets."""

    SFX = "sfx"
    MUSIC = "music"
    AMBIENCE = "ambience"


class LicenseType(str, Enum):
    """Common license types for audio assets."""

    CC0 = "cc0"  # Public domain
    CC_BY = "cc-by"  # Attribution required
    CC_BY_SA = "cc-by-sa"  # Attribution + ShareAlike
    CC_BY_NC = "cc-by-nc"  # Attribution + NonCommercial
    CC_BY_NC_SA = "cc-by-nc-sa"  # Attribution + NC + ShareAlike
    CC_BY_ND = "cc-by-nd"  # Attribution + NoDerivatives
    CC_BY_NC_ND = "cc-by-nc-nd"  # Attribution + NC + ND
    PIXABAY = "pixabay"  # Pixabay license (attribution-free)
    CUSTOM = "custom"  # Custom/other license
    UNKNOWN = "unknown"  # Unknown license


@dataclass
class LicenseInfo:
    """License information for an asset."""

    license_type: LicenseType
    license_url: str | None = None
    attribution: str | None = None  # Required attribution text
    commercial_use: bool = True
    modification_allowed: bool = True
    attribution_required: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "license_type": self.license_type.value,
            "license_url": self.license_url,
            "attribution": self.attribution,
            "commercial_use": self.commercial_use,
            "modification_allowed": self.modification_allowed,
            "attribution_required": self.attribution_required,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LicenseInfo":
        """Create from dictionary."""
        return cls(
            license_type=LicenseType(data.get("license_type", "unknown")),
            license_url=data.get("license_url"),
            attribution=data.get("attribution"),
            commercial_use=data.get("commercial_use", True),
            modification_allowed=data.get("modification_allowed", True),
            attribution_required=data.get("attribution_required", False),
        )

    @classmethod
    def cc0(cls) -> "LicenseInfo":
        """Create CC0 (public domain) license."""
        return cls(
            license_type=LicenseType.CC0,
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            commercial_use=True,
            modification_allowed=True,
            attribution_required=False,
        )

    @classmethod
    def pixabay(cls) -> "LicenseInfo":
        """Create Pixabay license."""
        return cls(
            license_type=LicenseType.PIXABAY,
            license_url="https://pixabay.com/service/license-summary/",
            commercial_use=True,
            modification_allowed=True,
            attribution_required=False,
        )


@dataclass
class Asset:
    """Represents an audio asset (sound effect, music, or ambience)."""

    id: str  # Unique ID (format: source:id, e.g., "pixabay:12345")
    name: str  # Display name
    asset_type: AssetType  # sfx, music, ambience
    source: str  # local, pixabay, freesound

    # Audio properties
    duration_ms: int = 0
    format: str = ""  # wav, mp3, ogg, flac
    sample_rate: int = 0
    channels: int = 0
    file_size_bytes: int = 0

    # Tags and search
    tags: list[str] = field(default_factory=list)
    description: str = ""

    # License
    license: LicenseInfo = field(default_factory=lambda: LicenseInfo(LicenseType.UNKNOWN))

    # Paths/URLs
    local_path: str | None = None  # Local file path (if downloaded or local)
    remote_url: str | None = None  # Full quality download URL
    preview_url: str | None = None  # Preview/sample URL

    # Source-specific metadata
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def is_local(self) -> bool:
        """Check if asset is available locally."""
        return self.local_path is not None

    @property
    def is_remote(self) -> bool:
        """Check if asset is from a remote source."""
        return self.source in ("pixabay", "freesound", "jamendo")

    @property
    def duration_secs(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "source": self.source,
            "duration_ms": self.duration_ms,
            "duration_secs": self.duration_secs,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "file_size_bytes": self.file_size_bytes,
            "tags": self.tags,
            "description": self.description,
            "license": self.license.to_dict(),
            "local_path": self.local_path,
            "remote_url": self.remote_url,
            "preview_url": self.preview_url,
            "is_local": self.is_local,
            "is_remote": self.is_remote,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Asset":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            asset_type=AssetType(data["asset_type"]),
            source=data["source"],
            duration_ms=data.get("duration_ms", 0),
            format=data.get("format", ""),
            sample_rate=data.get("sample_rate", 0),
            channels=data.get("channels", 0),
            file_size_bytes=data.get("file_size_bytes", 0),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            license=LicenseInfo.from_dict(data.get("license", {})),
            local_path=data.get("local_path"),
            remote_url=data.get("remote_url"),
            preview_url=data.get("preview_url"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class SearchResult:
    """Result from an asset search."""

    assets: list[Asset]
    total_count: int  # Total matching (may be more than returned)
    page: int = 1
    page_size: int = 20
    source: str | None = None  # Which source this came from (None = unified)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "assets": [a.to_dict() for a in self.assets],
            "total_count": self.total_count,
            "page": self.page,
            "page_size": self.page_size,
            "source": self.source,
            "has_more": self.total_count > self.page * self.page_size,
        }


@dataclass
class AssetSourceInfo:
    """Information about an asset source."""

    name: str
    source_id: str
    description: str
    requires_api_key: bool = False
    api_key_configured: bool = False
    is_available: bool = True
    supported_types: list[AssetType] = field(default_factory=list)
    attribution_required: bool = False
    rate_limit: str | None = None  # e.g., "100 requests/day"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source_id": self.source_id,
            "description": self.description,
            "requires_api_key": self.requires_api_key,
            "api_key_configured": self.api_key_configured,
            "is_available": self.is_available,
            "supported_types": [t.value for t in self.supported_types],
            "attribution_required": self.attribution_required,
            "rate_limit": self.rate_limit,
        }


class AssetSource(ABC):
    """Abstract base class for asset sources."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        pass

    @abstractmethod
    def get_info(self) -> AssetSourceInfo:
        """Get information about this source."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is available."""
        pass

    @abstractmethod
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
        """Search for assets.

        Args:
            query: Search query string
            asset_type: Filter by asset type
            tags: Filter by tags
            min_duration_ms: Minimum duration in milliseconds
            max_duration_ms: Maximum duration in milliseconds
            page: Page number (1-indexed)
            page_size: Results per page
            **kwargs: Source-specific parameters

        Returns:
            SearchResult with matching assets
        """
        pass

    @abstractmethod
    async def get_asset(self, asset_id: str) -> Asset | None:
        """Get a specific asset by ID.

        Args:
            asset_id: The asset ID (without source prefix)

        Returns:
            Asset if found, None otherwise
        """
        pass

    @abstractmethod
    async def download(self, asset: Asset, output_path: str) -> str:
        """Download an asset to local storage.

        Args:
            asset: The asset to download
            output_path: Where to save the file

        Returns:
            Local path to downloaded file

        Raises:
            Exception if download fails
        """
        pass


class LocalAssetSource(AssetSource):
    """Base class for local asset sources (file-based)."""

    @abstractmethod
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
            asset_type: Default asset type for imported files
            recursive: Whether to scan subdirectories
            auto_tag: Whether to auto-tag using AI

        Returns:
            Number of assets imported
        """
        pass

    @abstractmethod
    async def rescan_folder(self, folder_path: str) -> int:
        """Rescan an imported folder for changes.

        Args:
            folder_path: Path to folder to rescan

        Returns:
            Number of assets added/updated
        """
        pass

    @abstractmethod
    async def list_indexed_folders(self) -> list[dict]:
        """List all indexed folders.

        Returns:
            List of folder info dicts
        """
        pass


class RemoteAssetSource(AssetSource):
    """Base class for remote asset sources (API-based)."""

    @abstractmethod
    def configure_api_key(self, api_key: str) -> None:
        """Configure API key for this source.

        Args:
            api_key: The API key
        """
        pass

    @abstractmethod
    def is_api_key_configured(self) -> bool:
        """Check if API key is configured."""
        pass
