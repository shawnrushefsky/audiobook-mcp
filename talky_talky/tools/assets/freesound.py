"""Freesound.org API integration for sound effects.

Freesound is a collaborative database of Creative Commons licensed sounds.
API documentation: https://freesound.org/docs/api/

To get an API key (token):
1. Create a free account at https://freesound.org
2. Apply for API credentials at https://freesound.org/apiv2/apply
3. Use the "Client secret/Api key" as your API token (NOT the "Client id")

Note: Freesound uses "Token Authentication" where the "Client secret" serves
as the API token. The "Client id" is used for OAuth2 flows which are not
needed for basic API access.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .base import (
    Asset,
    AssetSourceInfo,
    AssetType,
    LicenseInfo,
    LicenseType,
    RemoteAssetSource,
    SearchResult,
)
from .database import get_api_key, save_api_key, save_asset, get_asset as db_get_asset

# Freesound API base URL
API_BASE = "https://freesound.org/apiv2"

# License mapping from Freesound to our types
LICENSE_MAP = {
    "Attribution": LicenseType.CC_BY,
    "Attribution Noncommercial": LicenseType.CC_BY_NC,
    "Creative Commons 0": LicenseType.CC0,
    "Attribution NonCommercial": LicenseType.CC_BY_NC,
    "Sampling+": LicenseType.CUSTOM,
}


def _parse_license(license_str: str, license_url: str | None = None) -> LicenseInfo:
    """Parse Freesound license string to LicenseInfo."""
    license_type = LICENSE_MAP.get(license_str, LicenseType.CUSTOM)

    attribution_required = license_type in (
        LicenseType.CC_BY,
        LicenseType.CC_BY_NC,
        LicenseType.CC_BY_SA,
        LicenseType.CC_BY_NC_SA,
    )

    commercial_use = license_type not in (
        LicenseType.CC_BY_NC,
        LicenseType.CC_BY_NC_SA,
        LicenseType.CC_BY_NC_ND,
    )

    return LicenseInfo(
        license_type=license_type,
        license_url=license_url,
        attribution=None,  # Filled when downloading
        commercial_use=commercial_use,
        modification_allowed=True,
        attribution_required=attribution_required,
    )


def _parse_sound(data: dict) -> Asset:
    """Parse Freesound API response to Asset."""
    # Extract preview URLs
    previews = data.get("previews", {})
    preview_url = (
        previews.get("preview-hq-mp3")
        or previews.get("preview-lq-mp3")
        or previews.get("preview-hq-ogg")
    )

    # Parse license
    license_info = _parse_license(
        data.get("license", ""),
        data.get("license_url"),
    )

    # Set attribution text
    username = data.get("username", "Unknown")
    license_info.attribution = f'"{data.get("name", "Sound")}" by {username} on Freesound.org'

    # Extract tags
    tags = data.get("tags", [])

    # Determine asset type from tags/description
    asset_type = AssetType.SFX  # Default
    tag_lower = [t.lower() for t in tags]
    desc_lower = (data.get("description") or "").lower()

    if any(t in tag_lower for t in ["music", "song", "melody", "loop", "beat"]):
        asset_type = AssetType.MUSIC
    elif any(t in tag_lower for t in ["ambience", "ambient", "atmosphere", "background"]):
        asset_type = AssetType.AMBIENCE

    return Asset(
        id=f"freesound:{data['id']}",
        name=data.get("name", f"Sound {data['id']}"),
        asset_type=asset_type,
        source="freesound",
        duration_ms=int((data.get("duration", 0)) * 1000),
        format=data.get("type", ""),
        sample_rate=data.get("samplerate", 0),
        channels=data.get("channels", 0),
        file_size_bytes=data.get("filesize", 0),
        tags=tags,
        description=data.get("description", ""),
        license=license_info,
        local_path=None,
        remote_url=data.get("download"),  # Requires OAuth2
        preview_url=preview_url,
        metadata={
            "freesound_id": data["id"],
            "username": username,
            "created": data.get("created"),
            "num_downloads": data.get("num_downloads", 0),
            "avg_rating": data.get("avg_rating", 0),
            "num_ratings": data.get("num_ratings", 0),
            "bitrate": data.get("bitrate", 0),
            "bitdepth": data.get("bitdepth", 0),
            "pack": data.get("pack"),
            "geotag": data.get("geotag"),
        },
        created_at=data.get("created"),
        updated_at=datetime.now().isoformat(),
    )


class FreesoundSource(RemoteAssetSource):
    """Freesound.org API integration."""

    def __init__(self):
        self._api_key: str | None = None
        # Try to load saved API key
        self._api_key = get_api_key("freesound")

    @property
    def source_id(self) -> str:
        return "freesound"

    @property
    def name(self) -> str:
        return "Freesound"

    def get_info(self) -> AssetSourceInfo:
        return AssetSourceInfo(
            name=self.name,
            source_id=self.source_id,
            description="Collaborative database of Creative Commons licensed sounds",
            requires_api_key=True,
            api_key_configured=self.is_api_key_configured(),
            is_available=self.is_available(),
            supported_types=[AssetType.SFX, AssetType.MUSIC, AssetType.AMBIENCE],
            attribution_required=True,  # Most CC licenses require attribution
            rate_limit="Reasonable use, no specific limit documented",
        )

    def is_available(self) -> bool:
        """Check if httpx is available and API key is configured."""
        try:
            import httpx  # noqa: F401
            return self.is_api_key_configured()
        except ImportError:
            return False

    def configure_api_key(self, api_key: str) -> None:
        """Configure the Freesound API key."""
        self._api_key = api_key
        save_api_key("freesound", api_key)

    def is_api_key_configured(self) -> bool:
        """Check if API key is configured."""
        if self._api_key:
            return True
        # Try loading from database
        self._api_key = get_api_key("freesound")
        return self._api_key is not None

    def _get_headers(self) -> dict:
        """Get authorization headers."""
        if not self._api_key:
            raise ValueError("Freesound API key not configured")
        return {"Authorization": f"Token {self._api_key}"}

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
        """Search Freesound for sounds.

        Args:
            query: Search query
            asset_type: Filter by type (adds tag filter)
            tags: Additional tag filters
            min_duration_ms: Minimum duration
            max_duration_ms: Maximum duration
            page: Page number (1-indexed)
            page_size: Results per page (max 150)
            **kwargs: Additional Freesound-specific params:
                - sort: Sort order (score, duration_desc, duration_asc,
                        created_desc, created_asc, downloads_desc,
                        downloads_asc, rating_desc, rating_asc)
                - license: Filter by license

        Returns:
            SearchResult with matching sounds
        """
        import httpx

        if not self._api_key:
            raise ValueError("Freesound API key not configured. Use configure_api_key()")

        # Build filter string
        filters = []

        # Duration filter (Freesound uses seconds)
        if min_duration_ms is not None:
            filters.append(f"duration:[{min_duration_ms / 1000} TO *]")
        if max_duration_ms is not None:
            filters.append(f"duration:[* TO {max_duration_ms / 1000}]")

        # Tag filters
        all_tags = list(tags or [])
        if asset_type == AssetType.MUSIC:
            all_tags.extend(["music", "loop", "melody"])
        elif asset_type == AssetType.AMBIENCE:
            all_tags.extend(["ambience", "ambient", "atmosphere"])

        if all_tags:
            # Use OR for type-based tags
            tag_filter = " OR ".join(f'tag:"{t}"' for t in all_tags)
            filters.append(f"({tag_filter})")

        # Build query params
        params = {
            "query": query,
            "page": page,
            "page_size": min(page_size, 150),  # Freesound max is 150
            "fields": "id,name,tags,description,duration,license,license_url,"
                      "previews,download,username,created,num_downloads,"
                      "avg_rating,num_ratings,type,samplerate,channels,"
                      "filesize,bitrate,bitdepth,pack,geotag",
        }

        if filters:
            params["filter"] = " AND ".join(filters)

        if "sort" in kwargs:
            params["sort"] = kwargs["sort"]

        if "license" in kwargs:
            params["license"] = kwargs["license"]

        url = f"{API_BASE}/search/text/?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers(), timeout=30.0)
            response.raise_for_status()
            data = response.json()

        # Parse results
        assets = [_parse_sound(sound) for sound in data.get("results", [])]

        return SearchResult(
            assets=assets,
            total_count=data.get("count", 0),
            page=page,
            page_size=page_size,
            source="freesound",
        )

    async def get_asset(self, asset_id: str) -> Asset | None:
        """Get a specific sound by ID."""
        import httpx

        if not self._api_key:
            return None

        # Strip prefix if present
        if asset_id.startswith("freesound:"):
            asset_id = asset_id[10:]

        # Check cache first
        cached = db_get_asset(f"freesound:{asset_id}")
        if cached:
            return cached

        url = f"{API_BASE}/sounds/{asset_id}/"
        params = {
            "fields": "id,name,tags,description,duration,license,license_url,"
                      "previews,download,username,created,num_downloads,"
                      "avg_rating,num_ratings,type,samplerate,channels,"
                      "filesize,bitrate,bitdepth,pack,geotag",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{url}?{urlencode(params)}",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

            asset = _parse_sound(data)

            # Cache the result
            save_asset(asset)

            return asset

        except Exception as e:
            print(f"Error fetching Freesound asset {asset_id}: {e}", file=sys.stderr)
            return None

    async def download(self, asset: Asset, output_path: str) -> str:
        """Download a sound's preview (full download requires OAuth2).

        Note: Full quality downloads require OAuth2 authentication.
        This method downloads the high-quality preview instead.
        """
        import httpx

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Use preview URL (doesn't require OAuth2)
        if not asset.preview_url:
            raise ValueError("No preview URL available for this asset")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                asset.preview_url,
                headers=self._get_headers(),
                timeout=60.0,
                follow_redirects=True,
            )
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

        # Update asset with local path
        asset.local_path = str(output_file)
        save_asset(asset)

        return str(output_file)

    async def get_similar(
        self,
        sound_id: str,
        page: int = 1,
        page_size: int = 15,
    ) -> SearchResult:
        """Get acoustically similar sounds."""
        import httpx

        if not self._api_key:
            raise ValueError("Freesound API key not configured")

        # Strip prefix if present
        if sound_id.startswith("freesound:"):
            sound_id = sound_id[10:]

        params = {
            "page": page,
            "page_size": min(page_size, 150),
            "fields": "id,name,tags,description,duration,license,license_url,"
                      "previews,download,username,created,type,samplerate,"
                      "channels,filesize",
        }

        url = f"{API_BASE}/sounds/{sound_id}/similar/?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers(), timeout=30.0)
            response.raise_for_status()
            data = response.json()

        assets = [_parse_sound(sound) for sound in data.get("results", [])]

        return SearchResult(
            assets=assets,
            total_count=data.get("count", 0),
            page=page,
            page_size=page_size,
            source="freesound",
        )


# Singleton instance
_freesound_source: FreesoundSource | None = None


def get_freesound_source() -> FreesoundSource:
    """Get the Freesound source singleton."""
    global _freesound_source
    if _freesound_source is None:
        _freesound_source = FreesoundSource()
    return _freesound_source
