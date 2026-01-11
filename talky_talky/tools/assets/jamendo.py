"""Jamendo API integration for music.

Jamendo is a platform for independent music under Creative Commons licenses.
API documentation: https://developer.jamendo.com/v3.0

To get a client ID:
1. Create a free account at https://www.jamendo.com
2. Register your app at https://developer.jamendo.com/v3.0
3. Use the client_id from your app registration

Note: The API is free for non-commercial use. For commercial use, contact Jamendo.
"""

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

# Jamendo API base URL
API_BASE = "https://api.jamendo.com/v3.0"

# Test client ID for development (limited use only)
TEST_CLIENT_ID = "709fa152"

# License mapping from Jamendo CC URLs to our types
# Jamendo licenses are based on CC license URLs
LICENSE_MAP = {
    "by": LicenseType.CC_BY,
    "by-sa": LicenseType.CC_BY_SA,
    "by-nc": LicenseType.CC_BY_NC,
    "by-nc-sa": LicenseType.CC_BY_NC_SA,
    "by-nd": LicenseType.CC_BY_ND,
    "by-nc-nd": LicenseType.CC_BY_NC_ND,
}


def _parse_license_url(license_url: str | None) -> LicenseInfo:
    """Parse Jamendo license URL to LicenseInfo."""
    if not license_url:
        return LicenseInfo(license_type=LicenseType.UNKNOWN)

    # Extract license type from URL like "http://creativecommons.org/licenses/by-nc-sa/3.0/"
    license_type = LicenseType.UNKNOWN
    commercial_use = True
    modification_allowed = True
    attribution_required = True

    url_lower = license_url.lower()

    # Check for specific CC license patterns
    for key, lt in LICENSE_MAP.items():
        if f"/licenses/{key}/" in url_lower or f"/licenses/{key.replace('-', '')}/" in url_lower:
            license_type = lt
            break

    # Set properties based on license type
    if license_type in (LicenseType.CC_BY_NC, LicenseType.CC_BY_NC_SA, LicenseType.CC_BY_NC_ND):
        commercial_use = False

    if license_type in (LicenseType.CC_BY_ND, LicenseType.CC_BY_NC_ND):
        modification_allowed = False

    return LicenseInfo(
        license_type=license_type,
        license_url=license_url,
        attribution=None,  # Set when parsing track
        commercial_use=commercial_use,
        modification_allowed=modification_allowed,
        attribution_required=attribution_required,
    )


def _parse_track(data: dict) -> Asset:
    """Parse Jamendo API track response to Asset."""
    # Parse license
    license_info = _parse_license_url(data.get("license_ccurl"))

    # Build attribution text
    artist_name = data.get("artist_name", "Unknown Artist")
    track_name = data.get("name", "Unknown Track")
    license_info.attribution = f'"{track_name}" by {artist_name} on Jamendo'

    # Extract tags from musicinfo if available
    tags = []
    musicinfo = data.get("musicinfo", {})
    if musicinfo:
        if "tags" in musicinfo:
            # Tags can be nested by category
            tag_data = musicinfo["tags"]
            if isinstance(tag_data, dict):
                for category_tags in tag_data.values():
                    if isinstance(category_tags, list):
                        tags.extend(category_tags)
            elif isinstance(tag_data, list):
                tags.extend(tag_data)

        # Add vocalinstrumental as tag
        if musicinfo.get("vocalinstrumental"):
            tags.append(musicinfo["vocalinstrumental"])

        # Add speed as tag
        if musicinfo.get("speed"):
            tags.append(f"speed:{musicinfo['speed']}")

        # Add gender if present
        if musicinfo.get("gender"):
            tags.append(f"vocal:{musicinfo['gender']}")

    # Get audio URLs
    audio_url = data.get("audio")  # Streaming URL
    audiodownload = data.get("audiodownload")  # Download URL
    audiodownload_allowed = data.get("audiodownload_allowed", False)

    # Use audiodownload if allowed, otherwise audio stream
    download_url = audiodownload if audiodownload_allowed and audiodownload else None
    preview_url = audio_url

    # Get album image for metadata
    album_image = data.get("album_image") or data.get("image")

    return Asset(
        id=f"jamendo:{data['id']}",
        name=track_name,
        asset_type=AssetType.MUSIC,
        source="jamendo",
        duration_ms=int(float(data.get("duration", 0)) * 1000),
        format="mp3",  # Jamendo provides MP3 by default
        sample_rate=0,  # Not provided by API
        channels=2,  # Assume stereo
        file_size_bytes=0,  # Not provided by API
        tags=tags,
        description=data.get("description", ""),
        license=license_info,
        local_path=None,
        remote_url=download_url,
        preview_url=preview_url,
        metadata={
            "jamendo_id": data["id"],
            "artist_id": data.get("artist_id"),
            "artist_name": artist_name,
            "artist_idstr": data.get("artist_idstr"),
            "album_id": data.get("album_id"),
            "album_name": data.get("album_name"),
            "album_image": album_image,
            "releasedate": data.get("releasedate"),
            "position": data.get("position"),
            "prourl": data.get("prourl"),
            "shorturl": data.get("shorturl"),
            "shareurl": data.get("shareurl"),
            "audiodownload_allowed": audiodownload_allowed,
            "musicinfo": musicinfo,
        },
        created_at=data.get("releasedate"),
        updated_at=datetime.now().isoformat(),
    )


class JamendoSource(RemoteAssetSource):
    """Jamendo API integration for Creative Commons music."""

    def __init__(self):
        self._client_id: str | None = None
        # Try to load saved client ID
        self._client_id = get_api_key("jamendo")

    @property
    def source_id(self) -> str:
        return "jamendo"

    @property
    def name(self) -> str:
        return "Jamendo"

    def get_info(self) -> AssetSourceInfo:
        return AssetSourceInfo(
            name=self.name,
            source_id=self.source_id,
            description="Platform for independent Creative Commons music with 500k+ tracks",
            requires_api_key=True,
            api_key_configured=self.is_api_key_configured(),
            is_available=self.is_available(),
            supported_types=[AssetType.MUSIC],
            attribution_required=True,  # All CC licenses except CC0 require attribution
            rate_limit="Free for non-commercial use",
        )

    def is_available(self) -> bool:
        """Check if httpx is available and client ID is configured."""
        try:
            import httpx  # noqa: F401

            return self.is_api_key_configured()
        except ImportError:
            return False

    def configure_api_key(self, api_key: str) -> None:
        """Configure the Jamendo client ID."""
        self._client_id = api_key
        save_api_key("jamendo", api_key)

    def is_api_key_configured(self) -> bool:
        """Check if client ID is configured."""
        if self._client_id:
            return True
        # Try loading from database
        self._client_id = get_api_key("jamendo")
        return self._client_id is not None

    def _get_client_id(self) -> str:
        """Get the configured client ID."""
        if not self._client_id:
            raise ValueError("Jamendo client ID not configured. Use configure_api_key()")
        return self._client_id

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
        """Search Jamendo for music tracks.

        Args:
            query: Search query (searches track name, artist, album, tags)
            asset_type: Ignored (Jamendo only has music)
            tags: Filter by genre/mood tags
            min_duration_ms: Minimum duration
            max_duration_ms: Maximum duration
            page: Page number (1-indexed)
            page_size: Results per page (max 200)
            **kwargs: Jamendo-specific params:
                - order: Sort order (relevance, popularity_total, popularity_week,
                        popularity_month, creationdate, releasedate, duration,
                        artist_name, album_name, track_name)
                - vocalinstrumental: "vocal" or "instrumental"
                - acousticelectric: "acoustic" or "electric"
                - speed: "verylow", "low", "medium", "high", "veryhigh"
                - gender: "male" or "female" (vocalist)
                - ccsa: Include ShareAlike licensed (bool)
                - ccnd: Include NoDerivs licensed (bool)
                - ccnc: Include NonCommercial licensed (bool)
                - audiodlformat: Download format ("mp31", "mp32", "ogg", "flac")

        Returns:
            SearchResult with matching tracks
        """
        import httpx

        client_id = self._get_client_id()

        # Build query params
        params = {
            "client_id": client_id,
            "format": "json",
            "limit": min(page_size, 200),  # Jamendo max is 200
            "offset": (page - 1) * page_size,
            "include": "musicinfo+licenses",  # Get tags and license info
        }

        # Search query
        if query:
            params["search"] = query

        # Duration filter (Jamendo uses seconds)
        if min_duration_ms is not None and max_duration_ms is not None:
            params["durationbetween"] = f"{min_duration_ms // 1000}_{max_duration_ms // 1000}"
        elif min_duration_ms is not None:
            params["durationbetween"] = f"{min_duration_ms // 1000}_9999"
        elif max_duration_ms is not None:
            params["durationbetween"] = f"0_{max_duration_ms // 1000}"

        # Tag filter
        if tags:
            params["tags"] = "+".join(tags)  # AND search

        # Additional Jamendo-specific filters
        for key in [
            "order",
            "vocalinstrumental",
            "acousticelectric",
            "speed",
            "gender",
            "ccsa",
            "ccnd",
            "ccnc",
            "audiodlformat",
        ]:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        url = f"{API_BASE}/tracks/?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        # Check for API errors
        headers = data.get("headers", {})
        if headers.get("status") in ("error", "failed"):
            error_msg = headers.get("error_message", "Unknown error")
            raise ValueError(f"Jamendo API error: {error_msg}")

        # Parse results
        results = data.get("results", [])
        assets = [_parse_track(track) for track in results]

        # Jamendo doesn't return total count in search, estimate from results
        total_count = headers.get("results_count", len(assets))
        # If we got a full page, there might be more
        if len(assets) == page_size:
            total_count = max(total_count, (page * page_size) + 1)

        return SearchResult(
            assets=assets,
            total_count=total_count,
            page=page,
            page_size=page_size,
            source="jamendo",
        )

    async def get_asset(self, asset_id: str) -> Asset | None:
        """Get a specific track by ID."""
        import httpx

        if not self._client_id:
            return None

        # Strip prefix if present
        if asset_id.startswith("jamendo:"):
            asset_id = asset_id[8:]

        # Check cache first
        cached = db_get_asset(f"jamendo:{asset_id}")
        if cached:
            return cached

        client_id = self._get_client_id()

        params = {
            "client_id": client_id,
            "format": "json",
            "id": asset_id,
            "include": "musicinfo+licenses",
        }

        url = f"{API_BASE}/tracks/?{urlencode(params)}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                data = response.json()

            headers = data.get("headers", {})
            if headers.get("status") in ("error", "failed"):
                return None

            results = data.get("results", [])
            if not results:
                return None

            asset = _parse_track(results[0])

            # Cache the result
            save_asset(asset)

            return asset

        except Exception as e:
            print(f"Error fetching Jamendo asset {asset_id}: {e}", file=sys.stderr)
            return None

    async def download(self, asset: Asset, output_path: str) -> str:
        """Download a track.

        Downloads the track in MP3 format. If direct download is not allowed
        by the artist, downloads the streaming version instead.
        """
        import httpx

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prefer download URL if available, otherwise use preview/stream URL
        download_url = asset.remote_url or asset.preview_url

        if not download_url:
            raise ValueError("No download URL available for this asset")

        # Add client_id to URL if not present
        if "client_id" not in download_url:
            separator = "&" if "?" in download_url else "?"
            download_url = f"{download_url}{separator}client_id={self._get_client_id()}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                download_url,
                timeout=120.0,  # Longer timeout for downloads
                follow_redirects=True,
            )
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

        # Update asset with local path
        asset.local_path = str(output_file)
        save_asset(asset)

        return str(output_file)

    async def get_artist_tracks(
        self,
        artist_id: str | int,
        page: int = 1,
        page_size: int = 20,
    ) -> SearchResult:
        """Get all tracks by a specific artist."""
        import httpx

        client_id = self._get_client_id()

        params = {
            "client_id": client_id,
            "format": "json",
            "artist_id": artist_id,
            "limit": min(page_size, 200),
            "offset": (page - 1) * page_size,
            "include": "musicinfo+licenses",
        }

        url = f"{API_BASE}/tracks/?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        assets = [_parse_track(track) for track in results]

        headers = data.get("headers", {})
        total_count = headers.get("results_count", len(assets))

        return SearchResult(
            assets=assets,
            total_count=total_count,
            page=page,
            page_size=page_size,
            source="jamendo",
        )

    async def get_album_tracks(
        self,
        album_id: str | int,
    ) -> SearchResult:
        """Get all tracks from a specific album."""
        import httpx

        client_id = self._get_client_id()

        params = {
            "client_id": client_id,
            "format": "json",
            "album_id": album_id,
            "limit": 200,  # Get all album tracks
            "include": "musicinfo+licenses",
        }

        url = f"{API_BASE}/tracks/?{urlencode(params)}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        assets = [_parse_track(track) for track in results]

        return SearchResult(
            assets=assets,
            total_count=len(assets),
            page=1,
            page_size=len(assets),
            source="jamendo",
        )


# Singleton instance
_jamendo_source: JamendoSource | None = None


def get_jamendo_source() -> JamendoSource:
    """Get the Jamendo source singleton."""
    global _jamendo_source
    if _jamendo_source is None:
        _jamendo_source = JamendoSource()
    return _jamendo_source
