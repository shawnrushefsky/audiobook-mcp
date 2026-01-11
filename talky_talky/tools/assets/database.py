"""SQLite database for asset indexing and caching."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from .base import Asset, AssetType, LicenseInfo, LicenseType, SearchResult

# Default database location
DEFAULT_DB_DIR = Path.home() / "Documents" / "talky-talky" / "assets"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "assets.db"

# Global database path (can be changed via set_database_path)
_db_path: Path = DEFAULT_DB_PATH


def get_database_path() -> Path:
    """Get the current database path."""
    return _db_path


def set_database_path(path: str | Path) -> Path:
    """Set the database path.

    Args:
        path: New database path. Use "default" to reset.

    Returns:
        The configured path.
    """
    global _db_path

    if path == "default":
        _db_path = DEFAULT_DB_PATH
    else:
        _db_path = Path(path)

    # Ensure parent directory exists
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    return _db_path


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with proper settings."""
    # Ensure directory exists
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.executescript(
            """
            -- Assets table
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                source TEXT NOT NULL,
                duration_ms INTEGER DEFAULT 0,
                format TEXT,
                sample_rate INTEGER DEFAULT 0,
                channels INTEGER DEFAULT 0,
                file_size_bytes INTEGER DEFAULT 0,
                description TEXT,
                license_type TEXT,
                license_url TEXT,
                attribution TEXT,
                commercial_use INTEGER DEFAULT 1,
                modification_allowed INTEGER DEFAULT 1,
                attribution_required INTEGER DEFAULT 0,
                local_path TEXT,
                remote_url TEXT,
                preview_url TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Tags table
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL COLLATE NOCASE,
                source TEXT DEFAULT 'manual'  -- manual, ai, api
            );

            -- Asset-Tag junction table
            CREATE TABLE IF NOT EXISTS asset_tags (
                asset_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence REAL DEFAULT 1.0,
                PRIMARY KEY (asset_id, tag_id),
                FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );

            -- Indexed folders table
            CREATE TABLE IF NOT EXISTS indexed_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                asset_type TEXT,
                recursive INTEGER DEFAULT 1,
                last_scanned TIMESTAMP,
                file_count INTEGER DEFAULT 0
            );

            -- API keys table (encrypted storage would be better, but simple for now)
            CREATE TABLE IF NOT EXISTS api_keys (
                source TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Configuration table
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Indexes for better search performance
            CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
            CREATE INDEX IF NOT EXISTS idx_assets_source ON assets(source);
            CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name);
            CREATE INDEX IF NOT EXISTS idx_assets_local_path ON assets(local_path);
            CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
            CREATE INDEX IF NOT EXISTS idx_asset_tags_asset ON asset_tags(asset_id);
            CREATE INDEX IF NOT EXISTS idx_asset_tags_tag ON asset_tags(tag_id);

            -- Full-text search virtual table
            CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
                id,
                name,
                description,
                content='assets',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS assets_ai AFTER INSERT ON assets BEGIN
                INSERT INTO assets_fts(rowid, id, name, description)
                VALUES (new.rowid, new.id, new.name, new.description);
            END;

            CREATE TRIGGER IF NOT EXISTS assets_ad AFTER DELETE ON assets BEGIN
                INSERT INTO assets_fts(assets_fts, rowid, id, name, description)
                VALUES ('delete', old.rowid, old.id, old.name, old.description);
            END;

            CREATE TRIGGER IF NOT EXISTS assets_au AFTER UPDATE ON assets BEGIN
                INSERT INTO assets_fts(assets_fts, rowid, id, name, description)
                VALUES ('delete', old.rowid, old.id, old.name, old.description);
                INSERT INTO assets_fts(rowid, id, name, description)
                VALUES (new.rowid, new.id, new.name, new.description);
            END;
        """
        )


def _row_to_asset(row: sqlite3.Row, tags: list[str] | None = None) -> Asset:
    """Convert a database row to an Asset object."""
    license_info = LicenseInfo(
        license_type=LicenseType(row["license_type"] or "unknown"),
        license_url=row["license_url"],
        attribution=row["attribution"],
        commercial_use=bool(row["commercial_use"]),
        modification_allowed=bool(row["modification_allowed"]),
        attribution_required=bool(row["attribution_required"]),
    )

    metadata = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            pass

    return Asset(
        id=row["id"],
        name=row["name"],
        asset_type=AssetType(row["asset_type"]),
        source=row["source"],
        duration_ms=row["duration_ms"] or 0,
        format=row["format"] or "",
        sample_rate=row["sample_rate"] or 0,
        channels=row["channels"] or 0,
        file_size_bytes=row["file_size_bytes"] or 0,
        tags=tags or [],
        description=row["description"] or "",
        license=license_info,
        local_path=row["local_path"],
        remote_url=row["remote_url"],
        preview_url=row["preview_url"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def get_asset_tags(conn: sqlite3.Connection, asset_id: str) -> list[str]:
    """Get tags for an asset."""
    cursor = conn.execute(
        """
        SELECT t.name FROM tags t
        JOIN asset_tags at ON t.id = at.tag_id
        WHERE at.asset_id = ?
        ORDER BY at.confidence DESC, t.name
        """,
        (asset_id,),
    )
    return [row["name"] for row in cursor.fetchall()]


def save_asset(asset: Asset) -> None:
    """Save or update an asset in the database."""
    with get_connection() as conn:
        # Insert or replace asset
        conn.execute(
            """
            INSERT OR REPLACE INTO assets (
                id, name, asset_type, source, duration_ms, format,
                sample_rate, channels, file_size_bytes, description,
                license_type, license_url, attribution, commercial_use,
                modification_allowed, attribution_required,
                local_path, remote_url, preview_url, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset.id,
                asset.name,
                asset.asset_type.value,
                asset.source,
                asset.duration_ms,
                asset.format,
                asset.sample_rate,
                asset.channels,
                asset.file_size_bytes,
                asset.description,
                asset.license.license_type.value,
                asset.license.license_url,
                asset.license.attribution,
                int(asset.license.commercial_use),
                int(asset.license.modification_allowed),
                int(asset.license.attribution_required),
                asset.local_path,
                asset.remote_url,
                asset.preview_url,
                json.dumps(asset.metadata) if asset.metadata else None,
                asset.created_at or datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

        # Update tags
        _save_asset_tags(conn, asset.id, asset.tags)


def _save_asset_tags(
    conn: sqlite3.Connection,
    asset_id: str,
    tags: list[str],
    source: str = "manual",
    confidence: float = 1.0,
) -> None:
    """Save tags for an asset."""
    # Remove existing tags
    conn.execute("DELETE FROM asset_tags WHERE asset_id = ?", (asset_id,))

    for tag in tags:
        tag = tag.strip().lower()
        if not tag:
            continue

        # Get or create tag
        cursor = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,))
        row = cursor.fetchone()

        if row:
            tag_id = row["id"]
        else:
            cursor = conn.execute("INSERT INTO tags (name, source) VALUES (?, ?)", (tag, source))
            tag_id = cursor.lastrowid

        # Link tag to asset
        conn.execute(
            "INSERT OR IGNORE INTO asset_tags (asset_id, tag_id, confidence) VALUES (?, ?, ?)",
            (asset_id, tag_id, confidence),
        )


def get_asset(asset_id: str) -> Asset | None:
    """Get an asset by ID."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,))
        row = cursor.fetchone()

        if not row:
            return None

        tags = get_asset_tags(conn, asset_id)
        return _row_to_asset(row, tags)


def delete_asset(asset_id: str) -> bool:
    """Delete an asset by ID."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        return cursor.rowcount > 0


def search_assets(
    query: str | None = None,
    asset_type: AssetType | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    min_duration_ms: int | None = None,
    max_duration_ms: int | None = None,
    local_only: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> SearchResult:
    """Search for assets in the database.

    Args:
        query: Free-text search query
        asset_type: Filter by asset type
        source: Filter by source
        tags: Filter by tags (AND logic)
        min_duration_ms: Minimum duration
        max_duration_ms: Maximum duration
        local_only: Only return assets with local_path
        page: Page number (1-indexed)
        page_size: Results per page

    Returns:
        SearchResult with matching assets
    """
    with get_connection() as conn:
        # Build query
        conditions = []
        params: list[Any] = []

        # Full-text search
        if query:
            # Use FTS for text search
            conditions.append(
                """
                id IN (
                    SELECT id FROM assets_fts WHERE assets_fts MATCH ?
                )
                """
            )
            # Escape special FTS characters and create search term
            escaped_query = query.replace('"', '""')
            params.append(f'"{escaped_query}"')

        if asset_type:
            conditions.append("asset_type = ?")
            params.append(asset_type.value)

        if source:
            conditions.append("source = ?")
            params.append(source)

        if local_only:
            conditions.append("local_path IS NOT NULL")

        if min_duration_ms is not None:
            conditions.append("duration_ms >= ?")
            params.append(min_duration_ms)

        if max_duration_ms is not None:
            conditions.append("duration_ms <= ?")
            params.append(max_duration_ms)

        # Tag filtering (AND logic - must have all specified tags)
        if tags:
            for tag in tags:
                conditions.append(
                    """
                    id IN (
                        SELECT at.asset_id FROM asset_tags at
                        JOIN tags t ON at.tag_id = t.id
                        WHERE t.name = ?
                    )
                    """
                )
                params.append(tag.lower())

        # Build WHERE clause
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM assets WHERE {where_clause}"
        cursor = conn.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Get paginated results
        offset = (page - 1) * page_size
        select_query = f"""
            SELECT * FROM assets
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """
        cursor = conn.execute(select_query, [*params, page_size, offset])
        rows = cursor.fetchall()

        # Convert to assets with tags
        assets = []
        for row in rows:
            asset_tags = get_asset_tags(conn, row["id"])
            assets.append(_row_to_asset(row, asset_tags))

        return SearchResult(
            assets=assets,
            total_count=total_count,
            page=page,
            page_size=page_size,
            source="local",
        )


def add_tags_to_asset(
    asset_id: str,
    tags: list[str],
    source: str = "manual",
    confidence: float = 1.0,
) -> bool:
    """Add tags to an existing asset.

    Args:
        asset_id: Asset ID
        tags: Tags to add
        source: Tag source (manual, ai, api)
        confidence: Confidence score for AI tags

    Returns:
        True if successful
    """
    with get_connection() as conn:
        # Verify asset exists
        cursor = conn.execute("SELECT id FROM assets WHERE id = ?", (asset_id,))
        if not cursor.fetchone():
            return False

        for tag in tags:
            tag = tag.strip().lower()
            if not tag:
                continue

            # Get or create tag
            cursor = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,))
            row = cursor.fetchone()

            if row:
                tag_id = row["id"]
            else:
                cursor = conn.execute(
                    "INSERT INTO tags (name, source) VALUES (?, ?)", (tag, source)
                )
                tag_id = cursor.lastrowid

            # Link tag to asset (ignore if already exists)
            conn.execute(
                """
                INSERT OR REPLACE INTO asset_tags (asset_id, tag_id, confidence)
                VALUES (?, ?, ?)
                """,
                (asset_id, tag_id, confidence),
            )

        return True


def remove_tags_from_asset(asset_id: str, tags: list[str]) -> bool:
    """Remove tags from an asset."""
    with get_connection() as conn:
        for tag in tags:
            conn.execute(
                """
                DELETE FROM asset_tags
                WHERE asset_id = ? AND tag_id IN (
                    SELECT id FROM tags WHERE name = ?
                )
                """,
                (asset_id, tag.lower()),
            )
        return True


def get_all_tags() -> list[dict]:
    """Get all tags with usage counts."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT t.name, t.source, COUNT(at.asset_id) as count
            FROM tags t
            LEFT JOIN asset_tags at ON t.id = at.tag_id
            GROUP BY t.id
            ORDER BY count DESC, t.name
            """
        )
        return [dict(row) for row in cursor.fetchall()]


# Indexed folders management


def save_indexed_folder(
    path: str,
    asset_type: AssetType | None = None,
    recursive: bool = True,
    file_count: int = 0,
) -> None:
    """Save or update an indexed folder."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO indexed_folders
            (path, asset_type, recursive, last_scanned, file_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                path,
                asset_type.value if asset_type else None,
                int(recursive),
                datetime.now().isoformat(),
                file_count,
            ),
        )


def get_indexed_folders() -> list[dict]:
    """Get all indexed folders."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM indexed_folders ORDER BY path")
        return [dict(row) for row in cursor.fetchall()]


def delete_indexed_folder(path: str) -> bool:
    """Delete an indexed folder and its assets."""
    with get_connection() as conn:
        # Delete assets from this folder
        conn.execute(
            "DELETE FROM assets WHERE source = 'local' AND local_path LIKE ?",
            (f"{path}%",),
        )
        # Delete folder record
        cursor = conn.execute("DELETE FROM indexed_folders WHERE path = ?", (path,))
        return cursor.rowcount > 0


# API key management


def save_api_key(source: str, api_key: str) -> None:
    """Save an API key for a source."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO api_keys (source, api_key) VALUES (?, ?)",
            (source, api_key),
        )


def get_api_key(source: str) -> str | None:
    """Get an API key for a source."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT api_key FROM api_keys WHERE source = ?", (source,))
        row = cursor.fetchone()
        return row["api_key"] if row else None


def delete_api_key(source: str) -> bool:
    """Delete an API key."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM api_keys WHERE source = ?", (source,))
        return cursor.rowcount > 0


# Configuration


def get_config(key: str, default: str | None = None) -> str | None:
    """Get a configuration value."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else default


def set_config(key: str, value: str) -> None:
    """Set a configuration value."""
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))


# Initialize database on import
init_database()
