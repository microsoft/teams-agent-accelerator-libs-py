import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from memory_module.storage.migrations_manager import MigrationManager

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """Base class for SQLite storage operations."""

    @staticmethod
    def ensure_db_folder(db_path: Path) -> None:
        """Create the database folder if it doesn't exist."""
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def __init__(self, db_path: str | Path):
        """Initialize SQLite storage."""
        self.ensure_db_folder(db_path)
        self.db_path = str(Path(db_path).resolve())
        # Run migrations once at startup
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Run migrations during initialization."""
        migration_manager = MigrationManager(self.db_path)
        migration_manager.run_migrations()
        migration_manager.close()

    async def execute(self, query: str, parameters: tuple = ()) -> None:
        """Execute a SQL query."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(query, parameters)
            await conn.commit()

    async def execute_many(self, query: str, parameters: List[tuple]) -> None:
        """Execute a SQL query multiple times with different parameters."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executemany(query, parameters)
            await conn.commit()

    async def fetch_one(self, query: str, parameters: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch a single row from the database."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute(query, parameters) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row, strict=False))

    async def fetch_all(self, query: str, parameters: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all matching rows from the database."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute(query, parameters) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row, strict=False)) for row in rows]
