import logging
import os
import sqlite3
from typing import List, Tuple

logger = logging.getLogger(__name__)


class MigrationManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_migrations_table()

    def create_migrations_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY,
                    migration_name TEXT NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get_applied_migrations(self) -> List[str]:
        with self.conn:
            cursor = self.conn.execute("SELECT migration_name FROM migrations ORDER BY id")
            return [row[0] for row in cursor.fetchall()]

    def apply_migration(self, migration_name: str, sql: str):
        with self.conn:
            self.conn.executescript(sql)
            self.conn.execute("INSERT INTO migrations (migration_name) VALUES (?)", (migration_name,))

    def run_migrations(self):
        logger.info("Migrations", "Running migrations")
        applied_migrations = set(self.get_applied_migrations())
        migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")

        files = os.listdir(migrations_dir)
        # File names are in format <int>_<name>.sql
        files.sort(key=lambda x: int(x.split("_")[0]))
        for filename in files:
            if filename.endswith(".sql"):
                migration_name = os.path.splitext(filename)[0]
                if migration_name not in applied_migrations:
                    logger.info("Migrations", f"Applying migration: {migration_name}")
                    with open(os.path.join(migrations_dir, filename), "r") as f:
                        sql = f.read()
                    self.apply_migration(migration_name, sql)
                    logger.info("Migrations", f"Migration applied: {migration_name}")

    def get_last_applied_migration(self) -> Tuple[int, str]:
        with self.conn:
            cursor = self.conn.execute("SELECT id, migration_name FROM migrations ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            return result if result else (0, "No migrations applied")

    def close(self):
        self.conn.close()
