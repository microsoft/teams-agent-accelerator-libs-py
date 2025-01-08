import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import List, Tuple

import sqlite_vec

logger = logging.getLogger(__name__)


class MigrationManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Initialize the database if needed
        with self.__get_connection() as conn:
            self.__create_vector_search_table(conn)
            self.__create_migrations_table(conn)

    def run_migrations(self):
        logger.info("Migrations", "Running migrations")
        with self.__get_connection() as conn:
            applied_migrations = set(self.__get_applied_migrations(conn))
            migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")

            files = os.listdir(migrations_dir)
            files.sort(key=lambda x: int(x.split("_")[0]))
            for filename in files:
                if filename.endswith(".sql"):
                    migration_name = os.path.splitext(filename)[0]
                    if migration_name not in applied_migrations:
                        logger.info("Migrations", f"Applying migration: {migration_name}")
                        with open(os.path.join(migrations_dir, filename), "r") as f:
                            sql = f.read()
                        self.__apply_migration(conn, migration_name, sql)
                        logger.info("Migrations", f"Migration applied: {migration_name}")

    # Changed to double underscore for true private methods
    @contextmanager
    def __get_connection(self):
        conn = None
        try:
            with sqlite3.connect(self.db_path) as conn:
                yield conn
        finally:
            if conn is not None:
                conn.close()

    def __create_vector_search_table(self, conn):
        logger.info("Creating vector search table", self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
            USING vec0(memory_embedding_id INTEGER PRIMARY KEY AUTOINCREMENT, embedding float[1536]);
        """)

    def __create_migrations_table(self, conn):
        conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY,
                    migration_name TEXT NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def __get_applied_migrations(self, conn) -> List[str]:
        cursor = conn.execute("SELECT migration_name FROM migrations ORDER BY id")
        return [row[0] for row in cursor.fetchall()]

    def __apply_migration(self, conn, migration_name: str, sql: str):
        conn.executescript(sql)
        conn.execute("INSERT INTO migrations (migration_name) VALUES (?)", (migration_name,))

    def get_last_applied_migration(self) -> Tuple[int, str]:
        with self.__get_connection() as conn:
            cursor = conn.execute("SELECT id, migration_name FROM migrations ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            return result if result else (0, "No migrations applied")

    def close(self):
        with self.__get_connection() as conn:
            conn.close()
