"""
Results Storage — Lightweight SQLite persistence for benchmark history.

Provides a simple API to save, query, and clear past benchmark runs
for the Laboratory view. Auto-creates the database on first use.

Design: Follows the Single Responsibility Principle — this module
handles only persistence, not UI rendering or analysis logic.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class ResultsStore:
    """
    SQLite-backed storage for benchmark run results.

    Stores results from both Microscope (single-pair) and
    Arena (batch) runs, enabling historical comparison in
    the Laboratory view.
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            view        TEXT    NOT NULL,
            strategy    TEXT    NOT NULL,
            file_name   TEXT,
            scores      TEXT    NOT NULL,
            metadata    TEXT
        )
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the results store.

        Args:
            db_path: Path to the SQLite database file.
                     Defaults to config.DB_PATH.
        """
        if db_path is None:
            import config
            db_path = config.DB_PATH

        self._db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create table if not exists
        self._execute(self._CREATE_TABLE_SQL)

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a new database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write query."""
        conn = self._get_connection()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def save_run(
        self,
        view: str,
        strategy: str,
        file_name: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save a benchmark run result.

        Args:
            view:      'microscope' or 'arena'.
            strategy:  Name of the extraction strategy used.
            file_name: Name of the file(s) processed.
            scores:    Dictionary of metric_name → score.
            metadata:  Optional dict with extra info (word count, time, etc.).

        Returns:
            The row ID of the inserted record.
        """
        timestamp = datetime.now().isoformat()
        scores_json = json.dumps(scores)
        metadata_json = json.dumps(metadata) if metadata else None

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO benchmark_runs
                    (timestamp, view, strategy, file_name, scores, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, view, strategy, file_name, scores_json, metadata_json)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_runs(
        self,
        view: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query past benchmark runs.

        Args:
            view:     Filter by view type ('microscope', 'arena').
            strategy: Filter by strategy name.
            limit:    Maximum number of rows to return.

        Returns:
            List of dicts, each representing a run.
        """
        query = "SELECT * FROM benchmark_runs WHERE 1=1"
        params = []

        if view:
            query += " AND view = ?"
            params.append(view)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_connection()
        try:
            rows = conn.execute(query, tuple(params)).fetchall()
            results = []
            for row in rows:
                entry = dict(row)
                entry["scores"] = json.loads(entry["scores"])
                if entry["metadata"]:
                    entry["metadata"] = json.loads(entry["metadata"])
                results.append(entry)
            return results
        finally:
            conn.close()

    def get_run_by_id(self, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single run by its ID.

        Args:
            run_id: The primary key ID of the run.

        Returns:
            Dict representing the run, or None if not found.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM benchmark_runs WHERE id = ?",
                (run_id,)
            ).fetchone()

            if row is None:
                return None

            entry = dict(row)
            entry["scores"] = json.loads(entry["scores"])
            if entry["metadata"]:
                entry["metadata"] = json.loads(entry["metadata"])
            return entry
        finally:
            conn.close()

    def clear_history(self) -> int:
        """
        Delete all stored benchmark runs.

        Returns:
            Number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM benchmark_runs")
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def __str__(self) -> str:
        return f"ResultsStore(db='{self._db_path}')"
