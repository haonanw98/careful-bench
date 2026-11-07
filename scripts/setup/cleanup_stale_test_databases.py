#!/usr/bin/env python3
"""
List and optionally purge stale cloned test databases on PostgreSQL.

By default this targets databases whose names begin with ``test_`` and skips any
database that currently has active connections. This is intended for cleaning up
benchmark scratch databases left behind by interrupted runs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2 import sql

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_database_config


TIMESTAMP_RE = re.compile(r"_(20\d{18})_[0-9a-f]{6,}$")


@dataclass
class CandidateDatabase:
    name: str
    size_bytes: int
    active_connections: int
    created_at: Optional[datetime]

    @property
    def age_hours(self) -> Optional[float]:
        if self.created_at is None:
            return None
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600.0


def _parse_created_at(name: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    raw = match.group(1)
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _admin_connection(mode: str):
    cfg = get_database_config(mode)
    conn = psycopg2.connect(**{**cfg, "database": "postgres"})
    conn.autocommit = True
    return conn


def _load_candidates(conn, prefix: str) -> list[CandidateDatabase]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH activity AS (
                SELECT datname, COUNT(*) AS active_connections
                FROM pg_stat_activity
                WHERE datname LIKE %s
                GROUP BY datname
            )
            SELECT d.datname,
                   pg_database_size(d.datname) AS size_bytes,
                   COALESCE(a.active_connections, 0) AS active_connections
            FROM pg_database d
            LEFT JOIN activity a ON a.datname = d.datname
            WHERE d.datistemplate = false
              AND d.datallowconn = true
              AND d.datname LIKE %s
            ORDER BY d.datname
            """,
            (f"{prefix}%", f"{prefix}%"),
        )
        return [
            CandidateDatabase(
                name=name,
                size_bytes=size_bytes,
                active_connections=active_connections,
                created_at=_parse_created_at(name),
            )
            for name, size_bytes, active_connections in cur.fetchall()
        ]


def _drop_database(conn, db_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (db_name,),
        )
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))


def _format_size_gb(size_bytes: int) -> str:
    return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup stale cloned benchmark databases.")
    parser.add_argument("--mode", choices=["local", "aws", "custom"], default=None)
    parser.add_argument("--prefix", default="test_", help="Database name prefix to target")
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=24.0,
        help="Only target databases at least this old. Use 0 to match all ages.",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Also delete databases with active connections after terminating them.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually drop databases. Without this flag, only print a dry-run summary.",
    )
    args = parser.parse_args()

    conn = _admin_connection(args.mode)
    try:
        candidates = _load_candidates(conn, args.prefix)
        selected: list[CandidateDatabase] = []
        for candidate in candidates:
            age_hours = candidate.age_hours
            if age_hours is not None and age_hours < args.min_age_hours:
                continue
            if candidate.active_connections > 0 and not args.include_active:
                continue
            selected.append(candidate)

        total_bytes = sum(candidate.size_bytes for candidate in selected)
        print(f"Matched {len(candidates)} databases with prefix {args.prefix!r}.")
        print(
            f"Selected {len(selected)} databases "
            f"({ _format_size_gb(total_bytes) }) "
            f"for {'deletion' if args.apply else 'dry-run'}."
        )
        for candidate in selected[:50]:
            age = "unknown-age" if candidate.age_hours is None else f"{candidate.age_hours:.1f}h"
            print(
                f"{candidate.name}\t{_format_size_gb(candidate.size_bytes)}"
                f"\tactive={candidate.active_connections}\tage={age}"
            )
        if len(selected) > 50:
            print(f"... and {len(selected) - 50} more")

        if not args.apply:
            return 0

        dropped = 0
        for candidate in selected:
            _drop_database(conn, candidate.name)
            dropped += 1

        print(f"Dropped {dropped} databases.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
