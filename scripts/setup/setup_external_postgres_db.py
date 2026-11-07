#!/usr/bin/env python3
"""
Create a template database from an external PostgreSQL SQL dump, then optionally
clone a real database from that template.

This is intended for onboarding new backbones such as:
- LiveSQLBench large PostgreSQL dumps
- OpenMRS PostgreSQL-ready exports
- other third-party PostgreSQL datasets

Example:
  python scripts/setup/setup_external_postgres_db.py \
    --mode aws \
    --db-name openmrs \
    --sql-file /path/to/openmrs.sql \
    --drop-existing
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_database_config, get_database_mode


def _connect_admin(mode: str):
    cfg = get_database_config(mode)
    conn = psycopg2.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        user=cfg["user"],
        password=cfg["password"],
        database="postgres",
        connect_timeout=int(cfg.get("connect_timeout", 10)),
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


def _database_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        return cur.fetchone() is not None


def _drop_database(conn, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (name,),
        )
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))


def _create_database(conn, name: str, owner: str, template: str | None = None) -> None:
    with conn.cursor() as cur:
        if template:
            cur.execute(
                sql.SQL("CREATE DATABASE {} WITH OWNER {} TEMPLATE {}").format(
                    sql.Identifier(name),
                    sql.Identifier(owner),
                    sql.Identifier(template),
                )
            )
        else:
            cur.execute(
                sql.SQL("CREATE DATABASE {} WITH OWNER {} ENCODING 'UTF8' TEMPLATE template0").format(
                    sql.Identifier(name),
                    sql.Identifier(owner),
                )
            )


def _iter_sql_sources(sql_file: Path | None, sql_dir: Path | None) -> List[Path]:
    if sql_file:
        return [sql_file]
    assert sql_dir is not None
    sources = sorted(
        p for p in sql_dir.iterdir() if p.is_file() and p.suffix in {".sql", ".gz"}
    )
    if not sources:
        raise FileNotFoundError(f"No .sql or .gz files found in {sql_dir}")
    return sources


def _run_psql_import(mode: str, db_name: str, sources: Iterable[Path], psql_bin: str) -> None:
    cfg = get_database_config(mode)
    base_cmd = [
        psql_bin,
        "-h",
        str(cfg["host"]),
        "-p",
        str(cfg["port"]),
        "-U",
        str(cfg["user"]),
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
    ]
    env = os.environ.copy()
    env["PGPASSWORD"] = str(cfg["password"])

    for source in sources:
        print(f"Importing {source} into {db_name} ...")
        if source.suffix == ".gz":
            with gzip.open(source, "rb") as fh:
                subprocess.run(base_cmd, stdin=fh, env=env, check=True)
        else:
            with source.open("rb") as fh:
                subprocess.run(base_cmd, stdin=fh, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PostgreSQL template database from an external SQL dump."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "aws", "custom"],
        default=get_database_mode(),
        help="Database config mode from src.config (default: DB_MODE env var)",
    )
    parser.add_argument("--db-name", required=True, help="Logical database name, e.g. openmrs")
    parser.add_argument(
        "--template-name",
        help="Template database name (default: <db-name>_template)",
    )
    parser.add_argument(
        "--sql-file",
        type=Path,
        help="Path to a PostgreSQL .sql or .sql.gz dump",
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        help="Directory containing ordered .sql or .sql.gz files to import",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing template/clone databases before recreating them",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Only build the template DB; do not create <db-name> from it",
    )
    parser.add_argument(
        "--psql-bin",
        default="psql",
        help="psql binary to use for imports (default: psql)",
    )
    args = parser.parse_args()

    if bool(args.sql_file) == bool(args.sql_dir):
        parser.error("Specify exactly one of --sql-file or --sql-dir")
    return args


def main() -> int:
    args = parse_args()
    template_name = args.template_name or f"{args.db_name}_template"
    sources = _iter_sql_sources(args.sql_file, args.sql_dir)
    owner = get_database_config(args.mode)["user"]

    conn = _connect_admin(args.mode)
    try:
        for db_name in [template_name, None if args.skip_clone else args.db_name]:
            if not db_name:
                continue
            exists = _database_exists(conn, db_name)
            if exists and not args.drop_existing:
                raise RuntimeError(
                    f"Database '{db_name}' already exists. Re-run with --drop-existing."
                )
            if exists:
                print(f"Dropping existing database {db_name} ...")
                _drop_database(conn, db_name)

        print(f"Creating template database {template_name} ...")
        _create_database(conn, template_name, owner=owner)
    finally:
        conn.close()

    _run_psql_import(args.mode, template_name, sources, args.psql_bin)

    if not args.skip_clone:
        conn = _connect_admin(args.mode)
        try:
            print(f"Creating clone database {args.db_name} from template {template_name} ...")
            _create_database(conn, args.db_name, owner=owner, template=template_name)
        finally:
            conn.close()

    print(
        {
            "mode": args.mode,
            "template_db": template_name,
            "clone_db": None if args.skip_clone else args.db_name,
            "sources": [str(p) for p in sources],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
