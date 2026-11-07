"""
Database integrity verification and restoration.

Verifies RDS databases match BIRD ground truth and restores if needed.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from ..config import get_database_config
from ..logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GROUNDTRUTH = PROJECT_ROOT / "data" / "groundtruth.json"
DEFAULT_DUMPS_DIR = PROJECT_ROOT / "data"


def get_db_connection(db_name: str, mode: str = "aws"):
    """Get connection to a specific database."""
    config = get_database_config(mode)
    return psycopg2.connect(
        host=config['host'],
        port=int(config['port']),
        user=config['user'],
        password=config['password'],
        database=db_name
    )


def load_groundtruth(path: Path = None) -> Dict:
    """Load ground truth data."""
    path = path or DEFAULT_GROUNDTRUTH
    with open(path, 'r') as f:
        return json.load(f)


def get_table_counts(db_name: str, mode: str = "aws") -> Optional[Dict[str, int]]:
    """Get row counts for all tables in a database."""
    try:
        conn = get_db_connection(db_name, mode)
        cur = conn.cursor()

        cur.execute("""
            SELECT c.relname FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r'
        """)

        tables = {}
        for (table_name,) in cur.fetchall():
            cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            tables[table_name] = cur.fetchone()[0]

        cur.close()
        conn.close()
        return tables
    except psycopg2.OperationalError as e:
        if 'does not exist' in str(e):
            return None
        raise


def verify_database(gt_name: str, groundtruth: Dict, mode: str = "aws") -> Dict:
    """Verify a single database against ground truth."""
    gt_db = groundtruth['databases'].get(gt_name)
    if not gt_db:
        return {'status': 'NOT_FOUND', 'message': 'Not in ground truth'}

    rds_name = gt_db.get('rds_name')
    if not rds_name:
        return {'status': 'NO_MAPPING', 'message': 'No RDS mapping'}

    rds_state = get_table_counts(rds_name, mode)
    if rds_state is None:
        return {'status': 'MISSING', 'rds_name': rds_name, 'message': 'Database missing'}

    gt_tables = gt_db['tables']
    errors = []

    for table, info in gt_tables.items():
        gt_count = info.get('row_count', 0)
        rds_count = rds_state.get(table, 0)
        if gt_count != rds_count:
            errors.append(f"{table}: expected {gt_count}, got {rds_count}")

    if errors:
        return {'status': 'MISMATCH', 'rds_name': rds_name, 'errors': errors}

    return {'status': 'OK', 'rds_name': rds_name}


def restore_database(gt_name: str, groundtruth: Dict, dumps_dir: Path = None, mode: str = "aws") -> bool:
    """Drop, recreate, and import a database from SQL dump."""
    dumps_dir = dumps_dir or DEFAULT_DUMPS_DIR
    gt_db = groundtruth['databases'].get(gt_name)
    rds_name = gt_db.get('rds_name')

    if not rds_name:
        logger.error(f"No RDS mapping for {gt_name}")
        return False

    # Find SQL dump
    dump_file = dumps_dir / f"{gt_name}_template" / f"{gt_name}_full.sql"
    if not dump_file.exists():
        logger.error(f"SQL dump not found: {dump_file}")
        return False

    config = get_database_config(mode)

    try:
        # Drop and recreate database
        conn = psycopg2.connect(
            host=config['host'], port=int(config['port']),
            user=config['user'], password=config['password'],
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        logger.info(f"Dropping {rds_name}...")
        cur.execute(f'DROP DATABASE IF EXISTS "{rds_name}"')

        logger.info(f"Creating {rds_name}...")
        cur.execute(f'CREATE DATABASE "{rds_name}"')

        cur.close()
        conn.close()

        # Import data
        logger.info(f"Importing from {dump_file}...")
        with open(dump_file, 'r') as f:
            sql = f.read()

        conn = get_db_connection(rds_name, mode)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Restored {rds_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to restore {rds_name}: {e}")
        return False


def verify_and_restore(databases: List[str] = None, mode: str = "aws", auto_restore: bool = False) -> bool:
    """
    Verify databases and optionally restore mismatched ones.

    Returns True if all databases are OK (or were restored successfully).
    """
    groundtruth = load_groundtruth()
    db_list = databases or list(groundtruth['databases'].keys())

    all_ok = True
    for db_name in db_list:
        result = verify_database(db_name, groundtruth, mode)
        status = result['status']

        if status == 'OK':
            print(f"✅ {db_name}")
        elif status in ['MISMATCH', 'MISSING']:
            print(f"❌ {db_name}: {result.get('message', status)}")
            if auto_restore:
                if restore_database(db_name, groundtruth, mode=mode):
                    print(f"   ✅ Restored {db_name}")
                else:
                    all_ok = False
            else:
                all_ok = False
        else:
            print(f"⚠️  {db_name}: {result.get('message', status)}")

    return all_ok


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--databases', nargs='+', help='Databases to check')
    parser.add_argument('--restore', action='store_true', help='Auto-restore mismatched databases')
    parser.add_argument('--mode', default='aws', choices=['local', 'aws', 'custom'])
    args = parser.parse_args()

    ok = verify_and_restore(args.databases, args.mode, args.restore)
    exit(0 if ok else 1)
