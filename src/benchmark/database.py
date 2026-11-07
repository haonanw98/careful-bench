"""
Consolidated database operations for benchmark framework.

This module combines:
- Database configuration
- Snapshot management (from test_database.py)
- SQL execution with hooks (from db_executor.py)
- Fast database cloning (from fast_test_runner.py)
- Database maintenance and verification
"""

import os
import json
import secrets
import time
from collections import Counter
from decimal import Decimal
import psycopg2
from psycopg2 import sql as psycopg2_sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Dict, Any, Optional, List, Tuple, Union, Iterable
from datetime import datetime, timezone
from pathlib import Path
import re
from dataclasses import dataclass
from ..logger import get_logger

logger = get_logger(__name__)

_JSON_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}(?::?\d{2})?)?$"
)


# =============================================================================
# Data Classes for State Modification Comparison
# =============================================================================

@dataclass
class TableModification:
    """
    Modifications to a single table using set difference operations.

    - columns: dict mapping column names to their data types (schema information)
    - present_in_initial/final: whether the table exists in the initial/final state
    - initial_except_final: rows in initial state EXCEPT final state (deleted or old versions)
    - final_except_initial: rows in final state EXCEPT initial state (inserted or new versions)
    """
    columns: Dict[str, str]  # {column_name: data_type}
    initial_except_final: List[Dict[str, Any]]
    final_except_initial: List[Dict[str, Any]]
    present_in_initial: bool = True
    present_in_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'columns': self.columns,
            'present_in_initial': self.present_in_initial,
            'present_in_final': self.present_in_final,
            'initial_except_final': self.initial_except_final,
            'final_except_initial': self.final_except_initial
        }

    def _dict_to_comparable(self, d: Dict[str, Any]) -> tuple:
        """Convert dict to comparable tuple for set operations."""
        return tuple(
            sorted(
                (key, self._value_to_comparable(key, value))
                for key, value in d.items()
            )
        )

    def _value_to_comparable(self, column: str, value: Any) -> Any:
        """Normalize values that are semantically equal but text-render differently."""
        if value is None:
            return ("__sql_null__",)

        column_type = (self.columns.get(column) or "").lower()
        if column_type in {"json", "jsonb"} and isinstance(value, str):
            try:
                parsed = json.loads(
                    value,
                    parse_float=Decimal,
                    parse_int=Decimal,
                )
            except Exception:
                return ("__json_text__", value)
            return ("__json__", self._json_to_comparable(parsed))

        if isinstance(value, bool):
            return ("__bool__", value)
        if isinstance(value, Decimal):
            return ("__decimal__", str(value.normalize()))
        if isinstance(value, (int, float)):
            return ("__number__", value)
        if isinstance(value, datetime):
            return ("__datetime__", value.isoformat())
        if isinstance(value, str):
            return ("__text__", value)

        try:
            return ("__jsonish__", json.dumps(value, sort_keys=True, default=str))
        except Exception:
            return ("__repr__", str(value))

    def _json_to_comparable(self, value: Any) -> Any:
        """Canonicalize parsed JSON, including numeric scale in JSONB text output."""
        if value is None:
            return ("__json_null__",)
        if isinstance(value, bool):
            return ("__json_bool__", value)
        if isinstance(value, Decimal):
            if value == value.to_integral_value():
                normalized = str(value.quantize(Decimal(1)))
            else:
                normalized = format(value.normalize(), "f")
            return ("__json_number__", normalized)
        if isinstance(value, str):
            if _JSON_TIMESTAMP_RE.match(value):
                normalized = value.replace(" ", "T").replace("Z", "+00:00")
                normalized = re.sub(r"([+-]\d{2})$", r"\1:00", normalized)
                try:
                    parsed = datetime.fromisoformat(normalized)
                    if parsed.tzinfo is not None:
                        normalized = parsed.astimezone(timezone.utc).isoformat()
                except ValueError:
                    pass
                return ("__json_timestamp__", normalized)
            return ("__json_string__", value)
        if isinstance(value, list):
            return ("__json_array__", tuple(self._json_to_comparable(v) for v in value))
        if isinstance(value, dict):
            return (
                "__json_object__",
                tuple(
                    sorted(
                        (str(k), self._json_to_comparable(v))
                        for k, v in value.items()
                    )
                ),
            )
        return value

    def compare_equal(self, other: 'TableModification') -> bool:
        """Compare if two table modifications are equal (preserves duplicates)."""
        if self.present_in_initial != other.present_in_initial:
            return False
        if self.present_in_final != other.present_in_final:
            return False

        # Compare columns (schema)
        if self.columns != other.columns:
            return False

        # Compare initial_except_final rows (order-independent but preserves duplicates)
        initial_except_final_self = sorted(self._dict_to_comparable(d) for d in self.initial_except_final)
        initial_except_final_other = sorted(self._dict_to_comparable(d) for d in other.initial_except_final)
        if initial_except_final_self != initial_except_final_other:
            return False

        # Compare final_except_initial rows (order-independent but preserves duplicates)
        final_except_initial_self = sorted(self._dict_to_comparable(d) for d in self.final_except_initial)
        final_except_initial_other = sorted(self._dict_to_comparable(d) for d in other.final_except_initial)
        if final_except_initial_self != final_except_initial_other:
            return False

        return True

    def without_columns(self, ignored_columns: Iterable[str]) -> Optional['TableModification']:
        """Return a copy with non-semantic columns removed from row diffs."""
        ignored = set(ignored_columns)
        if not ignored:
            return self

        filtered_columns = {
            column: data_type
            for column, data_type in self.columns.items()
            if column not in ignored
        }

        def filter_row(row: Dict[str, Any]) -> Dict[str, Any]:
            return {
                column: value
                for column, value in row.items()
                if column not in ignored
            }

        filtered = TableModification(
            columns=filtered_columns,
            present_in_initial=self.present_in_initial,
            present_in_final=self.present_in_final,
            initial_except_final=[filter_row(row) for row in self.initial_except_final],
            final_except_initial=[filter_row(row) for row in self.final_except_initial],
        )

        # If the only changed columns were ignored, the remaining before/after
        # row bags are identical and should not count as a state change.
        initial_counter = Counter(
            filtered._dict_to_comparable(row)
            for row in filtered.initial_except_final
        )
        final_counter = Counter(
            filtered._dict_to_comparable(row)
            for row in filtered.final_except_initial
        )
        if initial_counter == final_counter and filtered.present_in_initial == filtered.present_in_final:
            return None

        return filtered


@dataclass
class StateModification:
    """Database state modifications across multiple tables."""
    tables: Dict[str, TableModification]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            table_name: table_mod.to_dict()
            for table_name, table_mod in self.tables.items()
        }

    def compare_equal(self, other: 'StateModification') -> bool:
        """Compare if two state modifications are equal."""
        # Check if same tables are modified
        if set(self.tables.keys()) != set(other.tables.keys()):
            return False

        # Compare each table's modifications
        for table_name, table_mod in self.tables.items():
            other_table_mod = other.tables.get(table_name)
            if not table_mod.compare_equal(other_table_mod):
                return False

        return True

    def without_columns(self, ignored_columns_by_table: Dict[str, Iterable[str]]) -> 'StateModification':
        """Return a copy with explicitly ignored columns removed from comparison."""
        if not ignored_columns_by_table:
            return self

        filtered_tables: Dict[str, TableModification] = {}
        for table_name, table_mod in self.tables.items():
            ignored_columns = ignored_columns_by_table.get(table_name, [])
            filtered_table = table_mod.without_columns(ignored_columns)
            if filtered_table is not None:
                filtered_tables[table_name] = filtered_table

        return StateModification(tables=filtered_tables)

    def write_to_json(self, file_path: Path) -> None:
        """Write state modification to JSON file (expected_snapshot.json format)."""
        tables_data = {}
        for table_name, table_mod in self.tables.items():
            tables_data[table_name] = {
                'columns': table_mod.columns,
                'present_in_initial': table_mod.present_in_initial,
                'present_in_final': table_mod.present_in_final,
                'initial_except_final': table_mod.initial_except_final,
                'final_except_initial': table_mod.final_except_initial
            }

        data = {
            'timestamp': datetime.now().isoformat(),
            'tables': tables_data
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: Path) -> 'StateModification':
        """Load state modification from expected_snapshot.json file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        tables = {}
        for table_name, table_data in data['tables'].items():
            tables[table_name] = TableModification(
                columns=table_data.get('columns', {}),
                present_in_initial=table_data.get('present_in_initial', True),
                present_in_final=table_data.get('present_in_final', True),
                initial_except_final=table_data.get('initial_except_final', []),
                final_except_initial=table_data.get('final_except_initial', [])
            )

        return cls(tables=tables)


# =============================================================================
# Configuration
# =============================================================================

def get_db_config():
    """Get database configuration using the centralized config utility.

    Uses DB_MODE from environment (local/aws/custom) via src.config.get_database_config.
    """
    from src.config import get_database_config as _get
    return _get()


class SnapshotManager:
    """Manages database snapshots for testing using delta-based comparison."""

    def __init__(self, scn: str, db_config: Dict[str, Any]):
        """
        Initialize SnapshotManager.

        Args:
            scn: scenario id 
            db_config: Database configuration (point to the test database handled by an agent)
        """
        self.db_config = db_config
        self.connection = None
        self.initial_state = DatabaseCloner(db_config)
        self.init_snapshot_db, _ = self.initial_state.create_test_database(
            scenario_name=f'{scn}_init', 
            reference_db=db_config.get('database')
        )

    def _connect(self):
        """Establish database connection."""
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(**self.db_config)

    def _disconnect(self):
        """Close database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()

    def get_modification(self) -> StateModification:
        """
        Get modifications (delta) between current database and reference database.

        Returns:
            StateModification object containing all table modifications with full row data
        """
        reference_db = self.init_snapshot_db
        current_conn = None
        reference_conn = None
        try:
            current_conn = psycopg2.connect(**self.db_config)
            reference_conn = psycopg2.connect(**{**self.db_config, "database": reference_db})

            with current_conn.cursor() as current_cur, reference_conn.cursor() as reference_cur:
                current_tables = self._get_public_tables(current_cur)
                reference_tables = self._get_public_tables(reference_cur)
                tables = sorted(set(current_tables) | set(reference_tables))

                table_modifications: Dict[str, TableModification] = {}
                for table in tables:
                    current_columns_info = self._get_public_columns(current_cur, table)
                    reference_columns_info = self._get_public_columns(reference_cur, table)

                    present_in_initial = bool(reference_columns_info)
                    present_in_final = bool(current_columns_info)
                    if not present_in_initial and not present_in_final:
                        continue

                    final_column_names = [col[0] for col in current_columns_info]
                    final_column_types = {col[0]: col[1] for col in current_columns_info}
                    initial_column_names = [col[0] for col in reference_columns_info]
                    initial_column_types = {col[0]: col[1] for col in reference_columns_info}
                    column_names = final_column_names or initial_column_names
                    column_types = final_column_types or initial_column_types

                    initial_except_final: List[Dict[str, Any]] = []
                    final_except_initial: List[Dict[str, Any]] = []

                    if present_in_initial and present_in_final and initial_column_names == final_column_names:
                        initial_rows = self._fetch_table_rows(reference_cur, table, initial_column_names)
                        final_rows = self._fetch_table_rows(current_cur, table, final_column_names)
                        initial_except_final, final_except_initial = self._diff_row_lists(
                            initial_rows,
                            final_rows,
                            column_names,
                        )
                    elif present_in_initial and not present_in_final:
                        initial_rows = self._fetch_table_rows(reference_cur, table, initial_column_names)
                        initial_except_final = [
                            dict(zip(initial_column_names, row))
                            for row in initial_rows
                        ]
                    elif present_in_final and not present_in_initial:
                        final_rows = self._fetch_table_rows(current_cur, table, final_column_names)
                        final_except_initial = [
                            dict(zip(final_column_names, row))
                            for row in final_rows
                        ]

                    table_modifications[table] = TableModification(
                        columns=column_types,
                        present_in_initial=present_in_initial,
                        present_in_final=present_in_final,
                        initial_except_final=initial_except_final,
                        final_except_initial=final_except_initial,
                    )

                return StateModification(tables=table_modifications)
        finally:
            if current_conn is not None:
                current_conn.close()
            if reference_conn is not None:
                reference_conn.close()

    @staticmethod
    def _get_public_tables(cursor) -> List[str]:
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _get_public_columns(cursor, table: str) -> List[Tuple[str, str]]:
        cursor.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return cursor.fetchall()

    @staticmethod
    def _fetch_table_rows(cursor, table: str, column_names: List[str]) -> List[Tuple[Any, ...]]:
        if not column_names:
            return []
        projection = psycopg2_sql.SQL(", ").join(
            psycopg2_sql.SQL("{}::TEXT").format(psycopg2_sql.Identifier(column))
            for column in column_names
        )
        query = psycopg2_sql.SQL("SELECT {projection} FROM {table}").format(
            projection=projection,
            table=psycopg2_sql.Identifier(table),
        )
        cursor.execute(query)
        return list(cursor.fetchall())

    @staticmethod
    def _diff_row_lists(
        initial_rows: List[Tuple[Any, ...]],
        final_rows: List[Tuple[Any, ...]],
        column_names: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        initial_counter = Counter(initial_rows)
        final_counter = Counter(final_rows)

        initial_except_final: List[Dict[str, Any]] = []
        final_except_initial: List[Dict[str, Any]] = []

        for row, count in (initial_counter - final_counter).items():
            initial_except_final.extend(dict(zip(column_names, row)) for _ in range(count))
        for row, count in (final_counter - initial_counter).items():
            final_except_initial.extend(dict(zip(column_names, row)) for _ in range(count))

        return initial_except_final, final_except_initial

    def compare_modifications(self, actual_modifications: StateModification, expected_modifications: StateModification) -> bool:
        """
        Compare actual modifications with expected modifications.

        Args:
            actual_modifications: StateModification from get_modification() containing actual database changes
            expected_modifications: StateModification from expected_snapshot.json containing expected changes

        Returns:
            bool: True if modifications match exactly (same tables modified, same rows inserted/updated/deleted)
        """
        return actual_modifications.compare_equal(expected_modifications)


class SqlExecutor:

    def __init__(
        self,
        db_config: Dict[str, Any],
        timeout: int = None,
        db_session: Optional[Dict[str, Any]] = None,
    ):
        self.db_config = db_config
        self.timeout = timeout
        self.db_session = db_session

    def _apply_session_binding(self, cursor) -> None:
        """Apply optional per-request DB session bindings on the current cursor."""
        if not self.db_session:
            return

        role_name = (
            self.db_session.get("role")
            or self.db_session.get("set_role")
            or self.db_session.get("session_role")
        )
        raw_settings = self.db_session.get("settings")
        if raw_settings is None:
            raw_settings = self.db_session.get("session_settings")
        if raw_settings is None:
            settings = {}
        else:
            if not isinstance(raw_settings, dict):
                raise ValueError("db_session.settings must be a JSON object")
            settings = raw_settings

        if role_name:
            role_text = str(role_name).strip()
            if not role_text:
                raise ValueError("db_session role name must be non-empty")
            cursor.execute(
                psycopg2_sql.SQL("SET ROLE {}").format(psycopg2_sql.Identifier(role_text))
            )

        if settings:
            for key, value in settings.items():
                cursor.execute(
                    "SELECT set_config(%s, %s, true)",
                    (str(key), str(value)),
                )

    def prepare_cursor(self, cursor, allow_modify: bool = True, bind_session: bool = True) -> None:
        """Apply transaction settings and optional session bindings to a fresh cursor."""
        if not allow_modify:
            cursor.execute("BEGIN READ ONLY;")
            if bind_session:
                self._apply_session_binding(cursor)
            if self.timeout:
                cursor.execute(f"SET LOCAL statement_timeout = {int(self.timeout)}")
        else:
            if bind_session:
                self._apply_session_binding(cursor)
            if self.timeout:
                cursor.execute(f"SET statement_timeout = {int(self.timeout)}")

    def execute(
        self,
        sql: Union[str, List[str]],
        allow_modify: bool = True,
        bind_session: bool = True,
    ):
        """Execute one SQL statement or a list of statements and return structured result."""
        conn = None
        try:
            # Check connection health and reconnect if needed
            conn = psycopg2.connect(**self.db_config)

            with conn.cursor() as cursor:
                self.prepare_cursor(cursor, allow_modify=allow_modify, bind_session=bind_session)

                start_time = datetime.now()
                statements: List[str] = []
                if isinstance(sql, list):
                    for s in sql:
                        if s is None:
                            continue
                        text = str(s).strip()
                        if text:
                            statements.append(text)
                else:
                    if sql is not None:
                        text = str(sql).strip()
                        if text:
                            statements = [text]

                last_read_data = None
                last_columns = None
                last_rows_affected = 0

                for stmt in statements:
                    cursor.execute(stmt)

                    # Determine if this statement returned rows
                    has_results = cursor.description is not None
                    if has_results:
                        last_read_data = cursor.fetchall()
                        last_columns = [desc[0] for desc in cursor.description]
                    else:
                        last_rows_affected = cursor.rowcount

                time_elapse = (datetime.now() - start_time).total_seconds()

                # Always commit changes when allow_modify=True (even if last statement returned rows),
                # so statements like DELETE ... RETURNING are persisted.
                if allow_modify:
                    conn.commit()
                else:
                    conn.rollback()

                conn.close()

                if last_read_data is not None:
                    return {
                        'success': True,
                        'type': 'READ',
                        'full_data': last_read_data,
                        'row_count': len(last_read_data),
                        'columns': last_columns or [],
                        'time_elapse': time_elapse,
                    }

                return {
                    'success': True,
                    'type': 'WRITE',
                    'rows_affected': last_rows_affected,
                    'committed': allow_modify,
                    'time_elapse': time_elapse,
                }

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            return {
                'success': False,
                'error': str(e)
            }


# =============================================================================
# Fast Database Cloning (from fast_test_runner.py)
# =============================================================================

class DatabaseCloner:
    """Fast database cloning using PostgreSQL templates."""

    def __init__(self, db_config: Optional[Dict] = None, create_db_retries: Optional[int] = None):
        self.db_config = db_config or get_db_config()
        self.test_databases = set()
        if create_db_retries is None:
            try:
                create_db_retries = int(os.getenv("DB_CLONE_CREATE_RETRIES", "10"))
            except Exception:
                create_db_retries = 10
        self.create_db_retries = max(1, int(create_db_retries))
        try:
            cleanup_db_retries = int(os.getenv("DB_CLONE_CLEANUP_RETRIES", "5"))
        except Exception:
            cleanup_db_retries = 5
        self.cleanup_db_retries = max(1, int(cleanup_db_retries))

    def _connect_admin_database(self):
        conn = psycopg2.connect(**{**self.db_config, 'database': 'postgres'})
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn

    def _drop_database(self, cursor, db_name: str) -> None:
        cursor.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (db_name,),
        )
        cursor.execute(
            psycopg2_sql.SQL("DROP DATABASE IF EXISTS {}").format(
                psycopg2_sql.Identifier(db_name)
            )
        )

    def _cleanup_test_database_once(self, db_name: str) -> None:
        conn = None
        try:
            conn = self._connect_admin_database()
            with conn.cursor() as cursor:
                self._drop_database(cursor, db_name)
        finally:
            if conn:
                conn.close()

    def _cleanup_test_database_with_retry(self, db_name: str) -> None:
        last_error = None
        for attempt in range(1, self.cleanup_db_retries + 1):
            try:
                self._cleanup_test_database_once(db_name)
                return
            except Exception as e:
                last_error = e
                retryable = isinstance(
                    e,
                    (
                        psycopg2.OperationalError,
                        psycopg2.InterfaceError,
                    ),
                ) or getattr(e, "pgcode", None) in {"55006", "57P01", "57P02", "57P03"}
                if attempt >= self.cleanup_db_retries or not retryable:
                    break
                logger.warning(
                    "Retrying cleanup for %s after attempt %s/%s failed: %s",
                    db_name,
                    attempt,
                    self.cleanup_db_retries,
                    e,
                )
                time.sleep(min(0.5 * attempt, 2.0))
        assert last_error is not None
        raise last_error

    def create_test_database(self, scenario_name: str, reference_db: str) -> Tuple[str, float]:
        """Create a test database by cloning a reference database."""
        start_time = time.time()

        def _make_db_name() -> str:
            # Keep the uniqueness suffix at the end so truncation doesn't remove it.
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            suffix = f"_{os.getpid()}_{timestamp}_{secrets.token_hex(3)}"
            prefix = f"test_{scenario_name}".lower()
            prefix = re.sub(r'[^a-z0-9_]', '_', prefix)

            max_prefix_len = 63 - len(suffix)
            if max_prefix_len < 1:
                prefix = "test"
                max_prefix_len = 63 - len(suffix)
            prefix = prefix[:max_prefix_len].rstrip('_')
            return f"{prefix}{suffix}"[:63]

        conn = None
        try:
            # Connect to postgres database to create new database
            conn = self._connect_admin_database()

            with conn.cursor() as cursor:
                # Avoid hanging indefinitely if the template database is busy.
                # These can be overridden per environment as needed.
                try:
                    clone_stmt_timeout_ms = int(os.getenv("DB_CLONE_STATEMENT_TIMEOUT_MS", "60000"))
                except Exception:
                    clone_stmt_timeout_ms = 60000
                try:
                    clone_lock_timeout_ms = int(os.getenv("DB_CLONE_LOCK_TIMEOUT_MS", "10000"))
                except Exception:
                    clone_lock_timeout_ms = 10000

                if clone_stmt_timeout_ms > 0:
                    cursor.execute(f"SET statement_timeout = {clone_stmt_timeout_ms}")
                if clone_lock_timeout_ms > 0:
                    cursor.execute(f"SET lock_timeout = {clone_lock_timeout_ms}")

                test_db_name = None
                last_error = None
                template_db = reference_db
                if (
                    reference_db == "postgres"
                    and os.getenv("DB_CLONE_POSTGRES_FROM_TEMPLATE0", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                ):
                    template_db = "template0"

                for _ in range(self.create_db_retries):
                    test_db_name = _make_db_name()
                    try:
                        cursor.execute(f"CREATE DATABASE {test_db_name} WITH TEMPLATE {template_db}")
                        last_error = None
                        break
                    except psycopg2.Error as e:
                        last_error = e
                        # Name collision (common when running multiple evals in parallel).
                        # Also treat lock/timeouts as retryable to avoid hanging or failing flakily.
                        if getattr(e, "pgcode", None) in {"42P04", "23505", "55P03", "57014", "55006"}:
                            continue
                        raise
                if last_error is not None:
                    raise last_error

            self.test_databases.add(test_db_name)
            setup_time = time.time() - start_time

            logger.info(f"Created test database {test_db_name} from {template_db} in {setup_time:.2f}s")
            return test_db_name, setup_time

        except Exception as e:
            logger.error(f"Failed to create test database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def cleanup_test_databases(
        self,
        db_names: Optional[Iterable[Optional[str]]] = None,
    ) -> List[Tuple[str, Exception]]:
        """Clean up tracked test databases or a specific list of database names."""
        if db_names is None:
            targets = list(self.test_databases)
        else:
            targets = list(dict.fromkeys(db_name for db_name in db_names if db_name))

        failures: List[Tuple[str, Exception]] = []
        for db_name in targets:
            try:
                self._cleanup_test_database_with_retry(db_name)
                logger.info(f"Dropped test database {db_name}")
                self.test_databases.discard(db_name)
            except Exception as e:
                failures.append((db_name, e))

        return failures

    def cleanup_test_database(self, db_name: str):
        """Clean up a single test database if it exists."""
        self._cleanup_test_database_with_retry(db_name)
        logger.info(f"Dropped test database {db_name}")
        self.test_databases.discard(db_name)


# =============================================================================
# Database Maintenance (merged from database_checker and database_utils)
# =============================================================================

class DatabaseMaintenance:
    """Database maintenance and verification utilities."""

    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or get_db_config()

    def verify_database(self, db_name: str, expected_counts: Dict[str, int]) -> bool:
        """Verify database has expected row counts."""
        conn = None
        try:
            conn = psycopg2.connect(**{**self.db_config, 'database': db_name})
            with conn.cursor() as cursor:
                for table, expected_count in expected_counts.items():
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    actual_count = cursor.fetchone()[0]

                    if actual_count != expected_count:
                        logger.error(f"Table {table}: expected {expected_count} rows, got {actual_count}")
                        return False

            return True
        except Exception as e:
            logger.error(f"Failed to verify database {db_name}: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def import_from_sql_dump(self, db_name: str, sql_file: str) -> bool:
        """Import database from SQL dump file."""
        import subprocess

        try:
            # Create database if not exists
            conn = psycopg2.connect(**{**self.db_config, 'database': 'postgres'})
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                cursor.execute(f"CREATE DATABASE {db_name}")
            conn.close()

            # Import SQL dump
            cmd = [
                'psql',
                '-h', self.db_config['host'],
                '-U', self.db_config['user'],
                '-d', db_name,
                '-f', sql_file
            ]

            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to import SQL dump: {result.stderr}")
                return False

            logger.info(f"Successfully imported {sql_file} into {db_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

def get_default_snapshot_manager() -> SnapshotManager:
    """Get a snapshot manager with default configuration."""
    return SnapshotManager(get_db_config())

def get_default_executor() -> SqlExecutor:
    """Get a SQL executor with default configuration."""
    return SqlExecutor(get_db_config())

def get_default_cloner() -> DatabaseCloner:
    """Get a database cloner with default configuration."""
    return DatabaseCloner(get_db_config())

def get_default_maintenance() -> DatabaseMaintenance:
    """Get database maintenance utilities with default configuration."""
    return DatabaseMaintenance(get_db_config())
