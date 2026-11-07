"""
Test Scenario Definition

Defines test scenarios for agent evaluation.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .database import StateModification
from .kb_sources import load_base_column_meanings
from ..logger import get_logger


logger = get_logger(__name__)


class SessionRequest:
    """Represents a single request within a multi-request session scenario."""

    def __init__(self, config: Dict[str, Any], scenario_dir: Path):
        """
        Initialize a session request.

        Args:
            config: Single element from the "requests" field in scenario.json
            scenario_dir: Directory containing scenario files
        """
        # Required fields
        if 'session_id' not in config:
            raise ValueError("SessionRequest requires 'session_id' field")
        if 'request' not in config:
            raise ValueError("SessionRequest requires 'request' field")

        self.session_id = config['session_id']
        self.conversation_id = config.get('conversation_id', self.session_id)
        self.request = config['request']
        self.scenario_dir = scenario_dir

        if not isinstance(self.request, str):
            raise ValueError("SessionRequest 'request' must be a string")

        # User knowledge for simulation
        self.user_knowledge = config.get('user_knowledge', [])
        self.validation_metadata = config.get("validation_metadata", {})
        if self.validation_metadata is not None and not isinstance(self.validation_metadata, dict):
            raise ValueError("SessionRequest 'validation_metadata' must be a JSON object if provided")
        self.author_notes = config.get("author_notes", {})
        if self.author_notes is not None and not isinstance(self.author_notes, dict):
            raise ValueError("SessionRequest 'author_notes' must be a JSON object if provided")
        self.security_context = config.get('security_context')
        if self.security_context is not None and not isinstance(self.security_context, dict):
            raise ValueError("SessionRequest 'security_context' must be a JSON object if provided")
        self.db_session = config.get('db_session')
        if self.db_session is not None and not isinstance(self.db_session, dict):
            raise ValueError("SessionRequest 'db_session' must be a JSON object if provided")
        raw_gold = config.get('gold_query', '')
        if 'gold_queries' in config:
            raw_gold = config.get('gold_queries') or []

        self.gold_queries: List[str] = []
        if isinstance(raw_gold, list):
            for q in raw_gold:
                if q is None:
                    continue
                text = str(q).strip()
                if text:
                    self.gold_queries.append(text)
            self.gold_query = "\n\n".join(self.gold_queries)
        else:
            self.gold_query = str(raw_gold or "")
            if self.gold_query.strip():
                self.gold_queries = [self.gold_query]

        # Evaluation criteria
        self.evaluation = config.get('evaluation', {})
        self.expected_action = self.evaluation.get('correct_action')
        acceptable_actions = self.evaluation.get('acceptable_actions')
        if acceptable_actions is None:
            self.acceptable_actions = [self.expected_action] if self.expected_action else []
        elif isinstance(acceptable_actions, list) and all(
            isinstance(action, str) for action in acceptable_actions
        ):
            self.acceptable_actions = acceptable_actions
        else:
            raise ValueError(
                "SessionRequest evaluation.acceptable_actions must be a string array if provided"
            )
        if self.expected_action and self.expected_action not in self.acceptable_actions:
            self.acceptable_actions.insert(0, self.expected_action)
        self.disallow_write_db = bool(self.evaluation.get("disallow_write_db", False))
        self.disallow_failed_write_db = bool(
            self.evaluation.get("disallow_failed_write_db", False)
        )
        allowed_failed_write_db_error_patterns = (
            self.evaluation.get("allowed_failed_write_db_error_patterns") or []
        )
        if not isinstance(allowed_failed_write_db_error_patterns, list) or not all(
            isinstance(pattern, str) for pattern in allowed_failed_write_db_error_patterns
        ):
            raise ValueError(
                "SessionRequest evaluation.allowed_failed_write_db_error_patterns must be a string array if provided"
            )
        self.allowed_failed_write_db_error_patterns = allowed_failed_write_db_error_patterns
        self.require_qa_correct = bool(self.evaluation.get("require_qa_correct", False))
        forbidden_sql_patterns = self.evaluation.get("forbidden_sql_patterns") or []
        if not isinstance(forbidden_sql_patterns, list) or not all(
            isinstance(pattern, str) for pattern in forbidden_sql_patterns
        ):
            raise ValueError(
                "SessionRequest evaluation.forbidden_sql_patterns must be a string array if provided"
            )
        self.forbidden_sql_patterns = forbidden_sql_patterns
        self.forbidden_sql_patterns_scope = self._normalize_forbidden_sql_patterns_scope(
            self.evaluation.get("forbidden_sql_patterns_scope", "trace")
        )
        raw_final_state_check = config.get("final_state_check")
        if raw_final_state_check is None:
            raw_final_state_check = self.evaluation.get("final_state_check")
        self.final_state_check = self._normalize_final_state_check(raw_final_state_check)
        self.timeout = self.evaluation.get('time_limit_seconds')
        # Execute requests must be verifiable by at least one of:
        #  - gold_query/gold_queries (result and/or state derivation)
        #  - expected_snapshot (state diff verification)
        #  - qa_questions (LLM-judge understanding verification)
        #  - final_state_check (post-agent read-only final state verification)
        has_expected_snapshot = bool(config.get("expected_snapshot"))
        has_qa = bool(config.get("qa_questions"))
        has_final_state_check = bool(self.final_state_check)
        if (
            self.expected_action == "execute"
            and not self.gold_queries
            and not has_expected_snapshot
            and not has_qa
            and not has_final_state_check
        ):
            raise ValueError(
                "SessionRequest with evaluation.correct_action='execute' requires at least one "
                "verification field: non-empty 'gold_query'/'gold_queries', 'expected_snapshot', "
                "'qa_questions', or 'final_state_check'."
            )

        # Expected snapshots for correctness evaluation
        self.expected_snapshot = self._load_expected_snapshot(config.get('expected_snapshot'))

        # Q&A for understanding evaluation (per request)
        self.qa_questions = config.get('qa_questions', [])
        self.expected_answers = config.get('expected_answers', [])
        self.snapshot_ignore_columns = self._normalize_snapshot_ignore_columns(
            config.get('snapshot_ignore_columns')
        )

    def _normalize_snapshot_ignore_columns(self, raw: Any) -> Dict[str, List[str]]:
        """Normalize optional table->columns snapshot comparison exclusions."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(
                "SessionRequest 'snapshot_ignore_columns' must be a JSON object "
                "mapping table names to column-name arrays"
            )

        normalized: Dict[str, List[str]] = {}
        for table_name, columns in raw.items():
            if not isinstance(table_name, str):
                raise ValueError("snapshot_ignore_columns table names must be strings")
            if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
                raise ValueError(
                    f"snapshot_ignore_columns for table '{table_name}' must be a string array"
                )
            normalized[table_name] = columns
        return normalized

    def _normalize_forbidden_sql_patterns_scope(self, raw: Any) -> str:
        """Normalize which SQL trace subset forbidden_sql_patterns should scan."""
        if raw is None:
            return "trace"
        if not isinstance(raw, str):
            raise ValueError(
                "SessionRequest evaluation.forbidden_sql_patterns_scope must be "
                "'trace' or 'write_only' if provided"
            )
        scope = raw.strip().lower()
        if scope not in {"trace", "write_only"}:
            raise ValueError(
                "SessionRequest evaluation.forbidden_sql_patterns_scope must be "
                "'trace' or 'write_only' if provided"
            )
        return scope

    def _normalize_final_state_check(self, raw: Any) -> List[Dict[str, Any]]:
        """
        Normalize optional post-agent final state checks.

        Accepted shapes:
          - "SELECT ..."
          - {"sql": "SELECT ...", "expected_rows": [...]}
          - {"checks": [{...}, ...]}
          - [{...}, ...]
        If expected_rows/expected_value are omitted, the SQL must return a truthy
        first value.
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            checks = [{"sql": raw}]
        elif isinstance(raw, dict):
            if "checks" in raw:
                if not isinstance(raw["checks"], list):
                    raise ValueError("SessionRequest final_state_check.checks must be an array")
                checks = raw["checks"]
            else:
                checks = [raw]
        elif isinstance(raw, list):
            checks = raw
        else:
            raise ValueError(
                "SessionRequest final_state_check must be a SQL string, object, or array"
            )

        normalized: List[Dict[str, Any]] = []
        for idx, check in enumerate(checks):
            if isinstance(check, str):
                sql = check.strip()
                item: Dict[str, Any] = {"sql": sql}
            elif isinstance(check, dict):
                sql = str(check.get("sql") or check.get("query") or "").strip()
                item = dict(check)
                item["sql"] = sql
                item.pop("query", None)
            else:
                raise ValueError(
                    f"SessionRequest final_state_check[{idx}] must be a SQL string or object"
                )
            if not sql:
                raise ValueError(f"SessionRequest final_state_check[{idx}] requires non-empty sql")

            if "expected_result" in item and "expected_rows" not in item:
                expected_result = item.pop("expected_result")
                if isinstance(expected_result, dict):
                    item["expected_rows"] = expected_result.get("rows", [])
                else:
                    item["expected_rows"] = expected_result
            if "expected" in item and "expected_rows" not in item and "expected_value" not in item:
                expected = item.pop("expected")
                if isinstance(expected, list):
                    item["expected_rows"] = expected
                else:
                    item["expected_value"] = expected
            if "expected_rows" in item and not isinstance(item["expected_rows"], list):
                raise ValueError(
                    f"SessionRequest final_state_check[{idx}].expected_rows must be an array"
                )
            if "order_matters" in item and not isinstance(item["order_matters"], bool):
                raise ValueError(
                    f"SessionRequest final_state_check[{idx}].order_matters must be boolean"
                )
            normalized.append(item)
        return normalized

    def _load_expected_snapshot(self, filename: Optional[str]) -> Optional[StateModification]:
        """Load expected snapshot for this request."""
        if not filename:
            return None
        file_path = self.scenario_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"expected_snapshot file not found for request in '{self.scenario_dir}': {file_path}"
            )
        try:
            return StateModification.load_from_json(file_path)
        except Exception as e:
            raise ValueError(
                f"Failed to parse expected_snapshot for request in '{self.scenario_dir}': {file_path}"
            ) from e


class TestScenario:
    """Represents a test scenario for database agent evaluation."""
    
    def __init__(self, scenario_id: str, dir: Path, config: Dict[str, Any]):
        """
        Initialize test scenario from JSON configuration.

        Args:
            scenario_id: Unique identifier
            dir: Folder containing scenario.json
            config: Scenario configuration
        """
        self.id = scenario_id
        self.dir = dir
        self.name = config.get('name', scenario_id)
        self.setup_sql = config.get('setup_sql')
        self.reference_database = config.get('reference_database')
        self._config = config  # Store full config for access

        db_options = config.get("database_options") or {}
        if not isinstance(db_options, dict):
            raise ValueError(
                f"database_options must be a JSON object for scenario '{scenario_id}'"
            )
        use_clone = db_options.get("use_clone", True)
        read_only = db_options.get("read_only", False)
        if not isinstance(use_clone, bool):
            raise ValueError(
                f"database_options.use_clone must be boolean for scenario '{scenario_id}'"
            )
        if not isinstance(read_only, bool):
            raise ValueError(
                f"database_options.read_only must be boolean for scenario '{scenario_id}'"
            )
        # Safety: direct execution against a shared reference DB must be read-only.
        if use_clone is False and read_only is False:
            raise ValueError(
                f"Scenario '{scenario_id}' sets database_options.use_clone=false but read_only=false. "
                "Direct execution requires read_only=true."
            )
        self.database_use_clone = use_clone
        self.database_read_only = read_only

        self.requests: List[SessionRequest] = []
        for req_config in config.get('requests', []):
            self.requests.append(SessionRequest(config=req_config, scenario_dir=dir))
        self.concurrent_operations = config.get('concurrent_operations', [])
        
        # Q&A for understanding evaluation
        self.qa_questions = config.get('qa_questions', [])
        self.expected_answers = config.get('expected_answers', [])

        # Domain knowledge files
        self.knowledge = self._load_knowledge_jsonl()
        self.column_meanings = self._load_column_meanings()

    def _load_knowledge_jsonl(self) -> Optional[List[Dict[str, Any]]]:
        """Load knowledge.jsonl from scenario directory."""
        file_path = self.dir / 'knowledge.jsonl'
        if not file_path.exists():
            return None

        try:
            knowledge_items = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        knowledge_items.append(json.loads(line))
            return knowledge_items
        except Exception:
            logger.exception("Failed to load knowledge.jsonl for scenario %s", self.id)
            return None

    def _load_column_meanings(self) -> Optional[Dict[str, Any]]:
        """Load and validate column_meanings.json from scenario directory."""
        file_path = self.dir / 'column_meanings.json'
        if not file_path.exists():
            raise FileNotFoundError(
                f"column_meanings.json is required for scenario '{self.id}': {file_path}"
            )

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to parse column_meanings.json for scenario '{self.id}': {file_path}"
            ) from e

        if not isinstance(data, dict) or not data:
            raise ValueError(
                f"column_meanings.json must be a non-empty JSON object for scenario '{self.id}': {file_path}"
            )

        base = load_base_column_meanings(self.reference_database)
        if base:
            missing = sorted(set(base) - set(data))
            if missing:
                preview = ", ".join(missing[:8])
                raise ValueError(
                    f"column_meanings.json for scenario '{self.id}' is missing {len(missing)} base keys "
                    f"from LiveSQLBench for reference_database='{self.reference_database}'. "
                    f"Examples: {preview}"
                )

        return data
