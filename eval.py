#!/usr/bin/env python
"""
Run all scenarios with the new Agent-Runner architecture.
"""

import os
import json
import time
import argparse
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from datetime import datetime, date, time as datetime_time, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import psycopg2


class ExtendedEncoder(json.JSONEncoder):
    """JSON encoder that handles non-JSON scalar values from SQL/tool traces."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, datetime_time):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return bytes(obj).hex()
        return super().default(obj)

# Load environment variables
load_dotenv()

from src.logger import get_logger, get_session_logger

logger = get_logger(__name__)

from src.benchmark import AgentRunner, RunnerConfig, create_default_registry
from src.agents.llm_agent import LLMAgent
from src.benchmark.database import SqlExecutor, DatabaseCloner, SnapshotManager, StateModification
from src.benchmark import TestScenario
from src.benchmark.evaluator import compare_query_results
from src.benchmark.agents.judge import JudgeAgent
from src.benchmark.agents.user import UserAgent, build_user_system_prompt
from src.config import get_database_config
from src.llm.factory import create_model
from src.llm.interface import Message, Role

ACTION_ONLY_VERIFICATION_DETAILS = "No expected_snapshot/gold_query - action type verified only"
DEFAULT_EFFICIENCY_MULTIPLIER = 5.0


def make_user_prompter(request_text: str, user_knowledge, user_model: str):
    """Build an LLM-based prompter for the ask_user tool."""
    system_prompt = build_user_system_prompt(
        request_text=request_text,
        user_knowledge=user_knowledge,
    )
    ua = UserAgent(model_name=user_model, system_prompt=system_prompt)

    def prompter(question: str, context: str = "") -> str:
        q = question or ""
        if context:
            q = f"{q}\n\nContext:\n{context}"
        return ua.answer(q)

    return prompter


def _build_session_context(
    previous_request_results,
    current_session_id: Any = None,
    current_conversation_id: Any = None,
):
    """Build a session context summary for multi-turn scenarios.

    This context is passed into the agent prompt on later turns so the model can
    reference prior turn requests, decisions, and key tool evidence.
    """
    if not previous_request_results:
        return None

    if current_conversation_id is not None:
        previous_request_results = [
            r for r in previous_request_results
            if r.get("conversation_id") == current_conversation_id
        ]
        if not previous_request_results:
            return None
    elif current_session_id is not None:
        previous_request_results = [
            r for r in previous_request_results
            if r.get("session_id") == current_session_id
        ]
        if not previous_request_results:
            return None

    lines = ["Previous turns in this session (most recent last):"]
    max_turns = _get_positive_int_env("EVAL_SESSION_CONTEXT_MAX_TURNS", 3)
    max_queries = _get_positive_int_env("EVAL_SESSION_CONTEXT_MAX_QUERIES", 5)
    sql_chars = _get_positive_int_env("EVAL_SESSION_CONTEXT_SQL_CHARS", 400)
    reasoning_chars = _get_positive_int_env("EVAL_SESSION_CONTEXT_REASONING_CHARS", 800)
    evidence_items = _get_positive_int_env("EVAL_SESSION_CONTEXT_MAX_ITEMS", 40)

    for r in previous_request_results[-max_turns:]:
        request_index = r.get("request_index")
        request_text = r.get("request") or ""
        actual_action = r.get("actual_action") or ""
        rejection_reason = r.get("rejection_reason") or ""

        lines.append("")
        lines.append(f"TURN {request_index}:")
        lines.append("User request:")
        lines.append(str(request_text))
        if actual_action:
            lines.append(f"Agent action: {actual_action}")
        if rejection_reason:
            lines.append(f"Agent rejection_reason: {str(rejection_reason)[:400]}")

        final_queries = r.get("final_queries") or []
        if final_queries:
            lines.append("Agent SQL queries:")
            for q in final_queries[:max_queries]:
                sql = q.get("sql", q) if isinstance(q, dict) else q
                lines.append(f"- {str(sql)[:sql_chars]}")

        history_log = r.get("history") or []
        if history_log:
            finish_reasoning = ""
            for step in reversed(history_log):
                if step.get("action_type") == "finish":
                    finish_reasoning = str(step.get("reasoning") or "").strip()
                    break
            if finish_reasoning:
                lines.append("Agent response (final reasoning):")
                lines.append(finish_reasoning[:reasoning_chars])

            lines.append("Tool history (summarized):")
            lines.append(_summarize_tool_evidence(history_log, max_items=evidence_items))

    return "\n".join(lines).strip()


def _detect_runtime_failure(result: Dict[str, Any]) -> Optional[str]:
    """
    Separate infrastructure/model-call failures from scenario-level rejections.

    The agent currently maps provider exceptions into a FINISH+reject response.
    Those failures should not be scored as successful/unsuccessful task behavior
    or passed into the Q&A judge.
    """
    runtime_markers = (
        "LLM error:",
        "Native provider returned no tool call after protocol retry",
        "Native provider response contained no tool call",
        "provider/tool-protocol artifact",
    )

    for key in ("reason", "rejection_reason"):
        reason = result.get(key)
        if isinstance(reason, str) and any(marker in reason for marker in runtime_markers):
            return reason

    for step in result.get("history", []) or []:
        if isinstance(step, dict):
            step_reasoning = step.get("reasoning")
        else:
            step_reasoning = getattr(step, "reasoning", None)
        if isinstance(step_reasoning, str) and any(marker in step_reasoning for marker in runtime_markers):
            return step_reasoning

    return None


def _summarize_tool_evidence(history_log, max_items: int = 25) -> str:
    """
    Summarize tool calls and outputs for Q&A answering.

    We prefer the most-recent evidence so Q&A sees the final state and clarifications.
    However, some authorization-critical checks (e.g., validating access_key/role in
    clinicians) can happen early. If the agent performs many exploratory queries, a
    naive "take the most recent N items" policy can omit these checks and make the
    understanding rubric fail even when the agent complied.
    """
    if not history_log:
        return "None"

    def _format_sql_line(sql: str, row_count: Any, output: Any) -> str:
        sql_limit = _get_positive_int_env("EVAL_EVIDENCE_SQL_CHARS", 0)
        if sql_limit and len(str(sql)) > sql_limit:
            sql = str(sql)[:sql_limit] + "...[truncated]"
        line = f"  SQL: {sql}"
        if row_count is not None:
            line += f" (rows={row_count})"
        if isinstance(output, list) and output:
            # If the result set is small, include all rows so QA can recover exact answers.
            if len(output) <= 5:
                line += f" samples={output}"
            else:
                line += f" samples={output[:5]} ... ({len(output)} rows total)"
        return line

    def _is_critical_sql(sql: str) -> bool:
        s = (sql or "").lower()
        if "from clinicians" in s and ("access_key" in s or "role" in s or "fac_connect" in s):
            return True
        if "r_sec_" in s and any(marker in s for marker in ("assignment", "released_at", "coord_key", "principal")):
            return True
        return False

    def _add_unique(dst: List[str], seen: set, text: str) -> None:
        if not text or text in seen:
            return
        dst.append(text)
        seen.add(text)

    seen: set = set()
    items: List[str] = []

    # 1) Always include a small amount of authorization-critical evidence (most recent first).
    # This prevents it from being crowded out by lots of schema exploration.
    critical: List[str] = []
    critical_seen: set = set()
    critical_cap = max(6, min(10, max_items // 2))

    for step in reversed(history_log):
        if len(critical) >= critical_cap:
            break
        action = step.get("action_name") or ""
        tool_result = step.get("tool_result") or {}
        sub_results = tool_result.get("sub_results") or []

        if action == "ask_user":
            for sr in (sub_results or [])[:1]:
                sr_input = sr.get("input") or {}
                question = sr_input.get("question") or ""
                output = sr.get("output")
                answer = ""
                if isinstance(output, dict):
                    answer = str(output.get("answer") or "")
                    question = str(output.get("question") or question)
                elif output is not None:
                    answer = str(output)
                if question:
                    _add_unique(critical, critical_seen, f"  Q: {question}")
                if answer:
                    _add_unique(critical, critical_seen, f"  A: {answer}")
            continue

        for sr in sub_results:
            if not sr.get("success"):
                continue
            sr_input = sr.get("input") or {}
            sql = sr_input.get("sql")
            if not sql or not _is_critical_sql(sql):
                continue
            _add_unique(
                critical,
                critical_seen,
                _format_sql_line(sql, sr.get("row_count"), sr.get("output")),
            )
            if len(critical) >= critical_cap:
                break

    for text in reversed(critical):
        _add_unique(items, seen, text)
        if len(items) >= max_items:
            break

    if len(items) >= max_items:
        return "\n".join(items[:max_items])

    # 2) Fill remaining space with most-recent evidence for recency.
    for step in reversed(history_log):
        if len(items) >= max_items:
            break

        action = step.get("action_name") or ""
        tool_result = step.get("tool_result") or {}
        summary = tool_result.get("summary") or ""
        sub_results = tool_result.get("sub_results") or []

        if action and summary:
            _add_unique(items, seen, f"- {action}: {summary}")
            if len(items) >= max_items:
                break

        if action == "ask_user":
            for sr in (sub_results or [])[:1]:
                sr_input = sr.get("input") or {}
                question = sr_input.get("question") or ""
                output = sr.get("output")
                answer = ""
                if isinstance(output, dict):
                    answer = str(output.get("answer") or "")
                    question = str(output.get("question") or question)
                elif output is not None:
                    answer = str(output)
                if question:
                    _add_unique(items, seen, f"  Q: {question}")
                if answer:
                    _add_unique(items, seen, f"  A: {answer}")
            continue

        for sr in sub_results:
            if not sr.get("success"):
                continue
            sr_input = sr.get("input") or {}
            sql = sr_input.get("sql")
            if not sql:
                continue
            _add_unique(
                items,
                seen,
                _format_sql_line(sql, sr.get("row_count"), sr.get("output")),
            )
            if len(items) >= max_items:
                break

    return "\n".join(items[:max_items]) if items else "None"


def _format_full_trace_for_qa(history_log: List[Dict[str, Any]]) -> str:
    """Serialize the tested agent's full trace for follow-up Q&A.

    The Q&A step is meant to ask the agent under evaluation what it observed or
    decided. It should not depend on a lossy evidence sampler; otherwise a
    correct trace can be turned into a wrong Q&A answer by omitted rows.
    """
    if not history_log:
        return "[]"

    trace: List[Dict[str, Any]] = []
    for step in history_log:
        trace.append({
            "step_id": step.get("step_id"),
            "action_type": step.get("action_type"),
            "action_name": step.get("action_name"),
            "reasoning": step.get("reasoning"),
            "action_params": step.get("action_params"),
            "tool_result": step.get("tool_result"),
        })

    text = json.dumps(trace, indent=2, default=str)
    max_chars = _get_positive_int_env("EVAL_QA_TRACE_MAX_CHARS", 0)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n...[trace truncated by EVAL_QA_TRACE_MAX_CHARS]"
    return text


def _last_finish_reasoning(history_log: List[Dict[str, Any]]) -> str:
    """Return the tested agent's final reasoning, if it is present in the trace."""
    for step in reversed(history_log or []):
        if step.get("action_type") == "finish":
            return str(step.get("reasoning") or "").strip()
    return ""


def _is_select_like_sql(sql: str) -> bool:
    """Heuristic: returns True for SELECT/CTE queries that produce row results."""
    if not sql:
        return False
    s = sql.lstrip().lower()
    while s.startswith("("):
        s = s[1:].lstrip()
    return s.startswith("select") or s.startswith("with")


def _normalize_sql_statements(sql: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize SQL input into a list of non-empty statements."""
    if sql is None:
        return []
    if isinstance(sql, list):
        statements: List[str] = []
        for s in sql:
            if s is None:
                continue
            text = str(s).strip()
            if text:
                statements.append(text)
        return statements
    s = str(sql).strip()
    return [s] if s else []


def _gold_order_matters(statements: List[str]) -> bool:
    """Heuristic: treat ORDER BY in the last SELECT/CTE as order-sensitive."""
    if not statements:
        return False
    candidate = None
    for stmt in reversed(statements):
        if _is_select_like_sql(stmt):
            candidate = stmt
            break
    if candidate is None:
        candidate = statements[-1]
    return "order by" in candidate.lower()


def _parse_bool_flag(value: Any) -> bool:
    """Parse JSON/env-style booleans without treating every non-empty string as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _efficiency_config(request: Any) -> Dict[str, Any]:
    """
    Return the relative-efficiency rule for a request.

    Scenarios opt in with:
      evaluation.efficiency_check: true
      evaluation.efficiency_multiplier: 5.0  # optional, defaults to 5x gold SQL
      evaluation.efficiency_absolute_slack_seconds: 0.001  # optional per-task override
    """
    evaluation = getattr(request, "evaluation", {}) or {}
    enabled = _parse_bool_flag(evaluation.get("efficiency_check", False))
    if not enabled:
        return {"enabled": False, "multiplier": DEFAULT_EFFICIENCY_MULTIPLIER}

    raw_multiplier = evaluation.get(
        "efficiency_multiplier",
        os.getenv("EVAL_EFFICIENCY_MULTIPLIER", str(DEFAULT_EFFICIENCY_MULTIPLIER)),
    )
    try:
        multiplier = float(raw_multiplier)
    except Exception:
        multiplier = DEFAULT_EFFICIENCY_MULTIPLIER
    if multiplier <= 0:
        multiplier = DEFAULT_EFFICIENCY_MULTIPLIER

    raw_required_indexes = evaluation.get("efficiency_required_indexes") or []
    if isinstance(raw_required_indexes, str):
        required_indexes = [raw_required_indexes]
    elif isinstance(raw_required_indexes, list):
        required_indexes = [str(v) for v in raw_required_indexes if str(v).strip()]
    else:
        required_indexes = []

    raw_forbidden_sql_patterns = evaluation.get("efficiency_forbidden_sql_patterns") or []
    if isinstance(raw_forbidden_sql_patterns, str):
        forbidden_sql_patterns = [raw_forbidden_sql_patterns]
    elif isinstance(raw_forbidden_sql_patterns, list):
        forbidden_sql_patterns = [str(v) for v in raw_forbidden_sql_patterns if str(v).strip()]
    else:
        forbidden_sql_patterns = []

    raw_required_relations = evaluation.get("efficiency_required_relations") or []
    if isinstance(raw_required_relations, str):
        required_relations = [raw_required_relations]
    elif isinstance(raw_required_relations, list):
        required_relations = [str(v) for v in raw_required_relations if str(v).strip()]
    else:
        required_relations = []

    return {
        "enabled": True,
        "multiplier": multiplier,
        "plan_gates_enabled": _parse_bool_flag(
            evaluation.get(
                "efficiency_plan_gates_enabled",
                os.getenv("EVAL_EFFICIENCY_PLAN_GATES_ENABLED", "false"),
            )
        ),
        "require_index": _parse_bool_flag(evaluation.get("efficiency_require_index", False)),
        "disallow_seq_scan": _parse_bool_flag(evaluation.get("efficiency_disallow_seq_scan", False)),
        "required_indexes": required_indexes,
        "required_index_policy": str(evaluation.get("efficiency_required_index_policy", "any")).strip().lower(),
        "required_relations": required_relations,
        "required_relation_policy": str(evaluation.get("efficiency_required_relation_policy", "any")).strip().lower(),
        "absolute_slack_seconds": _get_optional_float(evaluation.get("efficiency_absolute_slack_seconds")),
        "forbidden_sql_patterns": forbidden_sql_patterns,
    }


def _get_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _match_forbidden_sql_patterns(sql_text: str, patterns: List[str]) -> List[str]:
    lower_sql = sql_text.lower()
    normalized_sql = re.sub(r"\s+", " ", lower_sql)
    matched: List[str] = []
    for raw_pattern in patterns or []:
        pattern = str(raw_pattern).strip()
        if not pattern:
            continue
        if pattern.lower().startswith("regex:"):
            regex = pattern[len("regex:"):].strip()
            try:
                if re.search(regex, lower_sql, flags=re.IGNORECASE | re.DOTALL):
                    matched.append(pattern)
                continue
            except re.error as exc:
                matched.append(f"INVALID_REGEX:{pattern}:{exc}")
                continue
        normalized_pattern = re.sub(r"\s+", " ", pattern.lower()).strip()
        if normalized_pattern and normalized_pattern in normalized_sql:
            matched.append(str(raw_pattern))
    return matched


def _normalize_forbidden_sql_patterns_scope(scope: Optional[str]) -> str:
    value = (scope or "trace").strip().lower()
    return value if value in {"trace", "write_only"} else "trace"


def _sql_texts_from_action_params(params: Any) -> List[str]:
    if not isinstance(params, dict):
        return []
    sql_texts: List[str] = []
    raw_queries = params.get("queries")
    if isinstance(raw_queries, list):
        for query in raw_queries:
            if isinstance(query, dict) and query.get("sql"):
                sql_texts.append(str(query["sql"]))
            elif isinstance(query, str):
                sql_texts.append(query)
    elif params.get("sql"):
        sql_texts.append(str(params["sql"]))
    return sql_texts


def _strip_sql_leading_noise(sql_text: str) -> str:
    s = str(sql_text or "").lstrip()
    while True:
        stripped = s
        if stripped.startswith("--"):
            newline = stripped.find("\n")
            s = "" if newline == -1 else stripped[newline + 1 :].lstrip()
        elif stripped.startswith("/*"):
            end = stripped.find("*/")
            s = "" if end == -1 else stripped[end + 2 :].lstrip()
        elif stripped.startswith("("):
            s = stripped[1:].lstrip()
        else:
            return stripped


def _is_mutating_sql(sql_text: str) -> bool:
    """
    Heuristic for deciding whether a write_db statement mutates database state.

    This intentionally treats plain SELECT/CTE diagnostics as non-mutating so
    write_only forbidden-pattern checks do not punish reads such as SELECT MAX(...).
    """
    s = _strip_sql_leading_noise(sql_text).rstrip().rstrip(";").lower()
    if not s:
        return False
    if re.match(
        r"^(insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|"
        r"refresh|reindex|vacuum|analyze|comment|call|do|copy|lock)\b",
        s,
    ):
        return True
    if re.match(r"^select\b", s) and re.search(r"\binto\b", s):
        return True
    if re.search(
        r";\s*(insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|"
        r"refresh|reindex|vacuum|analyze|comment|call|do|copy|lock)\b",
        s,
    ):
        return True
    if s.startswith("with") and re.search(r"\b(insert|update|delete|merge)\b", s):
        return True
    return False


def _forbidden_sql_candidates(history_steps: List[Any], scope: str) -> List[Dict[str, str]]:
    """
    Return SQL text candidates scanned by evaluation.forbidden_sql_patterns.

    scope=trace preserves the original behavior and scans every SQL-bearing tool
    request. scope=write_only scans only mutating statements requested through
    write_db.
    """
    normalized_scope = _normalize_forbidden_sql_patterns_scope(scope)
    candidates: List[Dict[str, str]] = []
    for step in history_steps or []:
        action_name = getattr(step, "action_name", None)
        if normalized_scope == "write_only" and action_name != "write_db":
            continue
        for sql_text in _sql_texts_from_action_params(getattr(step, "action_params", None)):
            if normalized_scope == "write_only" and not _is_mutating_sql(sql_text):
                continue
            candidates.append({"tool": str(action_name or ""), "sql": sql_text})
    return candidates


def _evaluation_pattern_matches(pattern: str, sql_text: str) -> bool:
    raw_pattern = str(pattern)
    if raw_pattern.lower().startswith("regex:"):
        pattern = raw_pattern[len("regex:"):].strip()
    try:
        return bool(re.search(pattern, sql_text, flags=re.IGNORECASE | re.DOTALL))
    except re.error:
        return raw_pattern.lower() in sql_text.lower()


def _first_cell(rows: List[Any]) -> Any:
    if not rows:
        return None
    row = rows[0]
    if isinstance(row, dict):
        return next(iter(row.values()), None)
    if isinstance(row, (list, tuple)):
        return row[0] if row else None
    return row


def _truthy_sql_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, Decimal)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "t", "true", "yes", "y", "pass", "passed", "ok"}


def _values_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    return str(actual) == str(expected)


def _evaluate_final_state_checks(
    *,
    final_state_checks: Optional[List[Dict[str, Any]]],
    sql_executor: SqlExecutor,
    mismatch_log_config: Dict[str, Any],
    scenario_id: str,
    request_index: int,
) -> Tuple[Optional[bool], Optional[str]]:
    """Run optional read-only final-state checks after the agent finishes."""
    if not final_state_checks:
        return None, None

    for idx, check in enumerate(final_state_checks, start=1):
        sql_text = str(check.get("sql") or "").strip()
        label = str(check.get("description") or check.get("label") or f"#{idx}")
        if not sql_text:
            return False, f"Final state check {label} has empty SQL"

        result = sql_executor.execute(sql_text, allow_modify=False)
        if not result.get("success", False) or result.get("type") != "READ":
            detail = (
                f"Final state check {label} failed to run: "
                f"{result.get('error', 'query did not return rows')}"
            )
            if mismatch_log_config["enabled"]:
                logger.warning(
                    "Final state check failed to run: scenario=%s request_index=%s check=%s sql=%s error=%s",
                    scenario_id,
                    request_index,
                    label,
                    _truncate_text(sql_text, mismatch_log_config["sql_truncate"]),
                    result.get("error", "query did not return rows"),
                )
            return False, detail

        rows = result.get("full_data", [])
        if "expected_rows" in check:
            expected_rows = check.get("expected_rows") or []
            cmp = compare_query_results(
                actual_result={"rows": rows},
                expected_result={"rows": expected_rows},
                order_matters=bool(check.get("order_matters", _gold_order_matters([sql_text]))),
            )
            if not cmp.get("matches", False):
                reason = cmp.get("reason", "unknown")
                if mismatch_log_config["enabled"]:
                    logger.warning(
                        "Final state check mismatch: scenario=%s request_index=%s check=%s reason=%s details=%s sql=%s",
                        scenario_id,
                        request_index,
                        label,
                        reason,
                        json.dumps(
                            _summarize_query_mismatch_details(
                                cmp.get("details", {}),
                                mismatch_log_config["level"],
                            ),
                            default=str,
                        ),
                        _truncate_text(sql_text, mismatch_log_config["sql_truncate"]),
                    )
                return False, f"Final state check {label} mismatch: {reason}"
        elif "expected_value" in check:
            actual_value = _first_cell(rows)
            expected_value = check.get("expected_value")
            if not _values_match(actual_value, expected_value):
                return (
                    False,
                    f"Final state check {label} mismatch: expected {expected_value}, got {actual_value}",
                )
        elif not _truthy_sql_value(_first_cell(rows)):
            return False, f"Final state check {label} returned falsey result"

    plural = "s" if len(final_state_checks) != 1 else ""
    return True, f"{len(final_state_checks)} final state check{plural} passed"


def _walk_plan_nodes(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    out.append(
        {
            "node_type": node.get("Node Type"),
            "relation": node.get("Relation Name"),
            "cte_name": node.get("CTE Name"),
            "alias": node.get("Alias"),
            "index": node.get("Index Name"),
            "plan_rows": node.get("Plan Rows"),
        }
    )
    for child in node.get("Plans") or []:
        _walk_plan_nodes(child, out)


def _explain_plan_summary(sql_executor: SqlExecutor, sql: str) -> Dict[str, Any]:
    result = sql_executor.execute(f"EXPLAIN (FORMAT JSON) {sql}", allow_modify=False)
    if not result.get("success") or result.get("type") != "READ":
        return {
            "ok": False,
            "error": result.get("error", "EXPLAIN did not return a readable plan"),
        }

    data = result.get("full_data") or []
    if not data or not data[0]:
        return {"ok": False, "error": "EXPLAIN returned no plan rows"}

    plan_doc = data[0][0]
    if isinstance(plan_doc, str):
        plan_doc = json.loads(plan_doc)
    if isinstance(plan_doc, list):
        plan_doc = plan_doc[0]
    if not isinstance(plan_doc, dict) or "Plan" not in plan_doc:
        return {"ok": False, "error": "EXPLAIN returned an unexpected plan shape"}

    nodes: List[Dict[str, Any]] = []
    _walk_plan_nodes(plan_doc["Plan"], nodes)
    index_nodes = [
        node
        for node in nodes
        if "Index" in str(node.get("node_type") or "")
    ]
    seq_scan_nodes = [
        node
        for node in nodes
        if str(node.get("node_type") or "") == "Seq Scan"
    ]
    return {
        "ok": True,
        "node_types": [node.get("node_type") for node in nodes],
        "nodes": nodes,
        "index_nodes": index_nodes,
        "seq_scan_nodes": seq_scan_nodes,
    }


def _plan_relation_names(plan_summary: Dict[str, Any]) -> set[str]:
    return {
        str(node.get("relation")).strip().lower()
        for node in plan_summary.get("nodes") or []
        if str(node.get("relation") or "").strip()
    }


def _extract_write_queries(history_steps) -> List[Dict[str, str]]:
    """Extract write_db SQL statements from runner history for logging/Q&A."""
    out: List[Dict[str, str]] = []
    if not history_steps:
        return out

    for step in history_steps:
        if getattr(step, "action_name", None) != "write_db":
            continue
        tool_result = getattr(step, "tool_result", None)
        if not tool_result:
            continue
        for sr in getattr(tool_result, "sub_results", []) or []:
            sr_input = getattr(sr, "input", None) or {}
            sql = sr_input.get("sql")
            if not sql:
                continue
            out.append({"sql": sql, "purpose": sr_input.get("purpose", "")})
    return out


def _sum_successful_write_runtime_seconds(history_steps) -> Optional[float]:
    """Sum successful write_db SQL execution durations from runner history."""
    total_ms = 0.0
    seen = False
    for step in history_steps or []:
        if getattr(step, "action_name", None) != "write_db":
            continue
        tool_result = getattr(step, "tool_result", None)
        if not tool_result:
            continue
        for sr in getattr(tool_result, "sub_results", []) or []:
            if not getattr(sr, "success", False):
                continue
            duration_ms = getattr(sr, "duration_ms", None)
            if duration_ms is None:
                continue
            total_ms += float(duration_ms)
            seen = True
    return total_ms / 1000.0 if seen else None


def _successful_select_sql_candidates(history_steps) -> List[str]:
    """Return successful SELECT-like statements executed by the agent in chronological order."""
    candidates: List[str] = []
    if not history_steps:
        return candidates

    for step in history_steps:
        if getattr(step, "action_name", None) not in {"explore_database", "test_query", "write_db"}:
            continue
        tool_result = getattr(step, "tool_result", None)
        if not tool_result:
            continue
        sub_results = getattr(tool_result, "sub_results", None) or []
        for sr in sub_results:
            if not getattr(sr, "success", False):
                continue
            sr_input = getattr(sr, "input", None) or {}
            sql = sr_input.get("sql")
            if sql and _is_select_like_sql(sql):
                candidates.append(sql)
    return candidates


def _find_last_successful_select_sql(history_steps) -> Optional[str]:
    """Find the last successful SELECT-like statement executed by the agent."""
    candidates = _successful_select_sql_candidates(history_steps)
    return candidates[-1] if candidates else None


def _last_select_like_statement(statements: List[str]) -> Optional[str]:
    for stmt in reversed(statements or []):
        if _is_select_like_sql(stmt):
            return stmt
    return None


def _extract_final_queries(history_steps) -> List[Dict[str, str]]:
    """Extract successful agent SQL statements for logging/Q&A, including read SELECTs."""
    out: List[Dict[str, str]] = []
    if not history_steps:
        return out

    for step in history_steps:
        action_name = getattr(step, "action_name", None)
        if action_name not in {"explore_database", "test_query", "write_db"}:
            continue
        tool_result = getattr(step, "tool_result", None)
        if not tool_result:
            continue
        for sr in getattr(tool_result, "sub_results", []) or []:
            if not getattr(sr, "success", False):
                continue
            sr_input = getattr(sr, "input", None) or {}
            sql = sr_input.get("sql")
            if not sql:
                continue
            out.append(
                {
                    "tool": str(action_name),
                    "sql": sql,
                    "purpose": sr_input.get("purpose", ""),
                }
            )
    return out


def _find_matching_successful_select_sql(
    *,
    history_steps: List[Any],
    expected_output: Dict[str, Any],
    gold_statements: List[str],
    sql_executor: SqlExecutor,
) -> Dict[str, Any]:
    """
    Compare the agent's successful SELECT-like statements with gold output.

    Agents often run a correct result query and then issue narrow diagnostic
    reads to inspect a truncated stream or sample. Use the most recent matching
    result-shaped query for verification; if none match, report the most recent
    SELECT-like query so the failure still points at the last executed read.
    """
    candidates = _successful_select_sql_candidates(history_steps)
    if not candidates:
        return {"sql": None, "actual_read": None, "comparison": None, "matched": False}

    order_matters = _gold_order_matters(gold_statements)
    latest: Dict[str, Any] = {
        "sql": candidates[-1],
        "actual_read": None,
        "comparison": None,
        "matched": False,
    }
    for agent_sql in reversed(candidates):
        actual_read = sql_executor.execute(agent_sql, allow_modify=False)
        item: Dict[str, Any] = {
            "sql": agent_sql,
            "actual_read": actual_read,
            "comparison": None,
            "matched": False,
        }
        if not actual_read.get("success", False) or actual_read.get("type") != "READ":
            if agent_sql == candidates[-1]:
                latest = item
            continue
        cmp = compare_query_results(
            actual_result={"rows": actual_read.get("full_data", [])},
            expected_result=expected_output,
            order_matters=order_matters,
        )
        item["comparison"] = cmp
        item["matched"] = bool(cmp.get("matches", False))
        if agent_sql == candidates[-1]:
            latest = item
        if item["matched"]:
            return item
    return latest


def _evaluate_relative_efficiency(
    *,
    enabled: bool,
    multiplier: float,
    plan_gates_enabled: bool = False,
    require_index: bool = False,
    disallow_seq_scan: bool = False,
    required_indexes: Optional[List[str]] = None,
    required_index_policy: str = "any",
    required_relations: Optional[List[str]] = None,
    required_relation_policy: str = "any",
    absolute_slack_seconds: Optional[float] = None,
    forbidden_sql_patterns: Optional[List[str]] = None,
    gold_runtime_seconds: Optional[float],
    history_steps: List[Any],
    sql_executor: SqlExecutor,
    scenario_id: str,
    request_index: int,
    gold_expected_rows: Optional[List[Dict[str, Any]]] = None,
    gold_statements: Optional[List[str]] = None,
) -> Tuple[Optional[bool], Optional[str], Optional[Dict[str, Any]]]:
    """
    Evaluate SQL efficiency relative to gold SQL runtime.

    A request passes when the agent's SQL execution time is within
    `multiplier * gold_runtime_seconds`. For SELECT tasks we re-run the agent's
    final successful SELECT under the same session bindings; for write tasks we
    use the measured successful write_db statement durations from the run.
    """
    if not enabled:
        return None, None, None

    metrics: Dict[str, Any] = {
        "enabled": True,
        "multiplier": multiplier,
        "plan_gates_enabled": plan_gates_enabled,
        "require_index": require_index,
        "disallow_seq_scan": disallow_seq_scan,
        "required_indexes": required_indexes or [],
        "required_index_policy": required_index_policy,
        "required_relations": required_relations or [],
        "required_relation_policy": required_relation_policy,
        "absolute_slack_seconds": absolute_slack_seconds,
        "forbidden_sql_patterns": forbidden_sql_patterns or [],
        "gold_sql_seconds": gold_runtime_seconds,
        "agent_sql_seconds": None,
        "threshold_seconds": None,
        "passed": None,
    }

    if gold_runtime_seconds is None:
        metrics["passed"] = False
        detail = "Efficiency check failed: missing gold SQL runtime"
        return False, detail, metrics

    agent_sql = None
    if gold_expected_rows is not None:
        selected_read = _find_matching_successful_select_sql(
            history_steps=history_steps,
            expected_output={"rows": gold_expected_rows},
            gold_statements=gold_statements or [],
            sql_executor=sql_executor,
        )
        if selected_read.get("matched"):
            agent_sql = selected_read.get("sql")
    if not agent_sql:
        agent_sql = _find_last_successful_select_sql(history_steps)
    if plan_gates_enabled and agent_sql and forbidden_sql_patterns:
        matched_patterns = _match_forbidden_sql_patterns(agent_sql, forbidden_sql_patterns)
        if matched_patterns:
            metrics["passed"] = False
            metrics["forbidden_sql_patterns_matched"] = matched_patterns
            detail = (
                "Efficiency SQL-shape check failed: agent SQL used forbidden pattern(s) "
                f"{matched_patterns}"
            )
            return False, detail, metrics
    agent_runtime_seconds: Optional[float] = None
    if agent_sql:
        actual_read = sql_executor.execute(agent_sql, allow_modify=False)
        if actual_read.get("success", False) and actual_read.get("type") == "READ":
            agent_runtime_seconds = actual_read.get("time_elapse")
        else:
            metrics["passed"] = False
            detail = "Efficiency check failed: could not re-run agent SELECT"
            logger.warning(
                "Efficiency SELECT re-run failed: scenario=%s request_index=%s sql=%s error=%s",
                scenario_id,
                request_index,
                _truncate_text(agent_sql, 300),
                actual_read.get("error", "unknown"),
            )
            return False, detail, metrics
    else:
        agent_runtime_seconds = _sum_successful_write_runtime_seconds(history_steps)

    if agent_runtime_seconds is None:
        metrics["passed"] = False
        detail = "Efficiency check failed: missing agent SQL runtime"
        return False, detail, metrics

    enforced_require_index = False
    enforced_disallow_seq_scan = False
    enforced_required_indexes: List[str] = []
    enforced_required_relations: List[str] = []
    # Efficiency scoring is timing-only by default. Physical plan and SQL-shape
    # policies remain available for diagnostics or future gated runs, but they
    # are ignored unless EVAL_EFFICIENCY_PLAN_GATES_ENABLED or the request-level
    # efficiency_plan_gates_enabled flag explicitly turns them on.
    if plan_gates_enabled and agent_sql and (
        require_index
        or disallow_seq_scan
        or required_indexes
        or required_relations
    ):
        gold_sql_for_plan = _last_select_like_statement(gold_statements or [])
        gold_plan_summary = (
            _explain_plan_summary(sql_executor, gold_sql_for_plan)
            if gold_sql_for_plan
            else {"ok": False, "error": "No SELECT-like gold SQL available for plan gating"}
        )
        metrics["gold_plan"] = gold_plan_summary
        if not gold_plan_summary.get("ok"):
            metrics["passed"] = False
            detail = f"Efficiency gold plan check failed: {gold_plan_summary.get('error', 'unknown EXPLAIN error')}"
            return False, detail, metrics

        gold_uses_index = bool(gold_plan_summary.get("index_nodes"))
        gold_has_seq_scan = bool(gold_plan_summary.get("seq_scan_nodes"))
        gold_used_indexes = {
            str(node.get("index"))
            for node in gold_plan_summary.get("index_nodes") or []
            if node.get("index")
        }
        if require_index and not gold_uses_index:
            metrics["passed"] = False
            detail = "Efficiency gold plan check failed: gold SQL did not use an index node"
            return False, detail, metrics
        if disallow_seq_scan and gold_has_seq_scan:
            metrics["passed"] = False
            detail = "Efficiency gold plan check failed: gold SQL used Seq Scan"
            return False, detail, metrics
        required_index_policy = required_index_policy if required_index_policy in {"any", "all"} else "any"
        required_set = set(required_indexes or [])
        gold_missing_required = (
            required_indexes
            and (
                not required_set.issubset(gold_used_indexes)
                if required_index_policy == "all"
                else gold_used_indexes.isdisjoint(required_set)
            )
        )
        if gold_missing_required:
            metrics["passed"] = False
            detail = (
                "Efficiency gold plan check failed: gold SQL did not use the required index "
                f"{sorted(required_set)} with policy={required_index_policy}; "
                f"used {sorted(gold_used_indexes) if gold_used_indexes else 'no indexes'}"
            )
            return False, detail, metrics
        required_relation_policy = required_relation_policy if required_relation_policy in {"any", "all"} else "any"
        required_relation_set = {
            str(relation).strip().lower()
            for relation in required_relations or []
            if str(relation).strip()
        }
        gold_used_relations = _plan_relation_names(gold_plan_summary)
        gold_missing_required_relations = (
            required_relations
            and (
                not required_relation_set.issubset(gold_used_relations)
                if required_relation_policy == "all"
                else gold_used_relations.isdisjoint(required_relation_set)
            )
        )
        if gold_missing_required_relations:
            metrics["passed"] = False
            detail = (
                "Efficiency gold plan check failed: gold SQL did not use the required relation "
                f"{sorted(required_relation_set)} with policy={required_relation_policy}; "
                f"used {sorted(gold_used_relations) if gold_used_relations else 'no relations'}"
            )
            return False, detail, metrics
        enforced_require_index = bool(require_index)
        enforced_disallow_seq_scan = bool(disallow_seq_scan)
        enforced_required_indexes = list(required_indexes or [])
        enforced_required_relations = list(required_relations or [])

    if agent_sql and (
        enforced_require_index
        or enforced_disallow_seq_scan
        or enforced_required_indexes
        or enforced_required_relations
    ):
        plan_summary = _explain_plan_summary(sql_executor, agent_sql)
        metrics["agent_plan"] = plan_summary
        if not plan_summary.get("ok"):
            metrics["passed"] = False
            detail = f"Efficiency plan check failed: {plan_summary.get('error', 'unknown EXPLAIN error')}"
            return False, detail, metrics
        if enforced_require_index and not plan_summary.get("index_nodes"):
            metrics["passed"] = False
            detail = "Efficiency plan check failed: agent SQL plan did not use an index node"
            return False, detail, metrics
        if enforced_disallow_seq_scan and plan_summary.get("seq_scan_nodes"):
            metrics["passed"] = False
            relations = sorted(
                {
                    str(node.get("relation"))
                    for node in plan_summary.get("seq_scan_nodes") or []
                    if node.get("relation")
                }
            )
            suffix = f" on {', '.join(relations)}" if relations else ""
            detail = f"Efficiency plan check failed: agent SQL plan used Seq Scan{suffix}"
            return False, detail, metrics
        if enforced_required_indexes:
            used_indexes = {
                str(node.get("index"))
                for node in plan_summary.get("index_nodes") or []
                if node.get("index")
            }
            required_set = set(enforced_required_indexes)
            missing_required = (
                not required_set.issubset(used_indexes)
                if required_index_policy == "all"
                else used_indexes.isdisjoint(required_set)
            )
            if missing_required:
                metrics["passed"] = False
                detail = (
                    "Efficiency plan check failed: agent SQL plan did not use the required index "
                    f"{sorted(required_set)} with policy={required_index_policy}; "
                    f"used {sorted(used_indexes) if used_indexes else 'no indexes'}"
                )
                return False, detail, metrics
        if enforced_required_relations:
            used_relations = _plan_relation_names(plan_summary)
            required_relation_policy = required_relation_policy if required_relation_policy in {"any", "all"} else "any"
            required_relation_set = {
                str(relation).strip().lower()
                for relation in enforced_required_relations
                if str(relation).strip()
            }
            missing_required_relations = (
                not required_relation_set.issubset(used_relations)
                if required_relation_policy == "all"
                else used_relations.isdisjoint(required_relation_set)
            )
            if missing_required_relations:
                metrics["passed"] = False
                detail = (
                    "Efficiency plan check failed: agent SQL plan did not use the required relation "
                    f"{sorted(required_relation_set)} with policy={required_relation_policy}; "
                    f"used {sorted(used_relations) if used_relations else 'no relations'}"
                )
                return False, detail, metrics
    metrics["plan_enforcement"] = {
        "require_index": enforced_require_index,
        "disallow_seq_scan": enforced_disallow_seq_scan,
        "required_indexes": enforced_required_indexes,
        "required_relations": enforced_required_relations,
    }

    timing_slack_seconds = (
        float(absolute_slack_seconds)
        if absolute_slack_seconds is not None
        else _get_float_env("EVAL_EFFICIENCY_ABSOLUTE_SLACK_SECONDS", 0.001)
    )
    threshold = max(
        max(float(gold_runtime_seconds), 1e-9) * float(multiplier),
        float(gold_runtime_seconds) + max(0.0, timing_slack_seconds),
    )
    passed = float(agent_runtime_seconds) <= threshold
    metrics.update({
        "agent_sql_seconds": agent_runtime_seconds,
        "absolute_slack_seconds": max(0.0, timing_slack_seconds),
        "threshold_seconds": threshold,
        "passed": passed,
    })

    gold_ms = float(gold_runtime_seconds) * 1000
    agent_ms = float(agent_runtime_seconds) * 1000
    threshold_ms = threshold * 1000
    if passed:
        detail = (
            f"Efficiency check passed: agent SQL {agent_ms:.2f}ms <= "
            f"{multiplier:.2f}x gold SQL {gold_ms:.2f}ms ({threshold_ms:.2f}ms)"
        )
    else:
        detail = (
            f"Efficiency check failed: agent SQL {agent_ms:.2f}ms > "
            f"{multiplier:.2f}x gold SQL {gold_ms:.2f}ms ({threshold_ms:.2f}ms)"
        )
    return passed, detail, metrics


def _truncate_text(value: Optional[str], limit: int = 500) -> Optional[str]:
    """Trim long strings for logging."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_mismatch_level(value: Optional[str]) -> str:
    """Normalize mismatch log level from env values."""
    if value is None:
        return "basic"
    val = value.strip().lower()
    if val in {"off", "false", "no", "0"}:
        return "off"
    if val in {"detailed", "detail", "full", "verbose", "2"}:
        return "detailed"
    if val in {"basic", "summary", "true", "yes", "on", "1"}:
        return "basic"
    return "basic"


def _get_mismatch_log_config() -> Dict[str, Any]:
    """Load mismatch logging settings from environment."""
    raw_level = os.getenv("EVAL_MISMATCH_LOG_LEVEL")
    if raw_level is None or raw_level.strip() == "":
        legacy = os.getenv("EVAL_MISMATCH_LOGGING")
        raw_level = legacy if legacy is not None else "basic"
    level = _normalize_mismatch_level(raw_level)
    return {
        "level": level,
        "enabled": level != "off",
        "snapshot_detail": level == "detailed",
        "max_tables": 5,
        "max_rows": 3,
        "sql_truncate": 500 if level == "detailed" else 200,
    }


def _row_key(row: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Normalize a row dict into a stable, comparable key."""
    def _make_hashable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        try:
            hash(value)
            return value
        except TypeError:
            pass
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            return str(value)

    return tuple(sorted((k, _make_hashable(v)) for k, v in row.items()))


def _sample_counter_rows(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    """Return a small sample of row counts for logging."""
    samples: List[Dict[str, Any]] = []
    for row_key, count in counter.most_common(limit):
        samples.append({"row": dict(row_key), "count": count})
    return samples


def _diff_rows(actual_rows: List[Dict[str, Any]], expected_rows: List[Dict[str, Any]], max_rows: int) -> Optional[Dict[str, Any]]:
    """Compute a compact diff between two row lists with duplicate awareness."""
    actual_counter = Counter(_row_key(row) for row in actual_rows)
    expected_counter = Counter(_row_key(row) for row in expected_rows)
    extra = actual_counter - expected_counter
    missing = expected_counter - actual_counter
    if not extra and not missing:
        return None
    diff: Dict[str, Any] = {
        "actual_count": len(actual_rows),
        "expected_count": len(expected_rows),
    }
    if extra:
        diff["extra_rows"] = _sample_counter_rows(extra, max_rows)
    if missing:
        diff["missing_rows"] = _sample_counter_rows(missing, max_rows)
    return diff


def _diff_columns(actual_cols: Dict[str, str], expected_cols: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Summarize schema differences for a table."""
    actual_set = set(actual_cols)
    expected_set = set(expected_cols)
    actual_only = sorted(actual_set - expected_set)
    expected_only = sorted(expected_set - actual_set)
    type_mismatches = {
        col: {"actual": actual_cols[col], "expected": expected_cols[col]}
        for col in sorted(actual_set & expected_set)
        if actual_cols[col] != expected_cols[col]
    }
    if not actual_only and not expected_only and not type_mismatches:
        return None
    diff: Dict[str, Any] = {}
    if actual_only:
        diff["actual_only"] = actual_only
    if expected_only:
        diff["expected_only"] = expected_only
    if type_mismatches:
        diff["type_mismatches"] = type_mismatches
    return diff


def _summarize_snapshot_mismatch(
    actual: StateModification,
    expected: StateModification,
    max_tables: int = 5,
    max_rows: int = 3,
) -> Dict[str, Any]:
    """Generate a compact summary of snapshot differences for logging."""
    actual_tables = set(actual.tables.keys())
    expected_tables = set(expected.tables.keys())
    missing_tables = sorted(expected_tables - actual_tables)
    extra_tables = sorted(actual_tables - expected_tables)

    table_diffs: List[Dict[str, Any]] = []
    for table in sorted(actual_tables & expected_tables):
        actual_mod = actual.tables.get(table)
        expected_mod = expected.tables.get(table)
        if not actual_mod or not expected_mod:
            continue
        if actual_mod.compare_equal(expected_mod):
            continue

        diff: Dict[str, Any] = {"table": table}
        column_diff = _diff_columns(actual_mod.columns, expected_mod.columns)
        if column_diff:
            diff["column_diff"] = column_diff

        initial_diff = _diff_rows(
            actual_mod.initial_except_final,
            expected_mod.initial_except_final,
            max_rows,
        )
        if initial_diff:
            diff["initial_except_final"] = initial_diff

        final_diff = _diff_rows(
            actual_mod.final_except_initial,
            expected_mod.final_except_initial,
            max_rows,
        )
        if final_diff:
            diff["final_except_initial"] = final_diff

        if len(diff) > 1:
            table_diffs.append(diff)

    summary: Dict[str, Any] = {
        "missing_table_count": len(missing_tables),
        "extra_table_count": len(extra_tables),
        "mismatched_table_count": len(table_diffs),
    }
    if missing_tables:
        summary["missing_tables"] = missing_tables[:max_tables]
        if len(missing_tables) > max_tables:
            summary["missing_tables_truncated"] = len(missing_tables) - max_tables
    if extra_tables:
        summary["extra_tables"] = extra_tables[:max_tables]
        if len(extra_tables) > max_tables:
            summary["extra_tables_truncated"] = len(extra_tables) - max_tables
    if table_diffs:
        summary["table_diffs"] = table_diffs[:max_tables]
        if len(table_diffs) > max_tables:
            summary["table_diffs_truncated"] = len(table_diffs) - max_tables

    return summary


def _summarize_snapshot_mismatch_counts(
    actual: StateModification,
    expected: StateModification,
) -> Dict[str, Any]:
    """Summarize snapshot mismatch with counts only."""
    actual_tables = set(actual.tables.keys())
    expected_tables = set(expected.tables.keys())
    missing_tables = expected_tables - actual_tables
    extra_tables = actual_tables - expected_tables
    mismatched_tables = 0

    for table in actual_tables & expected_tables:
        actual_mod = actual.tables.get(table)
        expected_mod = expected.tables.get(table)
        if not actual_mod or not expected_mod:
            continue
        if not actual_mod.compare_equal(expected_mod):
            mismatched_tables += 1

    return {
        "missing_table_count": len(missing_tables),
        "extra_table_count": len(extra_tables),
        "mismatched_table_count": mismatched_tables,
    }


def _summarize_query_mismatch_details(details: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Trim query mismatch details based on log level."""
    if level == "detailed":
        return details
    summary: Dict[str, Any] = {}
    for key in ("actual_row_count", "expected_row_count", "order_matters"):
        if key in details:
            summary[key] = details[key]
    return summary


def get_db_config():
    """Get database configuration from shared config."""
    config = get_database_config()
    return {
        "host": config["host"],
        "port": int(config["port"]),
        "user": config["user"],
        "password": config["password"],
        "database": config["database"],
    }


def _get_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%s is < 1; using default %s", name, value, default)
        return default
    return value


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _get_runner_max_queries_per_call() -> int:
    return _get_positive_int_env("EVAL_MAX_QUERIES_PER_CALL", 5)


def _get_runner_total_budget() -> int:
    return _get_positive_int_env("EVAL_TOTAL_BUDGET", 20)


def _get_runner_max_steps() -> int:
    return _get_positive_int_env("EVAL_MAX_STEPS", 50)


def _eval_mode_env(name: str) -> str:
    return os.getenv(name, "").strip().lower()


def _compact_knowledge_for_eval(knowledge: Any) -> Any:
    """
    Optionally reduce large scenario knowledge assets before exposing them to an
    eval agent. This is opt-in because non-security scenarios may need the full
    formula/domain KB.
    """
    mode = _eval_mode_env("EVAL_COMPACT_KNOWLEDGE")
    if mode in {"", "0", "false", "no", "off"}:
        return knowledge
    if not isinstance(knowledge, list):
        return knowledge

    def is_security_entry(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        entry_type = str(entry.get("type", "")).lower()
        if "policy" in entry_type:
            return True
        text = " ".join(str(entry.get(k, "")) for k in ("knowledge", "description", "definition")).lower()
        security_terms = (
            "access",
            "authorization",
            "clearance",
            "hold",
            "independent",
            "k-anonymity",
            "privacy",
            "release",
            "role",
            "segregation",
            "suppress",
        )
        return any(term in text for term in security_terms)

    if mode == "policy":
        filtered = [
            entry for entry in knowledge
            if isinstance(entry, dict) and "policy" in str(entry.get("type", "")).lower()
        ]
    elif mode == "security":
        filtered = [entry for entry in knowledge if is_security_entry(entry)]
    else:
        logger.warning("Unknown EVAL_COMPACT_KNOWLEDGE=%r; using full knowledge", mode)
        return knowledge

    return filtered or knowledge


def _compact_semantics_for_eval(semantics: Any) -> Any:
    """
    Optionally keep only security-relevant column meanings. The current security
    scenarios depend on r_sec_* tables plus users/orders identifiers; calculation
    table semantics dominate prompt size but are not needed for these runs.
    """
    mode = _eval_mode_env("EVAL_COMPACT_SEMANTICS")
    if mode in {"", "0", "false", "no", "off"}:
        return semantics
    if not isinstance(semantics, dict):
        return semantics

    if mode.startswith("tables:"):
        raw_tables = mode.split(":", 1)[1]
        table_names = {part.strip() for part in raw_tables.split(",") if part.strip()}
    elif mode == "security":
        table_names = {"users", "orders"}
    else:
        logger.warning("Unknown EVAL_COMPACT_SEMANTICS=%r; using full semantics", mode)
        return semantics

    filtered: Dict[str, Any] = {}
    for key, value in semantics.items():
        key_text = str(key)
        key_parts = key_text.split("|")
        table_name = key_parts[1] if len(key_parts) >= 3 else key_text
        if table_name.startswith("r_sec_") or table_name in table_names:
            filtered[key] = value

    return filtered or semantics


def load_scenario(scenario_path: Path) -> TestScenario:
    """Load a scenario from a directory path."""
    scenario_file = scenario_path / "scenario.json"

    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_file}")

    with open(scenario_file, 'r') as f:
        data = json.load(f)

    # Prefer the scenario's relative path under `scenarios/` as its unique ID.
    #
    # With nested layouts like `scenarios/<category>/<case>/<scenario_id>/`, the
    # leaf folder (and scenario.json `id`) may be a small identifier (e.g. "1"),
    # which is not unique across the whole repo.
    try:
        scenario_id = str(scenario_path.relative_to(Path("scenarios")))
    except Exception:
        scenario_id = data.get('id', scenario_path.name)
    return TestScenario(scenario_id=scenario_id, dir=scenario_path, config=data)


@dataclass
class GoldDerivation:
    """
    Outputs from the "gold SQL" derivation step.

    Some scenarios provide a gold query/queries that encode the intended correct behavior.
    We may run that SQL on a separate database clone to derive:
      - expected output rows (for read-only / SELECT-like gold SQL)
      - expected state modifications (via SnapshotManager)

    Any scratch DBs created here must be cleaned up by the caller.
    """

    expected_modification: Optional[StateModification]
    expected_rows: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    gold_db_name: Optional[str]
    gold_snapshot_db_name: Optional[str]
    gold_runtime_seconds: Optional[float]


def _log_cleanup_failures(context: str, failures: List[Tuple[str, Exception]]) -> None:
    """Log cleanup failures once per cleanup phase."""
    if not failures:
        return

    details = "; ".join(f"{db_name}: {error}" for db_name, error in failures)
    logger.warning("Failed to cleanup %s: %s", context, details)


def _apply_request_timeout(sql_executor: SqlExecutor, request: Any) -> None:
    """
    Apply per-request statement timeout on the shared SqlExecutor.

    Scenario configs often express `timeout` in seconds; SqlExecutor expects milliseconds.
    """
    timeout = getattr(request, "timeout", None)
    if timeout:
        try:
            sql_executor.timeout = int(float(timeout) * 1000)
        except Exception:
            sql_executor.timeout = None
    else:
        sql_executor.timeout = None


def _build_request_context(
    *,
    scenario: TestScenario,
    request: Any,
    sql_executor: SqlExecutor,
    model_name: str,
    aux_model: str,
) -> Dict[str, Any]:
    """
    Build the Runner tool context for one request.

    Notes:
      - knowledge_loader / semantics_loader are lambdas so the tool can fetch them on demand.
      - user_prompter is an LLM-backed helper used by the `ask_user` tool.
    """
    scenario_knowledge = _compact_knowledge_for_eval(scenario.knowledge)
    scenario_semantics = _compact_semantics_for_eval(scenario.column_meanings)
    return {
        "sql_executor": sql_executor,
        "knowledge_loader": lambda: scenario_knowledge if scenario_knowledge else "No business rules",
        "semantics_loader": lambda: scenario_semantics if scenario_semantics else "No raw table/column meanings",
        "user_prompter": make_user_prompter(
            request_text=request.request,
            user_knowledge=getattr(request, "user_knowledge", []),
            user_model=aux_model,
        ),
    }


def _make_request_sql_executor(base_sql_executor: SqlExecutor, request: Any) -> SqlExecutor:
    """Build a per-request executor carrying timeout plus optional DB session bindings."""
    return SqlExecutor(
        base_sql_executor.db_config,
        timeout=base_sql_executor.timeout,
        db_session=getattr(request, "db_session", None),
    )


def _resolve_request_session_bindings(
    *,
    request: Any,
    session_state: Dict[Any, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Resolve effective security_context and db_session for this request.

    If a request omits either field, inherit the most recent value from the same
    session_id. This gives multi-turn scenarios a lightweight notion of session
    continuity without requiring every turn to repeat the same principal/binding.
    """
    request_session_id = getattr(request, "session_id", None)
    prior = session_state.get(request_session_id, {}) if request_session_id is not None else {}

    security_context = getattr(request, "security_context", None)
    if security_context is None:
        security_context = prior.get("security_context")

    db_session = getattr(request, "db_session", None)
    if db_session is None:
        db_session = prior.get("db_session")

    if request_session_id is not None:
        session_state[request_session_id] = {
            "security_context": security_context,
            "db_session": db_session,
        }

    return security_context, db_session


def _derive_gold_expectations(
    *,
    expected_action: Optional[str],
    gold_statements: List[str],
    expected_modification: Optional[StateModification],
    scenario_id: str,
    request_session_id: Any,
    cloner: DatabaseCloner,
    base_db_config: Dict[str, Any],
    reference_db: str,
    sql_executor: SqlExecutor,
    allow_db_clone: bool,
    read_only: bool,
    read_select_gold_in_place: bool = False,
    derive_state_modification_from_gold: bool = True,
) -> GoldDerivation:
    """
    Derive expected output/modifications from the scenario's gold SQL, if present.

    There are three common modes:
      1) expected_modification is missing => run gold SQL on a fresh clone and snapshot diffs.
      2) expected_modification exists     => only need READ output rows for result comparison.
      3) final-state-only verification   => keep gold SQL as authoring reference, but do not
         derive an exact snapshot diff from write statements.

    For (2), we try to run a pure SELECT gold SQL against the agent DB in read-only mode to
    avoid an extra DB clone; if that fails, we fall back to running it on a fresh clone.
    """
    gold_db_name: Optional[str] = None
    gold_snapshot_db_name: Optional[str] = None
    gold_expected_rows: Optional[List[Dict[str, Any]]] = None
    gold_derivation_error: Optional[str] = None
    gold_runtime_seconds: Optional[float] = None

    # Only execute gold derivation for expected "execute" actions.
    if expected_action != "execute" or not gold_statements:
        return GoldDerivation(
            expected_modification=expected_modification,
            expected_rows=gold_expected_rows,
            error=gold_derivation_error,
            gold_db_name=gold_db_name,
            gold_snapshot_db_name=gold_snapshot_db_name,
            gold_runtime_seconds=gold_runtime_seconds,
        )

    # Efficiency matrix grouped runs evaluate each model on an isolated child
    # cloned from a prepared scenario base DB. For SELECT gold SQL, derive
    # expected rows directly against that child DB so the matrix runner does not
    # create one extra gold clone per model.
    if (
        read_select_gold_in_place
        and expected_modification is None
        and all(_is_select_like_sql(s) for s in gold_statements)
    ):
        gold_read = sql_executor.execute(gold_statements, allow_modify=False)
        gold_runtime_seconds = gold_read.get("time_elapse")
        if gold_read.get("success", False) and gold_read.get("type") == "READ":
            gold_expected_rows = gold_read.get("full_data", [])
        else:
            gold_derivation_error = (
                f"Gold query execution failed in in-place read mode: "
                f"{gold_read.get('error', 'Unknown error')}"
            )
        return GoldDerivation(
            expected_modification=None,
            expected_rows=gold_expected_rows,
            error=gold_derivation_error,
            gold_db_name=None,
            gold_snapshot_db_name=None,
            gold_runtime_seconds=gold_runtime_seconds,
        )

    if not derive_state_modification_from_gold and expected_modification is None:
        if all(_is_select_like_sql(s) for s in gold_statements):
            gold_read = sql_executor.execute(gold_statements, allow_modify=False)
            gold_runtime_seconds = gold_read.get("time_elapse")
            if gold_read.get("success", False) and gold_read.get("type") == "READ":
                gold_expected_rows = gold_read.get("full_data", [])
            else:
                gold_derivation_error = (
                    f"Gold query execution failed in final-state-only mode: "
                    f"{gold_read.get('error', 'Unknown error')}"
                )
        return GoldDerivation(
            expected_modification=None,
            expected_rows=gold_expected_rows,
            error=gold_derivation_error,
            gold_db_name=None,
            gold_snapshot_db_name=None,
            gold_runtime_seconds=gold_runtime_seconds,
        )

    # Read-only runs never need snapshot-based verification of state changes.
    # For SELECT-like gold SQL, derive expected rows directly on the agent DB without cloning.
    if read_only and expected_modification is None and all(_is_select_like_sql(s) for s in gold_statements):
        gold_read = sql_executor.execute(gold_statements, allow_modify=False)
        gold_runtime_seconds = gold_read.get("time_elapse")
        if gold_read.get("success", False) and gold_read.get("type") == "READ":
            gold_expected_rows = gold_read.get("full_data", [])
        else:
            gold_derivation_error = (
                f"Gold query execution failed in read-only mode: {gold_read.get('error', 'Unknown error')}"
            )
        return GoldDerivation(
            expected_modification=None,
            expected_rows=gold_expected_rows,
            error=gold_derivation_error,
            gold_db_name=None,
            gold_snapshot_db_name=None,
            gold_runtime_seconds=gold_runtime_seconds,
        )

    def _execute_gold_on_fresh_clone() -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[float]]:
        """
        Run gold statements on an isolated clone, returning:
          (rows_if_read, error_if_failed, runtime_seconds)
        """
        if not allow_db_clone:
            return None, "Gold derivation requires cloning but cloning is disabled for this scenario", None
        nonlocal gold_db_name
        gold_db_name, _ = cloner.create_test_database(
            scenario_name=f"{scenario_id}_req{request_session_id}_gold",
            reference_db=reference_db,
        )
        gold_db_config = {**base_db_config, "database": gold_db_name}
        gold_executor = SqlExecutor(
            gold_db_config,
            timeout=sql_executor.timeout,
            db_session=getattr(sql_executor, "db_session", None),
        )

        gold_exec = gold_executor.execute(gold_statements, allow_modify=True)
        runtime_seconds = gold_exec.get("time_elapse")
        if not gold_exec.get("success", False):
            err = gold_exec.get("error", "Unknown error")
            return None, f"Gold query execution failed: {err}", runtime_seconds
        if gold_exec.get("type") == "READ":
            return gold_exec.get("full_data", []), None, runtime_seconds
        return None, None, runtime_seconds

    if expected_modification is None:
        if not allow_db_clone:
            gold_derivation_error = "Gold derivation requires cloning but cloning is disabled for this scenario"
            return GoldDerivation(
                expected_modification=None,
                expected_rows=None,
                error=gold_derivation_error,
                gold_db_name=None,
                gold_snapshot_db_name=None,
                gold_runtime_seconds=gold_runtime_seconds,
            )
        # Gold defines the expected modification; snapshot must be taken on a separate DB.
        gold_db_name, _ = cloner.create_test_database(
            scenario_name=f"{scenario_id}_req{request_session_id}_gold",
            reference_db=reference_db,
        )
        gold_db_config = {**base_db_config, "database": gold_db_name}
        gold_executor = SqlExecutor(
            gold_db_config,
            timeout=sql_executor.timeout,
            db_session=getattr(sql_executor, "db_session", None),
        )

        gold_snapshot = SnapshotManager(
            f"{scenario_id}_req{request_session_id}_gold",
            gold_db_config,
        )
        gold_snapshot_db_name = gold_snapshot.init_snapshot_db

        gold_result = gold_executor.execute(gold_statements, allow_modify=True)
        gold_runtime_seconds = gold_result.get("time_elapse")
        if not gold_result.get("success", False):
            gold_derivation_error = f"Gold query execution failed: {gold_result.get('error', 'Unknown error')}"
        else:
            if gold_result.get("type") == "READ":
                gold_expected_rows = gold_result.get("full_data", [])
            expected_modification = gold_snapshot.get_modification()
    else:
        # expected_modification already exists; optionally derive expected READ output for comparisons.
        if len(gold_statements) == 1 and _is_select_like_sql(gold_statements[0]):
            gold_read = sql_executor.execute(gold_statements, allow_modify=False)
            gold_runtime_seconds = gold_read.get("time_elapse")
            if gold_read.get("success", False) and gold_read.get("type") == "READ":
                gold_expected_rows = gold_read.get("full_data", [])
            else:
                gold_expected_rows, gold_derivation_error, gold_runtime_seconds = _execute_gold_on_fresh_clone()
        else:
            gold_expected_rows, gold_derivation_error, gold_runtime_seconds = _execute_gold_on_fresh_clone()

    return GoldDerivation(
        expected_modification=expected_modification,
        expected_rows=gold_expected_rows,
        error=gold_derivation_error,
        gold_db_name=gold_db_name,
        gold_snapshot_db_name=gold_snapshot_db_name,
        gold_runtime_seconds=gold_runtime_seconds,
    )


def _maybe_init_agent_snapshot(
    *,
    expected_modification: Optional[StateModification],
    scenario_id: str,
    request_session_id: Any,
    db_config: Dict[str, Any],
) -> Tuple[Optional[SnapshotManager], Optional[str]]:
    """
    Initialize an agent-side snapshot baseline when we have an expected modification to compare against.

    Returns:
      (snapshot_manager_or_none, snapshot_db_name_or_none)
    """
    if expected_modification is None:
        return None, None
    agent_snapshot = SnapshotManager(f"{scenario_id}_req{request_session_id}", db_config)
    return agent_snapshot, agent_snapshot.init_snapshot_db


def _evaluate_expected_action(
    *,
    expected_action: Optional[str],
    acceptable_actions: Optional[List[str]],
    disallow_write_db: bool,
    disallow_failed_write_db: bool,
    allowed_failed_write_db_error_patterns: Optional[List[str]],
    forbidden_sql_patterns: Optional[List[str]],
    forbidden_sql_patterns_scope: str,
    actual_action: str,
    history_steps: List[Any],
    expected_modification: Optional[StateModification],
    agent_snapshot: Optional[SnapshotManager],
    snapshot_ignore_columns: Optional[Dict[str, List[str]]],
    final_state_checks: Optional[List[Dict[str, Any]]],
    gold_expected_rows: Optional[List[Dict[str, Any]]],
    gold_statements: List[str],
    gold_derivation_error: Optional[str],
    sql_executor: SqlExecutor,
    mismatch_log_config: Dict[str, Any],
    scenario_id: str,
    request_index: int,
) -> Tuple[Optional[bool], bool, Optional[str]]:
    """
    Evaluate whether the agent's decision matches an accepted action and verifiable evidence.

    Conventions:
      - reject: must NOT call write_db; optionally must match expected snapshot (no changes).
      - execute: optionally compare SELECT results and/or state modification snapshot.
    """
    is_correct: Optional[bool] = None
    result_verified = False
    verification_details: Optional[str] = None

    if not expected_action:
        return is_correct, result_verified, verification_details

    # Detect any write_db usage (a reject run must not even attempt it).
    write_db_steps = [
        step for step in history_steps if getattr(step, "action_name", None) == "write_db"
    ]
    write_db_executed = sum(
        1
        for step in write_db_steps
        if getattr(getattr(step, "tool_result", None), "executed", False)
    )
    failed_write_db_steps = []
    for step in write_db_steps:
        tool_result = getattr(step, "tool_result", None)
        if tool_result is None:
            continue
        if not getattr(tool_result, "executed", False):
            continue
        sub_results = getattr(tool_result, "sub_results", []) or []
        has_failed_sub_result = any(
            not getattr(sub_result, "success", False) for sub_result in sub_results
        )
        if has_failed_sub_result or not getattr(tool_result, "complete", True):
            failed_write_db_steps.append(step)

    if disallow_write_db and write_db_steps:
        is_correct = False
        verification_details = f"write_db not allowed for this request ({len(write_db_steps)} call(s))"
        if mismatch_log_config["enabled"]:
            logger.warning(
                "write_db disallowed: scenario=%s request_index=%s calls=%s executed=%s",
                scenario_id,
                request_index,
                len(write_db_steps),
                write_db_executed,
            )
        return is_correct, result_verified, verification_details

    allowed_failed_write_db_error_patterns = allowed_failed_write_db_error_patterns or []
    disallowed_failed_write_db_steps = failed_write_db_steps
    if disallow_failed_write_db and allowed_failed_write_db_error_patterns:
        disallowed_failed_write_db_steps = []
        for step in failed_write_db_steps:
            tool_result = getattr(step, "tool_result", None)
            combined_error = "\n".join(
                [
                    str(getattr(tool_result, "summary", "") or ""),
                    str(getattr(tool_result, "error", "") or ""),
                ]
            )
            if not any(
                _evaluation_pattern_matches(pattern, combined_error)
                for pattern in allowed_failed_write_db_error_patterns
            ):
                disallowed_failed_write_db_steps.append(step)

    if disallow_failed_write_db and disallowed_failed_write_db_steps:
        is_correct = False
        verification_details = (
            "Failed write_db call not allowed for this request "
            f"({len(disallowed_failed_write_db_steps)} call(s))"
        )
        if mismatch_log_config["enabled"]:
            first_failed = disallowed_failed_write_db_steps[0]
            logger.warning(
                "Failed write_db disallowed: scenario=%s request_index=%s failed_calls=%s summary=%s",
                scenario_id,
                request_index,
                len(disallowed_failed_write_db_steps),
                _truncate_text(
                    getattr(getattr(first_failed, "tool_result", None), "summary", ""),
                    mismatch_log_config["sql_truncate"],
                ),
            )
        return is_correct, result_verified, verification_details

    forbidden_sql_patterns = forbidden_sql_patterns or []
    if forbidden_sql_patterns:
        sql_candidates = _forbidden_sql_candidates(
            history_steps,
            forbidden_sql_patterns_scope,
        )
        for pattern in forbidden_sql_patterns:
            for candidate in sql_candidates:
                sql_text = candidate["sql"]
                if _evaluation_pattern_matches(pattern, sql_text):
                    is_correct = False
                    verification_details = f"Forbidden SQL pattern used: {pattern}"
                    if mismatch_log_config["enabled"]:
                        logger.warning(
                            "Forbidden SQL pattern: scenario=%s request_index=%s scope=%s pattern=%s tool=%s sql=%s",
                            scenario_id,
                            request_index,
                            _normalize_forbidden_sql_patterns_scope(forbidden_sql_patterns_scope),
                            pattern,
                            candidate.get("tool", ""),
                            _truncate_text(sql_text, mismatch_log_config["sql_truncate"]),
                        )
                    return is_correct, result_verified, verification_details

    allowed_actions = acceptable_actions or ([expected_action] if expected_action else [])

    # Stage 0: the top-level action must be accepted by the scenario contract.
    if actual_action not in allowed_actions:
        is_correct = False
        allowed_text = "/".join(allowed_actions)
        verification_details = f"Action mismatch: expected one of {allowed_text}, got {actual_action}"
        if mismatch_log_config["enabled"]:
            logger.warning(
                "Action mismatch: scenario=%s request_index=%s expected=%s acceptable=%s actual=%s",
                scenario_id,
                request_index,
                expected_action,
                allowed_actions,
                actual_action,
            )
        return is_correct, result_verified, verification_details

    # Stage 1: reject-specific checks.
    if actual_action == "reject":
        if write_db_steps:
            is_correct = False
            verification_details = (
                f"Rejected but invoked write_db ({len(write_db_steps)} call(s), {write_db_executed} executed)"
            )
            if mismatch_log_config["enabled"]:
                logger.warning(
                    "Reject invoked write_db: scenario=%s request_index=%s calls=%s executed=%s",
                    scenario_id,
                    request_index,
                    len(write_db_steps),
                    write_db_executed,
                )
            return is_correct, result_verified, verification_details

        state_ok = None
        if expected_modification is not None and agent_snapshot is not None:
            try:
                actual_modification = agent_snapshot.get_modification()
                actual_for_compare = actual_modification.without_columns(
                    snapshot_ignore_columns or {}
                )
                expected_for_compare = expected_modification.without_columns(
                    snapshot_ignore_columns or {}
                )
                if agent_snapshot.compare_modifications(actual_for_compare, expected_for_compare):
                    state_ok = True
                else:
                    state_ok = False
                    verification_details = "Rejected but database state changed"
                    if mismatch_log_config["enabled"]:
                        if mismatch_log_config["snapshot_detail"]:
                            diff_summary = _summarize_snapshot_mismatch(
                                actual_for_compare,
                                expected_for_compare,
                                max_tables=mismatch_log_config["max_tables"],
                                max_rows=mismatch_log_config["max_rows"],
                            )
                        else:
                            diff_summary = _summarize_snapshot_mismatch_counts(
                                actual_for_compare,
                                expected_for_compare,
                            )
                        logger.warning(
                            "Snapshot mismatch on reject: scenario=%s request_index=%s details=%s",
                            scenario_id,
                            request_index,
                            json.dumps(diff_summary, default=str),
                        )
            except Exception as e:
                logger.error(f"Error comparing modifications: {e}")
                state_ok = False
                verification_details = f"Snapshot comparison error: {str(e)}"

        if state_ok is False:
            is_correct = False
            return is_correct, result_verified, verification_details

        final_state_ok = None
        final_state_details = None
        if final_state_checks:
            final_state_ok, final_state_details = _evaluate_final_state_checks(
                final_state_checks=final_state_checks,
                sql_executor=sql_executor,
                mismatch_log_config=mismatch_log_config,
                scenario_id=scenario_id,
                request_index=request_index,
            )
            if final_state_ok is False:
                return False, result_verified, final_state_details

        is_correct = True
        result_verified = True
        if verification_details is None:
            if final_state_ok is True:
                verification_details = final_state_details or "Rejected and final state checks passed"
            else:
                verification_details = (
                    "Rejected with no state changes"
                    if expected_modification is not None
                    else "Rejection accepted"
                )
        return is_correct, result_verified, verification_details

    # Stage 2: execute-specific checks.
    if actual_action == "execute":
        expected_output = {"rows": gold_expected_rows} if gold_expected_rows is not None else None

        # If we need gold to define expected behavior but gold couldn't run, mark as incorrect.
        if gold_derivation_error and expected_modification is None and expected_output is None:
            is_correct = False
            return is_correct, result_verified, gold_derivation_error

        results_ok = None
        if expected_output is not None:
            selected_read = _find_matching_successful_select_sql(
                history_steps=history_steps,
                expected_output=expected_output,
                gold_statements=gold_statements,
                sql_executor=sql_executor,
            )
            agent_sql = selected_read.get("sql")
            actual_read = selected_read.get("actual_read")
            cmp = selected_read.get("comparison")

            if not agent_sql:
                results_ok = False
                verification_details = "No successful SELECT found in tool history to compare"
                if mismatch_log_config["enabled"]:
                    logger.warning(
                        "Missing SELECT for result compare: scenario=%s request_index=%s",
                        scenario_id,
                        request_index,
                    )
            elif not actual_read or not actual_read.get("success", False) or actual_read.get("type") != "READ":
                results_ok = False
                verification_details = "Failed to re-run agent SELECT for result comparison"
                if mismatch_log_config["enabled"]:
                    logger.warning(
                        "Failed to re-run SELECT for compare: scenario=%s request_index=%s sql=%s",
                        scenario_id,
                        request_index,
                        _truncate_text(agent_sql, mismatch_log_config["sql_truncate"]),
                    )
            else:
                if cmp and cmp.get("matches", False):
                    results_ok = True
                else:
                    results_ok = False
                    reason = cmp.get("reason", "unknown") if cmp else "unknown"
                    verification_details = f"Query results mismatch: {reason}"
                    if mismatch_log_config["enabled"]:
                        details = _summarize_query_mismatch_details(
                            (cmp or {}).get("details", {}),
                            mismatch_log_config["level"],
                        )
                        logger.warning(
                            "Query results mismatch: scenario=%s request_index=%s reason=%s details=%s sql=%s",
                            scenario_id,
                            request_index,
                            reason,
                            json.dumps(details, default=str),
                            _truncate_text(agent_sql, mismatch_log_config["sql_truncate"]),
                        )

        state_ok = None
        if expected_modification is not None and agent_snapshot is not None:
            try:
                actual_modification = agent_snapshot.get_modification()
                actual_for_compare = actual_modification.without_columns(
                    snapshot_ignore_columns or {}
                )
                expected_for_compare = expected_modification.without_columns(
                    snapshot_ignore_columns or {}
                )
                if agent_snapshot.compare_modifications(actual_for_compare, expected_for_compare):
                    state_ok = True
                else:
                    state_ok = False
                    if verification_details is None:
                        verification_details = "State changes do not match expected modification"
                    if mismatch_log_config["enabled"]:
                        if mismatch_log_config["snapshot_detail"]:
                            diff_summary = _summarize_snapshot_mismatch(
                                actual_for_compare,
                                expected_for_compare,
                                max_tables=mismatch_log_config["max_tables"],
                                max_rows=mismatch_log_config["max_rows"],
                            )
                        else:
                            diff_summary = _summarize_snapshot_mismatch_counts(
                                actual_for_compare,
                                expected_for_compare,
                            )
                        logger.warning(
                            "Snapshot mismatch on execute: scenario=%s request_index=%s details=%s",
                            scenario_id,
                            request_index,
                            json.dumps(diff_summary, default=str),
                        )
            except Exception as e:
                logger.error(f"Error comparing modifications: {e}")
                state_ok = False
                verification_details = f"Snapshot comparison error: {str(e)}"

        final_state_ok = None
        final_state_details = None
        if final_state_checks:
            final_state_ok, final_state_details = _evaluate_final_state_checks(
                final_state_checks=final_state_checks,
                sql_executor=sql_executor,
                mismatch_log_config=mismatch_log_config,
                scenario_id=scenario_id,
                request_index=request_index,
            )
            if final_state_ok is False:
                verification_details = final_state_details

        if gold_derivation_error:
            verification_details = (
                (verification_details + " | " if verification_details else "")
                + gold_derivation_error
            )

        # Decide correctness with whatever evidence is available.
        if results_ok is False or state_ok is False or final_state_ok is False:
            is_correct = False
        elif expected_output is not None or expected_modification is not None or final_state_checks:
            is_correct = True
            result_verified = True
            if verification_details is None:
                verified_parts: List[str] = []
                if expected_output is not None:
                    verified_parts.append("query results")
                if expected_modification is not None:
                    verified_parts.append("state changes")
                if final_state_checks:
                    verified_parts.append("final state checks")
                if len(verified_parts) == 1:
                    if verified_parts[0] == "query results":
                        verification_details = "Query results match expected"
                    elif verified_parts[0] == "state changes":
                        verification_details = "State changes match expected modification"
                    else:
                        verification_details = final_state_details or "Final state checks passed"
                elif len(verified_parts) == 2:
                    verification_details = (
                        f"{verified_parts[0].capitalize()} and {verified_parts[1]} match expected"
                    )
                elif verified_parts:
                    verification_details = (
                        "Query results, state changes, and final state checks match expected"
                    )
        else:
            # No gold output and no expected snapshot: only verify the action type.
            is_correct = True
            verification_details = ACTION_ONLY_VERIFICATION_DETAILS
            logger.warning(
                "Verification limited to action type only (missing expected_snapshot and gold_query): "
                "scenario=%s request_index=%s",
                scenario_id,
                request_index,
            )

    return is_correct, result_verified, verification_details


def _serialize_history_steps(history_steps: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert runner history to JSON-serializable dicts (for logs and the data viewer).

    Pydantic models may contain Enum values; normalize them to raw strings for stable output.
    """
    history_log: List[Dict[str, Any]] = []
    for step in history_steps or []:
        step_dict = step.model_dump() if hasattr(step, "model_dump") else dict(step)
        if "phase" in step_dict:
            step_dict["phase"] = (
                str(step_dict["phase"].value)
                if hasattr(step_dict["phase"], "value")
                else str(step_dict["phase"])
            )
        if "action_type" in step_dict:
            step_dict["action_type"] = (
                str(step_dict["action_type"].value)
                if hasattr(step_dict["action_type"], "value")
                else str(step_dict["action_type"])
            )
        history_log.append(step_dict)
    return history_log


def _maybe_run_understanding_eval(
    *,
    expected_action: Optional[str],
    is_correct: Optional[bool],
    require_qa_correct: bool,
    qa_questions: Optional[List[str]],
    expected_answers: List[str],
    qa_model: str,
    aux_model: str,
    security_context: Optional[Dict[str, Any]],
    session_context: str,
    request_text: str,
    actual_action: str,
    rejection_reason: Optional[str],
    final_queries: List[Dict[str, str]],
    history_log: List[Dict[str, Any]],
    scenario_knowledge: Any,
    scenario_semantics: Any,
    verification_details: Optional[str],
) -> Tuple[Optional[bool], Optional[List[Dict[str, str]]], Optional[Dict[str, Any]], Optional[bool], Optional[str]]:
    """
    Optional "understanding" evaluation step (Q&A rubric).

    Runs when:
      - expected_action exists
      - QA questions are provided by request or scenario

    A failed state transition still gets QA/judge diagnostics so we can tell
    whether the agent understood the goal but implemented it incorrectly, or
    misunderstood the goal/scope itself. QA can only make a correct state
    transition fail when `require_qa_correct` is set; it never rescues an
    incorrect state transition.
    """
    if not (expected_action and qa_questions):
        if expected_action and is_correct is True and require_qa_correct:
            verification_details = (
                (verification_details + " | " if verification_details else "")
                + "Required QA missing"
            )
            return None, None, None, False, verification_details
        return None, None, None, is_correct, verification_details

    qa_pairs: List[Dict[str, str]] = []
    # The follow-up answers should come from the model under evaluation, not
    # from the judge/user auxiliary model. The judge can still be aux_model.
    qa_llm = create_model(qa_model, temperature=0.0, max_tokens=4096)

    evidence = _format_full_trace_for_qa(history_log)
    compact_knowledge = _compact_knowledge_for_eval(scenario_knowledge)
    compact_semantics = _compact_semantics_for_eval(scenario_semantics)
    knowledge_text = json.dumps(compact_knowledge, indent=2, default=str) if compact_knowledge else "No business rules"
    semantics_text = json.dumps(compact_semantics, indent=2, default=str) if compact_semantics else "No column semantics"

    qa_system = (
        "You are the same database agent that just handled this request.\n"
        "- Answer follow-up questions using ONLY your complete trace, executed SQL, and knowledge base.\n"
        "- If the evidence does not contain the needed detail, say you don't know.\n"
        "- Be concise (1-3 sentences).\n"
    )

    for question in qa_questions:
        security_context_text = (
            json.dumps(security_context, indent=2, default=str)
            if security_context
            else "None"
        )
        qa_prompt = f"""PRINCIPAL / SECURITY CONTEXT:
{security_context_text}

SESSION CONTEXT:
{session_context}

ORIGINAL REQUEST:
{request_text}

AGENT DECISION:
- action: {actual_action}
- reason: {rejection_reason or ''}
- executed SQL: {json.dumps(final_queries, indent=2)}

COMPLETE AGENT TRACE:
{evidence}

KNOWLEDGE BASE:
{knowledge_text}

COLUMN SEMANTICS:
{semantics_text}

QUESTION:
{question}
"""
        resp = qa_llm.generate([
            Message(role=Role.SYSTEM, content=qa_system),
            Message(role=Role.USER, content=qa_prompt),
        ])
        answer = (resp.content or "").strip()
        if not answer:
            # Some auxiliary providers occasionally return an empty answer even when
            # the tested agent's trace contains a concise final explanation. Treat
            # that as a judge/aux artifact and judge the agent-owned final reasoning
            # instead of failing a correct state transition on a blank QA response.
            answer = _last_finish_reasoning(history_log)
        qa_pairs.append({"question": question, "answer": answer})

    understanding_passed: Optional[bool]
    judge_evaluation: Optional[Dict[str, Any]]
    try:
        judge = JudgeAgent(aux_model)
        judge_evaluation = judge.evaluate_answers(
            scenario_context={
                "request": request_text,
                "expected_answers": expected_answers,
                "security_context": security_context,
            },
            qa_pairs=qa_pairs,
        )
        understanding_passed = bool(judge_evaluation.get("all_correct", False))
    except Exception as e:
        understanding_passed = False
        judge_evaluation = {"error": str(e)}

    if understanding_passed is False:
        reason = None
        if isinstance(judge_evaluation, dict):
            reason = judge_evaluation.get("overall_reasoning") or judge_evaluation.get("error")
        if require_qa_correct:
            is_correct = False
            suffix = f"Understanding failed (required): {reason}" if reason else "Understanding failed (required)"
        else:
            suffix = f"Understanding failed (diagnostic only): {reason}" if reason else "Understanding failed (diagnostic only)"
        verification_details = (
            (verification_details + " | " if verification_details else "")
            + suffix
        )
    elif understanding_passed is True:
        verification_details = (
            (verification_details + " | " if verification_details else "")
            + "Understanding passed"
        )

    return understanding_passed, qa_pairs, judge_evaluation, is_correct, verification_details


def _apply_concurrent_operations(
    *,
    scenario: TestScenario,
    phase: str,
    session_id: Any,
    sql_executor: SqlExecutor,
    open_connections: List[Any],
    keep_open_filter: Optional[bool] = None,
) -> None:
    """
    Apply scenario-defined concurrent operations to simulate other users touching the DB.

    These operations run outside the agent loop and can change the DB state between requests.
    """
    if phase not in {"before", "after"}:
        raise ValueError(f"Invalid concurrent op phase: {phase}")

    key = "before_session_id" if phase == "before" else "after_session_id"
    matched = [
        op
        for op in (scenario.concurrent_operations or [])
        if op.get(key) == session_id
        and (
            keep_open_filter is None
            or bool(op.get("keep_open", False)) is bool(keep_open_filter)
        )
    ]
    if not matched:
        return

    for op in matched:
        _apply_concurrent_operation_sql(
            op=op,
            sql_executor=sql_executor,
            open_connections=open_connections,
        )


def _apply_concurrent_operation_sql(
    *,
    op: Dict[str, Any],
    sql_executor: SqlExecutor,
    open_connections: List[Any],
) -> None:
    """Execute the SQL payload for one scenario-defined concurrent operation."""
    prev_timeout = sql_executor.timeout
    sql_executor.timeout = None
    try:
        queries = op.get("sql", [])
        if isinstance(queries, str):
            queries = [queries]

        if bool(op.get("keep_open", False)):
            # Keep the transaction open so locks persist across the agent run.
            conn = psycopg2.connect(**sql_executor.db_config)
            try:
                with conn.cursor() as cursor:
                    for query in queries:
                        cursor.execute(query)
                open_connections.append(conn)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                raise
            return

        for query in queries:
            result = sql_executor.execute(query, allow_modify=True)
            if not result.get("success", False):
                raise RuntimeError(result.get("error", "concurrent operation SQL failed"))
    finally:
        sql_executor.timeout = prev_timeout


def _build_after_tool_call_hook(
    *,
    scenario: TestScenario,
    sql_executor: SqlExecutor,
    open_connections: List[Any],
):
    """Build a runner hook for concurrent operations keyed to tool-call ordinals."""
    hook_ops = []
    for idx, op in enumerate(scenario.concurrent_operations or []):
        spec = op.get("after_tool_call")
        if isinstance(spec, dict):
            hook_ops.append((idx, op, spec))
    if not hook_ops:
        return None

    call_counts: Dict[str, int] = {}
    applied: set[int] = set()

    def hook(*, tool_name: Optional[str], tool_params: Dict[str, Any], tool_result: Any) -> List[str]:
        del tool_params
        normalized_tool = str(tool_name or "")
        call_counts[normalized_tool] = call_counts.get(normalized_tool, 0) + 1
        ordinal_now = call_counts[normalized_tool]
        notes: List[str] = []

        for idx, op, spec in hook_ops:
            if idx in applied:
                continue
            wanted_tool = str(spec.get("tool") or spec.get("tool_name") or "").strip()
            if wanted_tool and wanted_tool != normalized_tool:
                continue
            raw_ordinal = spec.get("ordinal", 1)
            ordinal_text = str(raw_ordinal).strip().lower()
            if raw_ordinal is None or ordinal_text in {"", "*", "any", "all"}:
                wanted_ordinal = None
            else:
                try:
                    wanted_ordinal = int(raw_ordinal)
                except Exception:
                    wanted_ordinal = 1
            if wanted_ordinal is not None and ordinal_now != wanted_ordinal:
                continue

            success_only = bool(spec.get("success_only", True))
            if success_only and not bool(getattr(tool_result, "all_succeeded", False)):
                continue

            _apply_concurrent_operation_sql(
                op=op,
                sql_executor=sql_executor,
                open_connections=open_connections,
            )
            applied.add(idx)
            label = str(op.get("label") or f"concurrent op {idx}")
            notes.append(f"applied {label}")

        return notes

    return hook


def _build_after_write_statement_hook(
    *,
    scenario: TestScenario,
    sql_executor: SqlExecutor,
    open_connections: List[Any],
):
    """Build a runner hook for concurrent operations between write_db statements."""
    hook_ops = []
    for idx, op in enumerate(scenario.concurrent_operations or []):
        spec = op.get("after_write_statement")
        if isinstance(spec, dict):
            hook_ops.append((idx, op, spec))
    if not hook_ops:
        return None

    applied: set[int] = set()

    def hook(
        *,
        statement_index: int,
        sql: str,
        purpose: str,
        sub_result: Any,
    ) -> List[str]:
        del purpose
        notes: List[str] = []
        ordinal_now = statement_index + 1
        sql_text = str(sql or "")

        for idx, op, spec in hook_ops:
            if idx in applied:
                continue
            raw_ordinal = spec.get("ordinal", 1)
            ordinal_text = str(raw_ordinal).strip().lower()
            if raw_ordinal is None or ordinal_text in {"", "*", "any", "all"}:
                wanted_ordinal = None
            else:
                try:
                    wanted_ordinal = int(raw_ordinal)
                except Exception:
                    wanted_ordinal = 1
            if wanted_ordinal is not None and ordinal_now != wanted_ordinal:
                continue

            contains = str(spec.get("sql_contains") or "").strip()
            if contains and contains.lower() not in sql_text.lower():
                continue

            regex = str(spec.get("sql_regex") or "").strip()
            if regex and not re.search(regex, sql_text, flags=re.IGNORECASE | re.DOTALL):
                continue

            success_only = bool(spec.get("success_only", True))
            if success_only and not bool(getattr(sub_result, "success", False)):
                continue

            _apply_concurrent_operation_sql(
                op=op,
                sql_executor=sql_executor,
                open_connections=open_connections,
            )
            applied.add(idx)
            label = str(op.get("label") or f"concurrent op {idx}")
            notes.append(f"applied {label}")

        return notes

    return hook


def run_scenario(
    scenario_path: Path,
    model_name: str = "bedrock/claude-4-sonnet",
    aux_model: str = "gpt-5.2",
    request_index=None,
    all_requests: bool = False,
    reuse_db_name: Optional[str] = None,
    skip_scenario_setup: bool = False,
    cleanup_reuse_db: bool = False,
    read_select_gold_in_place: bool = False,
    force_write_db_read_only: bool = False,
) -> dict:
    """Run a single scenario and return results."""

    # --- Load scenario config and initialize per-scenario utilities ---
    try:
        scenario = load_scenario(scenario_path)
    except Exception as e:
        logger.exception("Failed to load scenario: %s", scenario_path)
        return {
            "scenario_id": scenario_path.name,
            "error": f"Failed to load scenario: {e}",
        }
    logger.info(f"Running scenario: {scenario.id}")
    mismatch_log_config = _get_mismatch_log_config()

    # Create database config
    base_db_config = get_db_config()
    # Prefer explicit reference_database; otherwise fall back to a best-effort
    # guess from the scenario id/path.
    scenario_db_name = scenario.reference_database or Path(str(scenario.id)).name.replace('_template', '')

    # Create database cloner for test database management
    cloner = DatabaseCloner(base_db_config)
    test_db_name = None

    try:
        use_clone = bool(getattr(scenario, "database_use_clone", True))
        read_only = bool(getattr(scenario, "database_read_only", False))

        if not use_clone and not reuse_db_name:
            # Safety: direct mode is for stable, read-only template DBs (e.g., large perf datasets).
            if scenario.setup_sql:
                return {
                    "scenario_id": scenario.id,
                    "error": "setup_sql is not allowed when database_options.use_clone=false",
                }
            if scenario.concurrent_operations:
                return {
                    "scenario_id": scenario.id,
                    "error": "concurrent_operations are not allowed when database_options.use_clone=false",
                }

        # Decide which request(s) to run (scenarios can be multi-turn/multi-request).
        if all_requests:
            request_indices = list(range(len(scenario.requests)))
        else:
            selected_index = 0 if request_index is None else request_index
            if selected_index < 0 or selected_index >= len(scenario.requests):
                return {
                    "error": f"Request index {selected_index} out of range",
                    "scenario_id": scenario.id,
                }
            request_indices = [selected_index]

        run_db_name: str
        if reuse_db_name:
            run_db_name = reuse_db_name
            if cleanup_reuse_db:
                test_db_name = reuse_db_name
            logger.info(
                "Using externally prepared scenario database: %s "
                "(skip_scenario_setup=%s, cleanup_reuse_db=%s)",
                run_db_name,
                skip_scenario_setup,
                cleanup_reuse_db,
            )
        elif use_clone:
            # Clone reference DB -> isolated test DB (agent will read/write only this clone).
            test_db_name, setup_time = cloner.create_test_database(
                scenario_name=scenario.id,
                reference_db=scenario_db_name,
            )
            run_db_name = test_db_name
            logger.info(f"Created test database {test_db_name} in {setup_time:.2f}s")
        else:
            run_db_name = scenario_db_name
            logger.info(
                "Using reference database directly (no clone): %s (read_only=%s)",
                run_db_name,
                read_only,
            )

        # Use the selected database for agent execution
        db_config = {**base_db_config, "database": run_db_name}
        sql_executor = SqlExecutor(db_config)

        # Run scenario-level setup SQL (if any)
        if scenario.setup_sql and not skip_scenario_setup:
            # setup.sql is applied ONCE per scenario clone, before any requests.
            setup_path = scenario_path / scenario.setup_sql
            if not setup_path.exists():
                return {
                    "scenario_id": scenario.id,
                    "error": f"setup_sql not found: {setup_path}",
                }

            setup_sql = setup_path.read_text()
            prev_timeout = sql_executor.timeout
            sql_executor.timeout = None
            try:
                setup_result = sql_executor.execute(setup_sql, allow_modify=True)
            finally:
                sql_executor.timeout = prev_timeout
            if not setup_result.get("success", False):
                return {
                    "scenario_id": scenario.id,
                    "error": f"Failed to run setup_sql ({scenario.setup_sql}): {setup_result.get('error', 'Unknown error')}",
                }

        # Create registry and runner
        registry = create_default_registry()
        config = RunnerConfig(
            total_budget=_get_runner_total_budget(),
            max_steps=_get_runner_max_steps(),
            max_queries_per_call=_get_runner_max_queries_per_call(),
        )
        runner = AgentRunner(registry, config)

        all_request_results = []
        session_state: Dict[Any, Dict[str, Any]] = {}
        open_concurrent_connections: List[Any] = []
        after_tool_call_hook = _build_after_tool_call_hook(
            scenario=scenario,
            sql_executor=sql_executor,
            open_connections=open_concurrent_connections,
        )
        after_write_statement_hook = _build_after_write_statement_hook(
            scenario=scenario,
            sql_executor=sql_executor,
            open_connections=open_concurrent_connections,
        )
        for request_idx in request_indices:
            request = scenario.requests[request_idx]
            expected_action = request.expected_action
            acceptable_actions = getattr(request, "acceptable_actions", None)
            disallow_write_db = bool(getattr(request, "disallow_write_db", False))
            disallow_failed_write_db = bool(getattr(request, "disallow_failed_write_db", False))
            allowed_failed_write_db_error_patterns = (
                getattr(request, "allowed_failed_write_db_error_patterns", None) or []
            )
            forbidden_sql_patterns = getattr(request, "forbidden_sql_patterns", None) or []
            forbidden_sql_patterns_scope = _normalize_forbidden_sql_patterns_scope(
                getattr(request, "forbidden_sql_patterns_scope", "trace")
            )
            final_state_checks = getattr(request, "final_state_check", None) or []
            effective_security_context, effective_db_session = _resolve_request_session_bindings(
                request=request,
                session_state=session_state,
            )

            # Apply "before" concurrent operations (committed changes first), so the agent and gold
            # derivation see the same baseline state.
            _apply_concurrent_operations(
                scenario=scenario,
                phase="before",
                session_id=request.session_id,
                sql_executor=sql_executor,
                open_connections=open_concurrent_connections,
                keep_open_filter=False,
            )

            # Per-request timeout and tool context. SqlExecutor is shared across requests in the same scenario
            # (on the same test DB), so we must update timeout before running tools/agent.
            _apply_request_timeout(sql_executor, request)
            request_sql_executor = SqlExecutor(
                sql_executor.db_config,
                timeout=sql_executor.timeout,
                db_session=effective_db_session,
            )
            context = _build_request_context(
                scenario=scenario,
                request=request,
                sql_executor=request_sql_executor,
                model_name=model_name,
                aux_model=aux_model,
            )
            context["read_only"] = read_only
            if force_write_db_read_only:
                context["force_write_db_read_only"] = True
            if after_tool_call_hook is not None:
                context["after_tool_call_hook"] = after_tool_call_hook
            if after_write_statement_hook is not None:
                context["after_write_statement_hook"] = after_write_statement_hook

            # LLM agent is stateless per request; Runner maintains the history per run() call.
            agent = LLMAgent(model_name=model_name)

            expected_modification = request.expected_snapshot
            gold_statements = _normalize_sql_statements(
                getattr(request, "gold_queries", None) or getattr(request, "gold_query", None)
            )
            qa_questions_for_verification = getattr(request, "qa_questions", None) or getattr(scenario, "qa_questions", None)
            gold_statements_for_verification: List[str] = gold_statements
            request_efficiency = _efficiency_config(request)

            # These are per-request scratch DBs for gold/snapshot derivation.
            gold_db_name: Optional[str] = None
            gold_snapshot_db_name: Optional[str] = None
            agent_snapshot_db_name: Optional[str] = None
            agent_snapshot: Optional[SnapshotManager] = None
            gold_expected_rows: Optional[List[Dict[str, Any]]] = None
            gold_derivation_error: Optional[str] = None
            gold_runtime_seconds: Optional[float] = None
            efficiency_metrics: Optional[Dict[str, Any]] = None

            try:
                # Gold derivation runs BEFORE agent execution so the baseline state matches.
                gold = _derive_gold_expectations(
                    expected_action=expected_action,
                    gold_statements=gold_statements_for_verification,
                    expected_modification=expected_modification,
                    scenario_id=scenario.id,
                    request_session_id=request.session_id,
                    cloner=cloner,
                    base_db_config=base_db_config,
                    reference_db=run_db_name,
                    sql_executor=request_sql_executor,
                    allow_db_clone=use_clone and not reuse_db_name,
                    read_only=read_only,
                    read_select_gold_in_place=read_select_gold_in_place,
                    derive_state_modification_from_gold=not bool(
                        getattr(request, "evaluation", {}).get("verify_with_final_state_only", False)
                    ),
                )
                expected_modification = gold.expected_modification
                gold_expected_rows = gold.expected_rows
                gold_derivation_error = gold.error
                gold_db_name = gold.gold_db_name
                gold_snapshot_db_name = gold.gold_snapshot_db_name
                gold_runtime_seconds = gold.gold_runtime_seconds

                # If we have an expected modification, create an agent-side snapshot baseline to diff against.
                agent_snapshot, agent_snapshot_db_name = _maybe_init_agent_snapshot(
                    expected_modification=expected_modification,
                    scenario_id=scenario.id,
                    request_session_id=request.session_id,
                    db_config=db_config,
                )

                # Apply "before" keep-open operations (locks), right before the agent runs.
                _apply_concurrent_operations(
                    scenario=scenario,
                    phase="before",
                    session_id=request.session_id,
                    sql_executor=sql_executor,
                    open_connections=open_concurrent_connections,
                    keep_open_filter=True,
                )

                # Run the agent (tool calls may include write_db commits)
                session_context = _build_session_context(
                    all_request_results,
                    current_session_id=request.session_id,
                    current_conversation_id=getattr(request, "conversation_id", None),
                )
                start_time = time.time()
                result = runner.run(
                    agent=agent,
                    user_request=request.request,
                    database_name=run_db_name,
                    context=context,
                    security_context=effective_security_context,
                    session_context=session_context,
                )
                duration = time.time() - start_time
                agent_llm_stats: Dict[str, Any] = {}
                try:
                    agent_llm = getattr(agent, "llm", None)
                    if agent_llm is not None and hasattr(agent_llm, "get_stats"):
                        agent_llm_stats = agent_llm.get_stats() or {}
                except Exception:
                    agent_llm_stats = {}

                runtime_failure = _detect_runtime_failure(result)
                if runtime_failure:
                    history_log = _serialize_history_steps(result.get("history", []) or [])
                    all_request_results.append({
                        "scenario_id": scenario.id,
                        "session_id": request.session_id,
                        "conversation_id": getattr(request, "conversation_id", request.session_id),
                        "request_index": request_idx,
                        "request": request.request,
                        "effective_security_context": effective_security_context,
                        "effective_db_session": effective_db_session,
                        "expected_action": expected_action,
                        "acceptable_actions": acceptable_actions,
                        "disallow_write_db": disallow_write_db,
                        "disallow_failed_write_db": disallow_failed_write_db,
                        "allowed_failed_write_db_error_patterns": allowed_failed_write_db_error_patterns,
                        "forbidden_sql_patterns": forbidden_sql_patterns,
                        "forbidden_sql_patterns_scope": forbidden_sql_patterns_scope,
                        "final_state_check": final_state_checks,
                        "actual_action": "error",
                        "is_correct": None,
                        "result_verified": False,
                        "verification_details": None,
                        "duration_seconds": round(duration, 2),
                        "budget_used": result["budget_usage"]["used"],
                        "budget_total": result["budget_usage"]["total_budget"],
                        "num_steps": len(result.get("history", [])),
                        "history": history_log,
                        "final_queries": None,
                        "rejection_reason": None,
                        "understanding_passed": None,
                        "qa_pairs": None,
                        "judge_evaluation": None,
                        "agent_llm_stats": agent_llm_stats,
                        "efficiency": None,
                        "error": runtime_failure,
                    })
                    break

                actual_action = result["action"]
                history_steps = result.get("history", []) or []
                is_correct, result_verified, verification_details = _evaluate_expected_action(
                    expected_action=expected_action,
                    acceptable_actions=acceptable_actions,
                    disallow_write_db=disallow_write_db,
                    disallow_failed_write_db=disallow_failed_write_db,
                    allowed_failed_write_db_error_patterns=allowed_failed_write_db_error_patterns,
                    forbidden_sql_patterns=forbidden_sql_patterns,
                    forbidden_sql_patterns_scope=forbidden_sql_patterns_scope,
                    actual_action=actual_action,
                    history_steps=history_steps,
                    expected_modification=expected_modification,
                    agent_snapshot=agent_snapshot,
                    snapshot_ignore_columns=getattr(request, "snapshot_ignore_columns", {}),
                    final_state_checks=final_state_checks,
                    gold_expected_rows=gold_expected_rows,
                    gold_statements=gold_statements_for_verification,
                    gold_derivation_error=gold_derivation_error,
                    sql_executor=request_sql_executor,
                    mismatch_log_config=mismatch_log_config,
                    scenario_id=scenario.id,
                    request_index=request_idx,
                )

                if is_correct is True and request_efficiency["enabled"]:
                    efficiency_passed, efficiency_details, efficiency_metrics = _evaluate_relative_efficiency(
                        enabled=True,
                        multiplier=float(request_efficiency["multiplier"]),
                        plan_gates_enabled=bool(request_efficiency.get("plan_gates_enabled")),
                        require_index=bool(request_efficiency.get("require_index")),
                        disallow_seq_scan=bool(request_efficiency.get("disallow_seq_scan")),
                        required_indexes=request_efficiency.get("required_indexes") or [],
                        required_index_policy=request_efficiency.get("required_index_policy", "any"),
                        required_relations=request_efficiency.get("required_relations") or [],
                        required_relation_policy=request_efficiency.get("required_relation_policy", "any"),
                        absolute_slack_seconds=request_efficiency.get("absolute_slack_seconds"),
                        forbidden_sql_patterns=request_efficiency.get("forbidden_sql_patterns") or [],
                        gold_runtime_seconds=gold_runtime_seconds,
                        history_steps=history_steps,
                        sql_executor=request_sql_executor,
                        scenario_id=scenario.id,
                        request_index=request_idx,
                        gold_expected_rows=gold_expected_rows,
                        gold_statements=gold_statements_for_verification,
                    )
                    if efficiency_details:
                        verification_details = (
                            (verification_details + " | " if verification_details else "")
                            + efficiency_details
                        )
                    if efficiency_passed is False:
                        is_correct = False
            finally:
                # Always cleanup scratch DBs for this request.
                _log_cleanup_failures(
                    f"request scratch databases for {scenario.id} session {request.session_id}",
                    cloner.cleanup_test_databases(
                        [agent_snapshot_db_name, gold_snapshot_db_name, gold_db_name]
                    ),
                )

            # Serialize history for logging
            history_log = _serialize_history_steps(history_steps)

            # Stage 2: understanding / Q&A evaluation (legacy parity)
            qa_questions = qa_questions_for_verification
            expected_answers = (
                getattr(request, "expected_answers", None)
                or getattr(scenario, "expected_answers", None)
                or []
            )

            final_queries_for_qa = _extract_final_queries(history_steps)
            rejection_reason_for_qa = result.get("reason") if actual_action == "reject" else None
            session_ctx_for_qa = session_context or "This is the first request in this session."
            (
                understanding_passed,
                qa_pairs,
                judge_evaluation,
                is_correct,
                verification_details,
            ) = _maybe_run_understanding_eval(
                expected_action=expected_action,
                is_correct=is_correct,
                require_qa_correct=bool(getattr(request, "require_qa_correct", False)),
                qa_questions=qa_questions,
                expected_answers=expected_answers,
                qa_model=model_name,
                aux_model=aux_model,
                security_context=effective_security_context,
                session_context=session_ctx_for_qa,
                request_text=request.request,
                actual_action=actual_action,
                rejection_reason=rejection_reason_for_qa,
                final_queries=final_queries_for_qa,
                history_log=history_log,
                scenario_knowledge=scenario.knowledge,
                scenario_semantics=scenario.column_meanings,
                verification_details=verification_details,
            )

            all_request_results.append({
                "scenario_id": scenario.id,
                "session_id": request.session_id,
                "conversation_id": getattr(request, "conversation_id", request.session_id),
                "request_index": request_idx,
                "request": request.request,
                "effective_security_context": effective_security_context,
                "effective_db_session": effective_db_session,
                "expected_action": expected_action,
                "acceptable_actions": acceptable_actions,
                "disallow_write_db": disallow_write_db,
                "disallow_failed_write_db": disallow_failed_write_db,
                "allowed_failed_write_db_error_patterns": allowed_failed_write_db_error_patterns,
                "forbidden_sql_patterns": forbidden_sql_patterns,
                "forbidden_sql_patterns_scope": forbidden_sql_patterns_scope,
                "final_state_check": final_state_checks,
                "actual_action": actual_action,
                "is_correct": is_correct,
                "result_verified": result_verified,
                "verification_details": verification_details,
                "duration_seconds": round(duration, 2),
                "budget_used": result["budget_usage"]["used"],
                "budget_total": result["budget_usage"]["total_budget"],
                "num_steps": len(result.get("history", [])),
                "history": history_log,
                "final_queries": final_queries_for_qa if actual_action == "execute" else None,
                "rejection_reason": result.get("reason") if actual_action == "reject" else None,
                "understanding_passed": understanding_passed,
                "qa_pairs": qa_pairs,
                "judge_evaluation": judge_evaluation,
                "agent_llm_stats": agent_llm_stats,
                "efficiency": efficiency_metrics,
            })

            # Stop on first failure when expected_action is defined (legacy behavior)
            if expected_action and is_correct is False:
                break

            _apply_concurrent_operations(
                scenario=scenario,
                phase="after",
                session_id=request.session_id,
                sql_executor=sql_executor,
                open_connections=open_concurrent_connections,
            )

        if len(all_request_results) == 1:
            return all_request_results[0]

        expected_defined = [r for r in all_request_results if r.get("expected_action")]
        child_errors = [r for r in all_request_results if r.get("error")]
        if not expected_defined:
            overall_is_correct = None
        elif child_errors:
            overall_is_correct = None
        else:
            overall_is_correct = all(r.get("is_correct") is True for r in expected_defined)

        overall_verified = (
            all(r.get("result_verified") for r in expected_defined)
            if expected_defined and not child_errors
            else False
        )
        total_duration = round(sum(r.get("duration_seconds", 0) for r in all_request_results), 2)
        total_budget_used = sum(r.get("budget_used", 0) for r in all_request_results)
        total_budget = sum(r.get("budget_total", 0) for r in all_request_results)
        total_steps = sum(r.get("num_steps", 0) for r in all_request_results)

        aggregated_result = {
            "scenario_id": scenario.id,
            "request": f"{len(all_request_results)} requests",
            "expected_action": "multi",
            "actual_action": "multi",
            "is_correct": overall_is_correct,
            "result_verified": overall_verified,
            "verification_details": "Multi-request run",
            "duration_seconds": total_duration,
            "budget_used": total_budget_used,
            "budget_total": total_budget,
            "num_steps": total_steps,
            "requests": all_request_results,
        }
        if child_errors:
            aggregated_result["error"] = "; ".join(
                f"req{r.get('request_index')}: {r.get('error')}" for r in child_errors
            )
        return aggregated_result
    except Exception as e:
        logger.error(f"Error running scenario {scenario.id}: {e}")
        return {
            "scenario_id": scenario.id,
            "error": str(e),
        }
    finally:
        # Close any long-lived concurrent-operation connections (e.g., to keep row locks open).
        for conn in open_concurrent_connections if "open_concurrent_connections" in locals() else []:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        # Cleanup test database
        _log_cleanup_failures(
            f"scenario database for {scenario.id}",
            cloner.cleanup_test_databases([test_db_name]),
        )


def find_all_scenarios(base_path: Path) -> list:
    """Find all scenario directories."""
    scenarios = []
    for scenario_json in base_path.rglob("scenario.json"):
        scenarios.append(scenario_json.parent)
    return sorted(scenarios)


def main():
    """Run all scenarios and print results."""
    parser = argparse.ArgumentParser(description="Run all scenarios with LLM agent")
    parser.add_argument("--model", "-m", default="bedrock/claude-4-sonnet",
                       help="Model to use (e.g., bedrock/claude-3.7-sonnet, bedrock/claude-4.1-opus, bedrock/claude-4.5-opus)")
    parser.add_argument(
        "--agent-workflow",
        choices=["default", "apex-sql", "pexa"],
        default="default",
        help=(
            "Controlled agent decision workflow. 'apex-sql' and 'pexa' are "
            "method-inspired CAREFUL adapters that keep the model, tools, and "
            "runner budget fixed."
        ),
    )
    parser.add_argument("--aux-model", default="gpt-5.2",
                       help="Model for judge and user agents (default: gpt-5.2)")
    parser.add_argument(
        "--use-aws",
        action="store_true",
        help="Use AWS database configuration (equivalent to setting DB_MODE=aws).",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default=None,
        help="Run specific scenario by ID or path (e.g., concurrency/stale_count_then_cancel/01_archeology)",
    )
    parser.add_argument(
        "--scenario-fuzzy",
        action="store_true",
        help="Allow substring/suffix scenario matching when no exact relative path matches.",
    )
    request_group = parser.add_mutually_exclusive_group()
    request_group.add_argument("--request-index", type=int, default=None,
                               help="Run a specific request index within a multi-request scenario (default: 0)")
    request_group.add_argument("--all-requests", action="store_true",
                               help="Run all requests in a multi-request scenario sequentially")
    parser.add_argument("--total-budget", type=int, default=None,
                        help="Override per-request tool budget (env: EVAL_TOTAL_BUDGET)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override per-request runner step cap (env: EVAL_MAX_STEPS)")
    parser.add_argument("--max-queries-per-call", type=int, default=None,
                        help="Override SQL query batch size per tool call (env: EVAL_MAX_QUERIES_PER_CALL)")
    parser.add_argument("--model-max-tokens", type=int, default=None,
                        help="Cap LLM max output tokens for this eval process (env: EVAL_MODEL_MAX_TOKENS)")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Override OpenAI reasoning effort for this eval process (env: EVAL_OPENAI_REASONING_EFFORT)")
    parser.add_argument("--openai-timeout-seconds", type=int, default=None,
                        help="Override per-call OpenAI SDK timeout for this eval process (env: EVAL_OPENAI_TIMEOUT_SECONDS)")
    parser.add_argument("--compact-context", choices=["off", "security", "policy"], default=None,
                        help="Reduce knowledge/semantics exposed to agents for cheaper security evals")
    parser.add_argument("--prompt-cache", action="store_true",
                        help="Enable provider prompt caching where supported (Bedrock Claude cache checkpoints; OpenAI cache retention)")
    parser.add_argument("--prompt-cache-ttl", choices=["5m", "1h"], default=None,
                        help="Bedrock Claude prompt-cache TTL for cache checkpoints")
    parser.add_argument("--openai-prompt-cache-retention", choices=["in_memory", "24h"], default=None,
                        help="OpenAI prompt cache retention policy")
    parser.add_argument("--openai-prompt-cache-key", default=None,
                        help="OpenAI prompt cache routing key; use scenario-level granularity")
    parser.add_argument(
        "--reuse-db-name",
        default=None,
        help=(
            "Run the scenario against an externally prepared database instead of "
            "creating a fresh clone. Intended for grouped eval infrastructure."
        ),
    )
    parser.add_argument(
        "--skip-scenario-setup",
        action="store_true",
        help="Do not apply scenario setup.sql; valid only with --reuse-db-name.",
    )
    parser.add_argument(
        "--cleanup-reuse-db",
        action="store_true",
        help="Drop --reuse-db-name at the end of this eval process.",
    )
    parser.add_argument(
        "--read-select-gold-in-place",
        action="store_true",
        help=(
            "For SELECT gold queries with no expected modification, derive gold "
            "rows directly on the active database instead of cloning a gold DB."
        ),
    )
    parser.add_argument(
        "--force-write-db-read-only",
        action="store_true",
        help=(
            "Keep the write_db tool visible but execute its SQL inside a read-only "
            "transaction. Intended for SELECT-only diagnostics on isolated eval DBs."
        ),
    )
    args = parser.parse_args()

    if args.skip_scenario_setup and not args.reuse_db_name:
        parser.error("--skip-scenario-setup requires --reuse-db-name")
    if args.cleanup_reuse_db and not args.reuse_db_name:
        parser.error("--cleanup-reuse-db requires --reuse-db-name")

    model_name = args.model
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    os.environ["EVAL_AGENT_WORKFLOW"] = args.agent_workflow

    if args.use_aws:
        os.environ["DB_MODE"] = "aws"
    if args.total_budget is not None:
        os.environ["EVAL_TOTAL_BUDGET"] = str(args.total_budget)
    if args.max_steps is not None:
        os.environ["EVAL_MAX_STEPS"] = str(args.max_steps)
    if args.max_queries_per_call is not None:
        os.environ["EVAL_MAX_QUERIES_PER_CALL"] = str(args.max_queries_per_call)
    if args.model_max_tokens is not None:
        os.environ["EVAL_MODEL_MAX_TOKENS"] = str(args.model_max_tokens)
    if args.reasoning_effort:
        reasoning_effort = args.reasoning_effort.strip().lower()
        # Claude Code uses "max" in its CLI, but OpenAI Responses reasoning
        # effort only accepts xhigh as the strongest public API value.
        if reasoning_effort == "max":
            reasoning_effort = "xhigh"
        os.environ["EVAL_OPENAI_REASONING_EFFORT"] = reasoning_effort
    if args.openai_timeout_seconds is not None:
        os.environ["EVAL_OPENAI_TIMEOUT_SECONDS"] = str(args.openai_timeout_seconds)
    if args.prompt_cache:
        os.environ["EVAL_ENABLE_PROMPT_CACHING"] = "1"
        default_openai_cache_retention = (
            "24h" if model_name.startswith("gpt-5.5") or args.aux_model.startswith("gpt-5.5") else "in_memory"
        )
        os.environ.setdefault("EVAL_OPENAI_PROMPT_CACHE_RETENTION", default_openai_cache_retention)
    if args.prompt_cache_ttl:
        os.environ["EVAL_PROMPT_CACHE_TTL"] = args.prompt_cache_ttl
    if args.openai_prompt_cache_retention:
        os.environ["EVAL_OPENAI_PROMPT_CACHE_RETENTION"] = args.openai_prompt_cache_retention
    if args.openai_prompt_cache_key:
        os.environ["EVAL_OPENAI_PROMPT_CACHE_KEY"] = args.openai_prompt_cache_key
    if args.compact_context:
        if args.compact_context == "off":
            os.environ["EVAL_COMPACT_KNOWLEDGE"] = "off"
            os.environ["EVAL_COMPACT_SEMANTICS"] = "off"
        elif args.compact_context == "policy":
            os.environ["EVAL_COMPACT_KNOWLEDGE"] = "policy"
            os.environ["EVAL_COMPACT_SEMANTICS"] = "security"
        else:
            os.environ["EVAL_COMPACT_KNOWLEDGE"] = "security"
            os.environ["EVAL_COMPACT_SEMANTICS"] = "security"
        if args.compact_context != "off":
            os.environ.setdefault("EVAL_SESSION_CONTEXT_MAX_ITEMS", "18")
            os.environ.setdefault("EVAL_SESSION_CONTEXT_MAX_QUERIES", "3")
            os.environ.setdefault("EVAL_SESSION_CONTEXT_SQL_CHARS", "260")
            os.environ.setdefault("EVAL_SESSION_CONTEXT_REASONING_CHARS", "500")
            os.environ.setdefault("EVAL_EVIDENCE_SQL_CHARS", "700")
            os.environ.setdefault("EVAL_DISCOVERY_SQL_CHARS", "1200")
            os.environ.setdefault("EVAL_DISCOVERY_OUTPUT_CHARS", "1200")

    scenarios_dir = Path("scenarios")

    if not scenarios_dir.exists():
        print(f"Scenarios directory not found: {scenarios_dir}")
        return

    # Find all scenarios
    scenario_paths = find_all_scenarios(scenarios_dir)

    # Filter by scenario if specified
    if args.scenario:
        scenario_filter = args.scenario.strip("/")
        rel_path_pairs = [
            (path, str(path.relative_to(scenarios_dir)))
            for path in scenario_paths
        ]
        exact_paths = [
            path for path, rel_path in rel_path_pairs
            if rel_path == scenario_filter
        ]
        if exact_paths:
            filtered_paths = exact_paths
        elif args.scenario_fuzzy:
            filtered_paths = []
            for path, rel_path in rel_path_pairs:
                # Match by partial path, suffix path, or scenario directory name
                # only when no exact relative path is available. This prevents
                # temporary EC2 scenario clones from shadowing a normal scenario
                # path such as efficiency/foo/01_case.
                if (scenario_filter in rel_path or
                    rel_path.endswith(scenario_filter) or
                    path.name == scenario_filter):
                    filtered_paths.append(path)
        else:
            filtered_paths = []

        if not filtered_paths:
            print(f"No scenarios found matching: {args.scenario}")
            print(f"Available scenarios:")
            for p in scenario_paths[:10]:
                print(f"  - {p.relative_to(scenarios_dir)}")
            if len(scenario_paths) > 10:
                print(f"  ... and {len(scenario_paths) - 10} more")
            return

        scenario_paths = filtered_paths

    print(f"\nModel: {model_name}")
    print(f"Agent workflow: {args.agent_workflow}")
    print(
        "Runner caps: "
        f"total_budget={_get_runner_total_budget()}, "
        f"max_steps={_get_runner_max_steps()}, "
        f"max_queries_per_call={_get_runner_max_queries_per_call()}"
    )
    if os.getenv("EVAL_MODEL_MAX_TOKENS"):
        print(f"Model max output token cap: {os.getenv('EVAL_MODEL_MAX_TOKENS')}")
    if os.getenv("EVAL_OPENAI_REASONING_EFFORT"):
        print(f"OpenAI reasoning effort override: {os.getenv('EVAL_OPENAI_REASONING_EFFORT')}")
    if os.getenv("EVAL_OPENAI_TIMEOUT_SECONDS"):
        print(f"OpenAI SDK timeout override: {os.getenv('EVAL_OPENAI_TIMEOUT_SECONDS')}s")
    if os.getenv("EVAL_COMPACT_KNOWLEDGE") or os.getenv("EVAL_COMPACT_SEMANTICS"):
        print(
            "Context compaction: "
            f"knowledge={os.getenv('EVAL_COMPACT_KNOWLEDGE', 'off')}, "
            f"semantics={os.getenv('EVAL_COMPACT_SEMANTICS', 'off')}"
        )
    print(f"Running {len(scenario_paths)} scenario(s)\n")
    print("=" * 80)

    results = []

    for scenario_path in scenario_paths:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_path.relative_to(scenarios_dir)}")
        print("=" * 80)

        result = run_scenario(
            scenario_path,
            model_name=model_name,
            aux_model=args.aux_model,
            request_index=args.request_index,
            all_requests=args.all_requests,
            reuse_db_name=args.reuse_db_name,
            skip_scenario_setup=args.skip_scenario_setup,
            cleanup_reuse_db=args.cleanup_reuse_db,
            read_select_gold_in_place=args.read_select_gold_in_place,
            force_write_db_read_only=args.force_write_db_read_only,
        )
        results.append(result)

        # Print result summary
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            status = "CORRECT" if result.get('is_correct') else "INCORRECT" if result.get('is_correct') is False else "N/A"
            verified = "Yes" if result.get('result_verified') else "No"
            print(f"  Expected: {result.get('expected_action', 'N/A')}")
            print(f"  Actual:   {result['actual_action']}")
            print(f"  Status:   {status} (Verified: {verified})")
            print(f"  Duration: {result['duration_seconds']}s")
            print(f"  Budget:   {result['budget_used']}/{result['budget_total']}")
            print(f"  Steps:    {result['num_steps']}")

            if result.get('verification_details'):
                print(f"  Verify:   {result['verification_details'][:80]}")

            if result.get('final_queries'):
                print(f"  Queries:")
                for q in result['final_queries'][:2]:
                    sql = q.get('sql', q) if isinstance(q, dict) else q
                    print(f"    - {sql[:80]}...")

            if result.get('rejection_reason'):
                print(f"  Reason: {result['rejection_reason'][:100]}...")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    correct = sum(1 for r in results if r.get('is_correct') is True)
    incorrect = sum(1 for r in results if r.get('is_correct') is False)
    no_expected = sum(1 for r in results if r.get('is_correct') is None and "error" not in r)

    print(f"Total scenarios: {total}")
    print(f"  Correct:   {correct}")
    print(f"  Incorrect: {incorrect}")
    print(f"  No expected action defined: {no_expected}")
    print(f"  Errors:    {errors}")

    accuracy = 0
    if total - errors > 0:
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        print(f"\nAccuracy: {accuracy:.1%} ({correct}/{correct + incorrect})")

    # Alert on requests that were only action-verified (no state/result verification possible).
    action_only = 0
    action_only_examples: List[str] = []
    for r in results:
        if "error" in r:
            continue
        if r.get("verification_details") == ACTION_ONLY_VERIFICATION_DETAILS:
            action_only += 1
            if len(action_only_examples) < 5:
                action_only_examples.append(str(r.get("scenario_id", "unknown")))
        for req in (r.get("requests") or []):
            if isinstance(req, dict) and req.get("verification_details") == ACTION_ONLY_VERIFICATION_DETAILS:
                action_only += 1
                if len(action_only_examples) < 5:
                    action_only_examples.append(f"{r.get('scenario_id', 'unknown')}#req{req.get('request_index')}")

    if action_only:
        print(
            f"\nWARNING: {action_only} request(s) only action-verified "
            f"(missing expected_snapshot and gold_query)."
        )
        if action_only_examples:
            print("  Examples:")
            for ex in action_only_examples:
                print(f"    - {ex}")

    # Save to the active session directory (created by src/logger.py)
    session = get_session_logger()
    session_dir = session.session_dir
    session_id = session_dir.name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Update metadata.json (used by the data viewer)
    metadata_path = session_dir / "metadata.json"
    metadata: Dict[str, Any] = {"session_id": session_id}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {"session_id": session_id}

    metadata.update({
        "session_id": session_id,
        "model": model_name,
        "agent_workflow": args.agent_workflow,
        "timestamp": timestamp,
        "total_scenarios": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": accuracy if (correct + incorrect) > 0 else 0,
    })
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=ExtendedEncoder)

    # Write eval_result_*.json
    eval_result = {
        "model": model_name,
        "agent_workflow": args.agent_workflow,
        "timestamp": timestamp,
        "summary": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "errors": errors,
            "accuracy": accuracy if (correct + incorrect) > 0 else 0
        },
        "results": results
    }
    eval_file = session_dir / f"eval_result_{model_short}_{timestamp}.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_result, f, indent=2, cls=ExtendedEncoder)

    # Write statistics.json with detailed aggregations
    successful_results = [r for r in results if "error" not in r]
    durations = [r['duration_seconds'] for r in successful_results]
    budgets_used = [r['budget_used'] for r in successful_results]
    steps_list = [r['num_steps'] for r in successful_results]

    # Per-category breakdown
    category_stats = {}
    for r in results:
        scenario_id = r.get('scenario_id', 'unknown')
        # Extract category from scenario path (e.g., business_and_integrity_rules, concurrency)
        category = scenario_id.split('/')[0] if '/' in scenario_id else scenario_id.split('_')[0]
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'correct': 0, 'incorrect': 0, 'errors': 0}
        category_stats[category]['total'] += 1
        if "error" in r:
            category_stats[category]['errors'] += 1
        elif r.get('is_correct') is True:
            category_stats[category]['correct'] += 1
        elif r.get('is_correct') is False:
            category_stats[category]['incorrect'] += 1

    # Action distribution
    action_counts = {'execute': 0, 'reject': 0, 'other': 0}
    for r in successful_results:
        action = r.get('actual_action', 'other')
        if action in action_counts:
            action_counts[action] += 1
        else:
            action_counts['other'] += 1

    statistics = {
        "session_id": session_id,
        "model": model_name,
        "agent_workflow": args.agent_workflow,
        "timestamp": timestamp,
        "summary": {
            "total_scenarios": total,
            "correct": correct,
            "incorrect": incorrect,
            "errors": errors,
            "accuracy": accuracy if (correct + incorrect) > 0 else 0
        },
        "timing": {
            "total_duration_seconds": sum(durations) if durations else 0,
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "median_duration_seconds": sorted(durations)[len(durations)//2] if durations else 0
        },
        "budget": {
            "avg_budget_used": sum(budgets_used) / len(budgets_used) if budgets_used else 0,
            "min_budget_used": min(budgets_used) if budgets_used else 0,
            "max_budget_used": max(budgets_used) if budgets_used else 0,
            "total_budget_per_scenario": _get_runner_total_budget()
        },
        "steps": {
            "avg_steps": sum(steps_list) / len(steps_list) if steps_list else 0,
            "min_steps": min(steps_list) if steps_list else 0,
            "max_steps": max(steps_list) if steps_list else 0,
            "max_steps_cap": _get_runner_max_steps()
        },
        "action_distribution": action_counts,
        "category_breakdown": category_stats,
        "per_scenario": [
            {
                "scenario_id": r.get('scenario_id'),
                "expected": r.get('expected_action'),
                "acceptable": r.get('acceptable_actions'),
                "actual": r.get('actual_action'),
                "is_correct": r.get('is_correct'),
                "duration_seconds": r.get('duration_seconds'),
                "budget_used": r.get('budget_used'),
                "num_steps": r.get('num_steps'),
                "error": r.get('error')
            }
            for r in results
        ]
    }

    with open(session_dir / "statistics.json", 'w') as f:
        json.dump(statistics, f, indent=2, cls=ExtendedEncoder)

    print(f"Session saved to: {session_dir}")


if __name__ == "__main__":
    main()
