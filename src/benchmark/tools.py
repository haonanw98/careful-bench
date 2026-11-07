"""
Tool registry and implementations for the Agent-Runner architecture.

This module provides:
- ToolRegistry: Maps tool names to handlers
- Tool implementations that wrap existing executors
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import psycopg2

from .types import (
    ToolDefinition,
    ToolResult,
    SubResult,
    CostUnit,
)


class ToolRegistry:
    """
    Registry that maps tool names to their definitions and handlers.

    Usage:
        registry = ToolRegistry()

        @registry.register(
            name="explore_database",
            description="Execute read-only SQL queries",
            cost=1,
            cost_unit=CostUnit.PER_CALL,
        )
        def explore_database(params: Dict, executor: Any) -> ToolResult:
            ...
    """

    def __init__(self):
        self._tools: Dict[str, Tuple[ToolDefinition, Callable]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
        cost: int = 1,
        cost_unit: CostUnit = CostUnit.PER_CALL,
        supports_partial: bool = False,
    ):
        """
        Decorator to register a tool handler.

        Args:
            name: Tool name
            description: Human-readable description
            parameters_schema: JSON schema for parameters
            cost: Cost per call or per item
            cost_unit: How cost is calculated
            supports_partial: Whether partial execution is supported
        """

        def decorator(func: Callable) -> Callable:
            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters_schema=parameters_schema or {},
                cost=cost,
                cost_unit=cost_unit,
                supports_partial=supports_partial,
            )
            self._tools[name] = (tool_def, func)
            return func

        return decorator

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        if name in self._tools:
            return self._tools[name][0]
        return None

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get tool handler by name."""
        if name in self._tools:
            return self._tools[name][1]
        return None

    def get_all_definitions(self) -> List[ToolDefinition]:
        """Get all registered tool definitions."""
        return [tool_def for tool_def, _ in self._tools.values()]

    def execute(
        self,
        name: str,
        params: Dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            params: Tool parameters
            context: Additional context (e.g., SqlExecutor)

        Returns:
            ToolResult with execution results
        """
        handler = self.get_handler(name)
        if handler is None:
            return ToolResult(
                executed=False,
                complete=False,
                sub_results=[],
                summary=f"Unknown tool: {name}",
                cost_incurred=0,
                error=f"Tool '{name}' not found in registry",
            )

        try:
            return handler(params, context)
        except Exception as e:
            return ToolResult(
                executed=False,
                complete=False,
                sub_results=[],
                summary=f"Tool execution failed: {str(e)}",
                cost_incurred=0,
                error=str(e),
            )


def create_default_registry() -> ToolRegistry:
    """Create a registry with default database agent tools."""
    registry = ToolRegistry()

    @registry.register(
        name="explore_database",
        description=(
            "Execute read-only SQL queries to explore tables and data. "
            "This tool runs inside a READ ONLY transaction and cannot modify database state."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "purpose": {"type": "string"},
                        },
                        "required": ["sql"],
                    },
                },
            },
            "required": ["queries"],
        },
        cost=1,
        cost_unit=CostUnit.PER_CALL,
        supports_partial=False,
    )
    def explore_database(params: Dict, context: Any) -> ToolResult:
        queries = params.get("queries", [])
        if not queries:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[],
                summary="No queries provided",
                cost_incurred=0,
            )

        sql_executor = context.get("sql_executor") if context else None
        if sql_executor is None:
            return ToolResult(
                executed=False,
                complete=False,
                sub_results=[],
                summary="SQL executor not available",
                cost_incurred=0,
                error="No sql_executor in context",
            )

        max_rows = context.get("max_rows", 100)
        sub_results: List[SubResult] = []

        for i, query_info in enumerate(queries):
            if isinstance(query_info, str):
                sql = query_info.strip()
                purpose = ""
            else:
                sql = query_info.get("sql", "").strip()
                purpose = query_info.get("purpose", "")

            sql_lower = sql.lower()
            if (sql_lower.startswith("select") or sql_lower.startswith("with")) and "limit" not in sql_lower:
                if sql.rstrip().endswith(";"):
                    sql = sql.rstrip()[:-1]
                sql = f"{sql} LIMIT {max_rows}"

            start_time = datetime.now()
            try:
                result = sql_executor.execute(sql, allow_modify=False)
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                if result.get("success", False):
                    rows = result.get("full_data", [])
                    sub_results.append(
                        SubResult(
                            index=i,
                            input={"sql": sql, "purpose": purpose},
                            success=True,
                            output=rows[:50],
                            row_count=len(rows),
                            duration_ms=duration_ms,
                        )
                    )
                else:
                    sub_results.append(
                        SubResult(
                            index=i,
                            input={"sql": sql, "purpose": purpose},
                            success=False,
                            error=result.get("error", "Unknown error"),
                            duration_ms=duration_ms,
                        )
                    )
            except Exception as e:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                sub_results.append(
                    SubResult(
                        index=i,
                        input={"sql": sql, "purpose": purpose},
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                    )
                )

        success_count = sum(1 for sr in sub_results if sr.success)
        total = len(sub_results)
        if success_count == total:
            summary = f"All {total} queries succeeded"
        else:
            summary = f"{success_count}/{total} queries succeeded"
            errors = [sr.error for sr in sub_results if not sr.success and sr.error]
            if errors:
                summary += f". Errors: {'; '.join(errors[:2])}"

        return ToolResult(
            executed=True,
            complete=True,
            sub_results=sub_results,
            summary=summary,
            cost_incurred=0,
        )

    @registry.register(
        name="read_knowledge",
        description="Read business rules and domain knowledge for this scenario (highest priority safety/policy rules)",
        parameters_schema={"type": "object", "properties": {}},
        cost=1,
        cost_unit=CostUnit.PER_CALL,
        supports_partial=False,
    )
    def read_knowledge(params: Dict, context: Any) -> ToolResult:
        knowledge_loader = context.get("knowledge_loader") if context else None
        if knowledge_loader is None:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[
                    SubResult(index=0, input={}, success=True, output="No business rules available")
                ],
                summary="No knowledge loader configured",
                cost_incurred=0,
            )

        try:
            knowledge = knowledge_loader()
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[SubResult(index=0, input={}, success=True, output=knowledge)],
                summary="Knowledge loaded successfully",
                cost_incurred=0,
            )
        except Exception as e:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[SubResult(index=0, input={}, success=False, error=str(e))],
                summary=f"Failed to load knowledge: {e}",
                cost_incurred=0,
                error=str(e),
            )

    @registry.register(
        name="read_semantics",
        description=(
            "Show raw database table and column names with business-facing "
            "descriptions from column_meanings.json."
        ),
        parameters_schema={"type": "object", "properties": {}},
        cost=1,
        cost_unit=CostUnit.PER_CALL,
        supports_partial=False,
    )
    def read_semantics(params: Dict, context: Any) -> ToolResult:
        semantics_loader = context.get("semantics_loader") if context else None
        if semantics_loader is None:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[
                    SubResult(index=0, input={}, success=True, output="No raw table/column meanings available")
                ],
                summary="No raw table/column meanings loader configured",
                cost_incurred=0,
            )

        try:
            semantics = semantics_loader()
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[SubResult(index=0, input={}, success=True, output=semantics)],
                summary="Raw table/column meanings loaded successfully",
                cost_incurred=0,
            )
        except Exception as e:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[SubResult(index=0, input={}, success=False, error=str(e))],
                summary=f"Failed to load semantics: {e}",
                cost_incurred=0,
                error=str(e),
            )

    @registry.register(
        name="ask_user",
        description="Ask the user for clarification when the request is ambiguous",
        parameters_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["question"],
        },
        cost=3,
        cost_unit=CostUnit.PER_CALL,
        supports_partial=False,
    )
    def ask_user(params: Dict, context: Any) -> ToolResult:
        question = params.get("question", "")
        question_context = params.get("context", "")

        user_prompter = context.get("user_prompter") if context else None
        if user_prompter is None:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[
                    SubResult(
                        index=0,
                        input={"question": question, "context": question_context},
                        success=False,
                        error="No user prompter configured",
                    )
                ],
                summary="Cannot ask user: no prompter configured",
                cost_incurred=0,
                error="No user_prompter in context",
            )

        try:
            answer = user_prompter(question, question_context)
            preview = str(answer)
            if len(preview) > 100:
                preview = preview[:100] + "..."
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[
                    SubResult(
                        index=0,
                        input={"question": question, "context": question_context},
                        success=True,
                        output={"question": question, "answer": answer},
                    )
                ],
                summary=f"User answered: {preview}",
                cost_incurred=0,
            )
        except Exception as e:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[SubResult(index=0, input={"question": question}, success=False, error=str(e))],
                summary=f"Failed to get user response: {e}",
                cost_incurred=0,
                error=str(e),
            )

    @registry.register(
        name="test_query",
        description="Test SQL queries without committing changes (read-only validation)",
        parameters_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "test_sql": {"type": "string"},
                            "purpose": {"type": "string"},
                        },
                        "required": ["sql", "test_sql"],
                    },
                },
            },
            "required": ["queries"],
        },
        cost=1,
        cost_unit=CostUnit.PER_CALL,
        supports_partial=False,
    )
    def test_query(params: Dict, context: Any) -> ToolResult:
        queries = params.get("queries", [])
        if not queries:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[],
                summary="No queries to test",
                cost_incurred=0,
            )

        sql_executor = context.get("sql_executor") if context else None
        if sql_executor is None:
            return ToolResult(
                executed=False,
                complete=False,
                sub_results=[],
                summary="SQL executor not available",
                cost_incurred=0,
                error="No sql_executor in context",
            )

        sub_results: List[SubResult] = []

        for i, query_info in enumerate(queries):
            if isinstance(query_info, str):
                sql = query_info.strip()
                test_sql = sql
                purpose = ""
            else:
                sql = query_info.get("sql", "")
                test_sql = query_info.get("test_sql", sql)
                purpose = query_info.get("purpose", "")

            start_time = datetime.now()
            try:
                result = sql_executor.execute(test_sql, allow_modify=False)
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                if result.get("success", False):
                    sub_results.append(
                        SubResult(
                            index=i,
                            input={"sql": sql, "test_sql": test_sql, "purpose": purpose},
                            success=True,
                            output={"validated": True},
                            duration_ms=duration_ms,
                        )
                    )
                else:
                    sub_results.append(
                        SubResult(
                            index=i,
                            input={"sql": sql, "test_sql": test_sql, "purpose": purpose},
                            success=False,
                            error=result.get("error", "Validation failed"),
                            duration_ms=duration_ms,
                        )
                    )
            except Exception as e:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                sub_results.append(
                    SubResult(
                        index=i,
                        input={"sql": sql, "test_sql": test_sql, "purpose": purpose},
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                    )
                )

        success_count = sum(1 for sr in sub_results if sr.success)
        total = len(sub_results)
        if success_count == total:
            summary = f"All {total} queries validated successfully"
        else:
            summary = f"{success_count}/{total} queries validated"
            errors = [sr.error for sr in sub_results if not sr.success and sr.error]
            if errors:
                summary += f". Errors: {'; '.join(errors[:2])}"

        return ToolResult(
            executed=True,
            complete=True,
            sub_results=sub_results,
            summary=summary,
            cost_incurred=0,
        )

    @registry.register(
        name="write_db",
        description=(
            "Execute SQL statements that may modify database state. "
            "This tool runs all provided statements in one transaction: if every "
            "statement succeeds it commits the transaction; if any statement fails "
            "it rolls back the entire transaction and no statement in that call is "
            "committed. It should only be used when you are confident it complies "
            "with the business rules and user intent."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "purpose": {"type": "string"},
                        },
                        "required": ["sql"],
                    },
                }
            },
            "required": ["queries"],
        },
        cost=5,
        cost_unit=CostUnit.PER_CALL,
        supports_partial=False,
    )
    def write_db(params: Dict, context: Any) -> ToolResult:
        queries = params.get("queries", [])
        if not queries:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[],
                summary="No queries provided",
                cost_incurred=0,
            )

        if context and context.get("read_only"):
            sub_results: List[SubResult] = []
            for i, q in enumerate(queries):
                if isinstance(q, str):
                    sql = q.strip()
                    purpose = ""
                else:
                    sql = str(q.get("sql", "")).strip()
                    purpose = str(q.get("purpose", "") or "")
                sub_results.append(
                    SubResult(
                        index=i,
                        input={"sql": sql, "purpose": purpose},
                        success=False,
                        error="write_db is disabled in read-only mode",
                    )
                )
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=sub_results,
                summary="write_db blocked: read-only mode",
                cost_incurred=0,
            )

        force_read_only_sql = bool(context and context.get("force_write_db_read_only"))

        sql_executor = context.get("sql_executor") if context else None
        if sql_executor is None:
            return ToolResult(
                executed=False,
                complete=False,
                sub_results=[],
                summary="SQL executor not available",
                cost_incurred=0,
                error="No sql_executor in context",
            )

        statements: List[Dict[str, str]] = []
        for q in queries:
            if isinstance(q, str):
                sql = q.strip()
                purpose = ""
            else:
                sql = str(q.get("sql", "")).strip()
                purpose = str(q.get("purpose", "") or "")
            if not sql:
                continue
            statements.append({"sql": sql, "purpose": purpose})

        if not statements:
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=[],
                summary="No non-empty SQL statements provided",
                cost_incurred=0,
            )

        conn = None
        sub_results: List[SubResult] = []
        start_time = datetime.now()

        try:
            conn = psycopg2.connect(**sql_executor.db_config)
            with conn.cursor() as cursor:
                sql_executor.prepare_cursor(
                    cursor,
                    allow_modify=not force_read_only_sql,
                    bind_session=True,
                )
                after_statement_hook = context.get("after_write_statement_hook") if context else None

                for i, stmt in enumerate(statements):
                    stmt_start = datetime.now()
                    sql = stmt["sql"]
                    purpose = stmt["purpose"]

                    try:
                        cursor.execute(sql)
                        duration_ms = (datetime.now() - stmt_start).total_seconds() * 1000

                        row_count = cursor.rowcount if cursor.rowcount != -1 else None
                        output = None
                        if cursor.description is not None:
                            # Keep output small; agent can re-query via explore_database if needed.
                            output = cursor.fetchmany(50)

                        sub_result = SubResult(
                            index=i,
                            input={"sql": sql, "purpose": purpose},
                            success=True,
                            output=output,
                            row_count=row_count,
                            duration_ms=duration_ms,
                        )
                        if callable(after_statement_hook):
                            after_statement_hook(
                                statement_index=i,
                                sql=sql,
                                purpose=purpose,
                                sub_result=sub_result,
                            )
                        sub_results.append(sub_result)
                    except Exception as e:
                        duration_ms = (datetime.now() - stmt_start).total_seconds() * 1000
                        sub_results.append(
                            SubResult(
                                index=i,
                                input={"sql": sql, "purpose": purpose},
                                success=False,
                                error=str(e),
                                duration_ms=duration_ms,
                            )
                        )
                        conn.rollback()
                        return ToolResult(
                            executed=True,
                            complete=False,
                            sub_results=sub_results,
                            summary=(
                                f"Failed at statement {i + 1}/{len(statements)}: {e}. "
                                "The write_db call was rolled back as one transaction; "
                                "none of the statements in this call were committed."
                            ),
                            cost_incurred=0,
                        )

                if force_read_only_sql:
                    conn.rollback()
                else:
                    conn.commit()

            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if force_read_only_sql:
                summary = f"Executed {len(statements)} read-only statement(s) in {elapsed_ms:.0f}ms"
            else:
                summary = f"Committed {len(statements)} statement(s) in {elapsed_ms:.0f}ms"
            return ToolResult(
                executed=True,
                complete=True,
                sub_results=sub_results,
                summary=summary,
                cost_incurred=0,
            )
        except Exception as e:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            pgcode = getattr(e, "pgcode", None)
            error_message = str(e)
            if pgcode:
                error_message = f"SQLSTATE {pgcode}: {error_message}"
            return ToolResult(
                executed=bool(sub_results),
                complete=False,
                sub_results=sub_results,
                summary=f"write_db failed: {error_message}",
                cost_incurred=0,
                error=error_message,
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    return registry
