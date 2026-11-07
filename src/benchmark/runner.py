"""
Main AgentRunner class for the Agent-Runner architecture.

The Runner is the "operating system" that:
- Enforces budget
- Executes tools
- Provides context to the agent
- Terminates only when the agent sends a FINISH action
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .budget import BudgetManager
from .tools import ToolRegistry
from .types import (
    ActionType,
    AgentContext,
    AgentResponse,
    CostUnit,
    HistoryStep,
    ToolDefinition,
    ToolResult,
)
from ..logger import get_logger
from ..agents.protocol import AgentProtocol

logger = get_logger(__name__)


class RunnerConfig:
    """Configuration for the AgentRunner."""

    def __init__(
        self,
        total_budget: int = 20,
        max_steps: int = 50,
        max_rows_per_query: int = 100,
        max_queries_per_call: int = 5,
    ):
        self.total_budget = int(total_budget)
        self.max_steps = int(max_steps)
        self.max_rows_per_query = int(max_rows_per_query)
        self.max_queries_per_call = int(max_queries_per_call)


class AgentRunner:
    """
    Main runner that controls agent execution.

    The Runner:
    - Defines available tools
    - Enforces budget
    - Executes tools
    - Provides context to the agent
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        config: Optional[RunnerConfig] = None,
    ):
        self.tool_registry = tool_registry
        self.config = config or RunnerConfig()

    def run(
        self,
        agent: AgentProtocol,
        user_request: str,
        database_name: str,
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
        session_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent loop until completion.

        Returns:
            {
              "action": "execute" | "reject",
              "reason": "..." (if reject),
              "history": [...],
              "budget_usage": {...},
            }
        """
        context = context or {}

        budget = BudgetManager(self.config.total_budget)
        history: List[HistoryStep] = []
        clarifications: List[Dict[str, str]] = []
        step_id = 0
        consecutive_no_tool_rounds = 0
        consecutive_insufficient_budget = 0
        consecutive_invalid_tool_rounds = 0

        # Tool helpers read these limits from runner context.
        context["max_rows"] = self.config.max_rows_per_query

        available_tools = self.tool_registry.get_all_definitions()
        if context.get("read_only"):
            # Expose only read-safe tools to the agent. Writes are still blocked at the tool layer
            # for defense-in-depth in case a model tries to invoke write_db anyway.
            available_tools = [t for t in available_tools if t.name != "write_db"]

        while True:
            step_id += 1

            if step_id > self.config.max_steps:
                logger.warning("Max steps exceeded, forcing completion")
                history.append(
                    self._create_error_step(
                        step_id=step_id,
                        budget=budget.remaining,
                        error_message="Max steps exceeded, forcing completion",
                    )
                )
                return self._build_result(
                    action="reject",
                    reason="Max steps exceeded",
                    history=history,
                    budget=budget,
                )

            affordable_tools = [t for t in available_tools if budget.can_afford(t)]
            if not affordable_tools:
                consecutive_no_tool_rounds += 1
                # Give the agent a chance to finish even when no tools are affordable.
                if consecutive_no_tool_rounds > 3:
                    return self._build_result(
                        action="reject",
                        reason="Budget exhausted",
                        history=history,
                        budget=budget,
                    )
            else:
                consecutive_no_tool_rounds = 0

            agent_context = AgentContext(
                budget_remaining=budget.remaining,
                budget_total=budget.total_budget,
                max_queries_per_call=self.config.max_queries_per_call,
                max_rows_per_query=self.config.max_rows_per_query,
                user_request=user_request,
                database_name=database_name,
                available_tools=affordable_tools,
                session_tools=available_tools,
                history=history,
                clarifications=clarifications,
                security_context=security_context,
                session_context=session_context,
            )

            try:
                response = agent.step(agent_context)
            except Exception as e:
                logger.error(f"Agent error: {e}")
                history.append(
                    self._create_error_step(
                        step_id=step_id,
                        budget=budget.remaining,
                        error_message=f"Agent error: {str(e)}",
                    )
                )
                continue

            validation_errors = response.validate_for_action()
            if validation_errors:
                history.append(
                    self._create_error_step(
                        step_id=step_id,
                        budget=budget.remaining,
                        error_message=f"Invalid response: {'; '.join(validation_errors)}",
                    )
                )
                continue

            budget_before = budget.remaining

            if response.action_type == ActionType.TOOL:
                result = self._handle_tool_action(response, budget, context)
                tool_result = result.get("tool_result")
                notes = result.get("message")

                if result["status"] == "ok":
                    try:
                        hook_notes = self._run_after_tool_call_hook(
                            response=response,
                            tool_result=tool_result,
                            context=context,
                        )
                    except Exception as e:
                        hook_notes = []
                        notes = self._join_notes(notes, f"after_tool_call hook failed: {e}")
                        history.append(
                            self._create_history_step(
                                step_id=step_id,
                                budget_before=budget_before,
                                budget_after=budget.remaining,
                                response=response,
                                tool_result=tool_result,
                                notes=notes,
                            )
                        )
                        return self._build_result(
                            action="reject",
                            reason=f"after_tool_call hook failed: {e}",
                            history=history,
                            budget=budget,
                        )
                    if hook_notes:
                        notes = self._join_notes(notes, "; ".join(hook_notes))

                history.append(
                    self._create_history_step(
                        step_id=step_id,
                        budget_before=budget_before,
                        budget_after=budget.remaining,
                        response=response,
                        tool_result=tool_result,
                        notes=notes,
                    )
                )

                if result["status"] != "ok":
                    if notes == "Insufficient budget":
                        consecutive_insufficient_budget += 1
                        if consecutive_insufficient_budget > 3:
                            logger.warning(
                                "Repeated unaffordable tool requests, forcing completion"
                            )
                            return self._build_result(
                                action="reject",
                                reason="Budget exhausted",
                                history=history,
                                budget=budget,
                            )
                        continue

                    consecutive_insufficient_budget = 0
                    consecutive_invalid_tool_rounds += 1
                    if consecutive_invalid_tool_rounds > 5:
                        logger.warning(
                            "Repeated invalid tool requests, forcing completion"
                        )
                        return self._build_result(
                            action="reject",
                            reason="Invalid tool requests",
                            history=history,
                            budget=budget,
                        )
                    else:
                        continue

                consecutive_insufficient_budget = 0
                consecutive_invalid_tool_rounds = 0

                if response.tool_name == "ask_user" and tool_result and tool_result.all_succeeded:
                    for sr in tool_result.sub_results:
                        if sr.success and isinstance(sr.output, dict):
                            clarifications.append(
                                {
                                    "question": sr.output.get("question", ""),
                                    "answer": sr.output.get("answer", ""),
                                }
                            )
                continue

            if response.action_type == ActionType.FINISH:
                history.append(
                    self._create_history_step(
                        step_id=step_id,
                        budget_before=budget_before,
                        budget_after=budget.remaining,
                        response=response,
                    )
                )

                if response.is_rejection:
                    return self._build_result(
                        action="reject",
                        reason=response.rejection_reason,
                        history=history,
                        budget=budget,
                    )
                return self._build_result(
                    action="execute",
                    history=history,
                    budget=budget,
                )

            # Future-proofing: only TOOL/FINISH are supported.
            history.append(
                self._create_error_step(
                    step_id=step_id,
                    budget=budget.remaining,
                    error_message=f"Unsupported action_type: {response.action_type}",
                )
            )

    def _handle_tool_action(
        self,
        response: AgentResponse,
        budget: BudgetManager,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if context.get("read_only") and response.tool_name == "write_db":
            return {"status": "error", "message": "write_db is not available in read-only mode"}

        tool_def = self.tool_registry.get_definition(response.tool_name)
        if tool_def is None:
            return {"status": "error", "message": f"Unknown tool: {response.tool_name}"}

        params = response.tool_params or {}
        if response.tool_name in {"explore_database", "test_query", "write_db"}:
            queries = params.get("queries")
            if not isinstance(queries, list) or not queries:
                return {
                    "status": "error",
                    "message": "Invalid tool_params: expected non-empty 'queries' list",
                }
            for query in queries:
                if isinstance(query, str):
                    sql = query.strip()
                elif isinstance(query, dict):
                    sql = str(query.get("sql", "") or "").strip()
                else:
                    return {
                        "status": "error",
                        "message": "Invalid tool_params: each query must be a string or object with 'sql'",
                    }
                if not sql:
                    return {
                        "status": "error",
                        "message": "Invalid tool_params: each query requires non-empty 'sql'",
                    }
        if "queries" in params and isinstance(params.get("queries"), list):
            if len(params["queries"]) > self.config.max_queries_per_call:
                return {
                    "status": "error",
                    "message": (
                        f"Too many queries in one call: requested {len(params['queries'])}, "
                        f"max is {self.config.max_queries_per_call}"
                    ),
                }
        if response.tool_name == "ask_user":
            question = str(params.get("question", "") or "").strip()
            if not question:
                return {"status": "error", "message": "Invalid tool_params: 'question' is required"}

        item_count = self._count_items_for_billing(tool_def, params)
        if not budget.can_afford(tool_def, item_count=item_count):
            return {"status": "error", "message": "Insufficient budget"}

        tool_result = self.tool_registry.execute(
            response.tool_name,
            params,
            context,
        )

        charged = budget.charge(
            tool_def,
            items_requested=item_count,
            items_executed=self._count_items_executed(tool_def, tool_result, item_count),
            was_partial=False,
        )

        tool_result = ToolResult(
            executed=tool_result.executed,
            complete=tool_result.complete,
            sub_results=tool_result.sub_results,
            summary=tool_result.summary,
            cost_incurred=charged,
            error=tool_result.error,
        )

        return {"status": "ok", "tool_result": tool_result}

    @staticmethod
    def _run_after_tool_call_hook(
        *,
        response: AgentResponse,
        tool_result: Optional[ToolResult],
        context: Dict[str, Any],
    ) -> List[str]:
        hook = context.get("after_tool_call_hook")
        if not callable(hook):
            return []

        outcome = hook(
            tool_name=response.tool_name,
            tool_params=response.tool_params or {},
            tool_result=tool_result,
        )
        if outcome is None:
            return []
        if isinstance(outcome, str):
            return [outcome]
        if isinstance(outcome, list):
            return [str(item) for item in outcome if str(item)]
        return [str(outcome)]

    @staticmethod
    def _join_notes(existing: Optional[str], addition: str) -> str:
        if existing:
            return f"{existing}; {addition}"
        return addition

    @staticmethod
    def _count_items_for_billing(tool_def: ToolDefinition, params: Dict[str, Any]) -> int:
        if tool_def.cost_unit == CostUnit.PER_CALL:
            return 1
        items = params.get("queries", [])
        return len(items) if isinstance(items, list) else 1

    @staticmethod
    def _count_items_executed(tool_def: ToolDefinition, tool_result: ToolResult, requested: int) -> int:
        if tool_def.cost_unit == CostUnit.PER_CALL:
            return 1
        if tool_result.executed:
            return requested
        return 0

    @staticmethod
    def _create_history_step(
        step_id: int,
        budget_before: int,
        budget_after: int,
        response: AgentResponse,
        tool_result: Optional[ToolResult] = None,
        notes: Optional[str] = None,
    ) -> HistoryStep:
        return HistoryStep(
            step_id=step_id,
            timestamp=datetime.utcnow().isoformat(),
            budget_before=budget_before,
            budget_after=budget_after,
            action_type=response.action_type,
            action_name=response.tool_name,
            action_params=response.tool_params or {},
            reasoning=response.reasoning,
            tool_result=tool_result,
            notes=notes,
        )

    @staticmethod
    def _create_error_step(step_id: int, budget: int, error_message: str) -> HistoryStep:
        return HistoryStep(
            step_id=step_id,
            timestamp=datetime.utcnow().isoformat(),
            budget_before=budget,
            budget_after=budget,
            action_type=ActionType.TOOL,
            action_name="_error",
            action_params={},
            reasoning="",
            notes=error_message,
        )

    @staticmethod
    def _build_result(
        action: str,
        history: List[HistoryStep],
        budget: BudgetManager,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "action": action,
            "history": history,
            "budget_usage": budget.get_usage_report(),
        }
        if action == "reject":
            result["reason"] = reason or "Request rejected"
        return result
