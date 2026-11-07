"""
Type definitions for the Agent-Runner architecture.

This module defines all the data structures used for communication
between the Runner and Agent components.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, computed_field


class ActionType(str, Enum):
    """Types of actions an agent can take."""
    TOOL = "tool"
    FINISH = "finish"


class CostUnit(str, Enum):
    """How tool cost is calculated."""
    PER_CALL = "per_call"   # Fixed cost per tool invocation
    PER_ITEM = "per_item"   # Cost multiplied by number of items (e.g., queries)


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agent."""
    name: str
    description: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    cost: int = 1
    cost_unit: CostUnit = CostUnit.PER_CALL
    supports_partial: bool = False  # Can execute partial if budget limited

    model_config = {"frozen": True}


class SubResult(BaseModel):
    """Result of a single operation within a tool call (e.g., one SQL query)."""
    index: int
    input: Dict[str, Any] = Field(default_factory=dict)
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    row_count: Optional[int] = None

    model_config = {"frozen": True}


class ToolResult(BaseModel):
    """Result of a tool execution, may contain multiple sub-results."""
    executed: bool = True           # Tool handler ran (even if SQL failed)
    complete: bool = True           # Request fully satisfied vs truncated by budget
    sub_results: List[SubResult] = Field(default_factory=list)
    summary: str = ""               # Human-readable summary for prompt
    cost_incurred: int = 0
    error: Optional[str] = None     # Infrastructure failure only

    model_config = {"frozen": True}

    @computed_field
    @property
    def all_succeeded(self) -> bool:
        """True if all sub-results succeeded."""
        return self.executed and all(sr.success for sr in self.sub_results)

    @computed_field
    @property
    def success_count(self) -> int:
        """Number of successful sub-results."""
        return sum(1 for sr in self.sub_results if sr.success)

    @computed_field
    @property
    def failure_count(self) -> int:
        """Number of failed sub-results."""
        return sum(1 for sr in self.sub_results if not sr.success)


class HistoryStep(BaseModel):
    """Record of one step in the agent's execution history."""
    step_id: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    budget_before: int
    budget_after: int

    # What the agent requested
    action_type: ActionType
    action_name: Optional[str] = None       # Tool name or transition target
    action_params: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""

    # What happened
    tool_result: Optional[ToolResult] = None
    notes: Optional[str] = None             # Runner notes (validation errors, etc.)

    model_config = {"frozen": True}


class AgentContext(BaseModel):
    """Context provided by Runner to Agent for decision-making."""
    # Budget and limits
    budget_remaining: int
    budget_total: int
    max_queries_per_call: int
    max_rows_per_query: int

    # Task
    user_request: str
    database_name: str

    # Available tools
    available_tools: List[ToolDefinition] = Field(default_factory=list)
    session_tools: List[ToolDefinition] = Field(default_factory=list)

    # History
    history: List[HistoryStep] = Field(default_factory=list)
    clarifications: List[Dict[str, str]] = Field(default_factory=list)

    # Structured principal / authorization context for the current request
    security_context: Optional[Dict[str, Any]] = None

    # Session context (for multi-turn)
    session_context: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from Agent to Runner."""
    action_type: ActionType
    reasoning: str

    # For action_type="tool"
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None

    # For action_type="finish"
    is_rejection: bool = False
    rejection_reason: Optional[str] = None

    def validate_for_action(self) -> List[str]:
        """Validate that required fields are present for the action type."""
        errors = []

        if self.action_type == ActionType.TOOL:
            if not self.tool_name:
                errors.append("tool_name is required for action_type='tool'")

        elif self.action_type == ActionType.FINISH:
            if self.is_rejection and not self.rejection_reason:
                errors.append("rejection_reason is required when is_rejection=True")

        return errors
