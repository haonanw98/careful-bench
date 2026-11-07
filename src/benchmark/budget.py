"""
Budget management for the Agent-Runner architecture.

The BudgetManager tracks and enforces budget limits for tool execution.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from .types import ToolDefinition, CostUnit


@dataclass
class BudgetUsage:
    """Record of budget usage for a single tool call."""
    tool_name: str
    items_requested: int
    items_executed: int
    cost_incurred: int
    was_partial: bool


class BudgetManager:
    """
    Manages budget for agent tool execution.

    The Runner owns this and uses it to:
    - Check if a tool call can be afforded
    - Calculate partial execution limits
    - Track usage per tool
    - Enforce budget limits
    """

    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self._remaining = total_budget
        self._usage_history: List[BudgetUsage] = []
        self._usage_by_tool: Dict[str, int] = {}

    @property
    def remaining(self) -> int:
        """Current remaining budget."""
        return self._remaining

    @property
    def used(self) -> int:
        """Total budget used."""
        return self.total_budget - self._remaining

    def can_afford(self, tool: ToolDefinition, item_count: int = 1) -> bool:
        """
        Check if we can afford the requested tool execution.

        Args:
            tool: The tool definition
            item_count: Number of items requested (for per_item cost tools)

        Returns:
            True if we can afford the requested execution
        """
        count = max(1, int(item_count or 1))
        return self._remaining >= self.quote_cost(tool, count)

    def quote_cost(self, tool: ToolDefinition, item_count: int = 1) -> int:
        """
        Calculate the cost for a tool call.

        Args:
            tool: The tool definition
            item_count: Number of items (for per_item cost tools)

        Returns:
            Total cost for this tool call
        """
        if tool.cost_unit == CostUnit.PER_CALL:
            return tool.cost
        else:  # PER_ITEM
            return tool.cost * item_count

    def calculate_affordable_items(
        self,
        tool: ToolDefinition,
        requested_count: int
    ) -> Tuple[int, bool]:
        """
        Calculate how many items we can afford for a per_item tool.

        Args:
            tool: The tool definition
            requested_count: Number of items requested

        Returns:
            Tuple of (affordable_count, is_partial)
        """
        if tool.cost_unit == CostUnit.PER_CALL:
            # Fixed cost - either all or nothing
            if self._remaining >= tool.cost:
                return (requested_count, False)
            else:
                return (0, True)

        # PER_ITEM: calculate how many we can afford
        affordable = self._remaining // tool.cost

        if affordable >= requested_count:
            return (requested_count, False)
        elif affordable > 0 and tool.supports_partial:
            return (affordable, True)
        else:
            return (0, True)

    def charge(
        self,
        tool: ToolDefinition,
        items_requested: int,
        items_executed: int,
        was_partial: bool = False
    ) -> int:
        """
        Charge the budget for a tool execution.

        Args:
            tool: The tool definition
            items_requested: Number of items originally requested
            items_executed: Number of items actually executed
            was_partial: Whether this was a partial execution

        Returns:
            The cost charged
        """
        cost = self.quote_cost(tool, items_executed)

        if cost > self._remaining:
            raise ValueError(
                f"Cannot charge {cost} for {tool.name}: only {self._remaining} remaining"
            )

        self._remaining -= cost

        # Track usage
        usage = BudgetUsage(
            tool_name=tool.name,
            items_requested=items_requested,
            items_executed=items_executed,
            cost_incurred=cost,
            was_partial=was_partial,
        )
        self._usage_history.append(usage)

        # Update per-tool counter
        if tool.name not in self._usage_by_tool:
            self._usage_by_tool[tool.name] = 0
        self._usage_by_tool[tool.name] += cost

        return cost

    def get_usage_report(self) -> Dict[str, any]:
        """
        Get a report of budget usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_budget": self.total_budget,
            "remaining": self._remaining,
            "used": self.used,
            "usage_by_tool": dict(self._usage_by_tool),
            "call_count": len(self._usage_history),
            "partial_executions": sum(1 for u in self._usage_history if u.was_partial),
        }

    def get_min_tool_cost(self, tools: List[ToolDefinition]) -> int:
        """
        Get the minimum cost among a list of tools.

        Useful for checking if any tool can still be afforded.
        """
        if not tools:
            return float('inf')
        return min(self._calculate_min_cost(t) for t in tools)

    def _calculate_min_cost(self, tool: ToolDefinition) -> int:
        """Calculate minimum cost for one unit of a tool."""
        return tool.cost  # Both PER_CALL and PER_ITEM have same minimum

    def reset(self):
        """Reset budget to initial state (for new request in same session)."""
        self._remaining = self.total_budget
        self._usage_history = []
        self._usage_by_tool = {}
