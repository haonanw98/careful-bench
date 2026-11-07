"""
Benchmark framework for evaluating database agents.

This module provides both the evaluation pipeline and the agent execution engine.

Agent-Runner Architecture:
- The Runner controls the environment (defines tools and enforces a tool budget)
- The Agent is a pure decision function (receives context, returns response)

Usage:
    from src.benchmark import AgentRunner, RunnerConfig, create_default_registry
    from src.agents.llm_agent import LLMAgent

    registry = create_default_registry()
    config = RunnerConfig(total_budget=20)
    runner = AgentRunner(registry, config)

    result = runner.run(
        agent=agent,
        user_request="...",
        database_name="...",
        context={...},
    )
"""

# Scenario loading
from .scenario import TestScenario, SessionRequest
from .evaluator import compare_query_results

# Agent-Runner types
from .types import (
    ActionType,
    CostUnit,
    ToolDefinition,
    SubResult,
    ToolResult,
    HistoryStep,
    AgentContext,
    AgentResponse,
)

# Budget management
from .budget import BudgetManager, BudgetUsage

# Tool registry
from .tools import ToolRegistry, create_default_registry

# Agent runner
from .runner import AgentRunner, RunnerConfig

__all__ = [
    # Scenario loading
    'TestScenario',
    'SessionRequest',
    'compare_query_results',
    # Types
    "ActionType",
    "CostUnit",
    "ToolDefinition",
    "SubResult",
    "ToolResult",
    "HistoryStep",
    "AgentContext",
    "AgentResponse",
    # Budget
    "BudgetManager",
    "BudgetUsage",
    # Tools
    "ToolRegistry",
    "create_default_registry",
    # Runner
    "AgentRunner",
    "RunnerConfig",
]
