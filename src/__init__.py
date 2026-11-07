"""
DatabaseAgentEval: Evaluation Framework for Database Agents

This package provides a framework for evaluating LLM-based agents
that perform database write operations.
"""

__version__ = "1.0.0"
__author__ = "DB-AgentSafe Team"

# Export main components from benchmark module
from .benchmark import (
    AgentRunner,
    RunnerConfig,
    TestScenario,
    SessionRequest,
    create_default_registry,
)

# Export agent implementation
from .agents.llm_agent import LLMAgent
from .agents.protocol import AgentProtocol

__all__ = [
    # Runner
    "AgentRunner",
    "RunnerConfig",
    "create_default_registry",
    # Scenario
    "TestScenario",
    "SessionRequest",
    # Agent
    "LLMAgent",
    "AgentProtocol",
]
