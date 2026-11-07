"""
Agents for the benchmark framework.
"""

from .llm_agent import LLMAgent
from .protocol import AgentProtocol, BaseAgent, SimpleAgent

__all__ = [
    "LLMAgent",
    "AgentProtocol",
    "BaseAgent",
    "SimpleAgent",
]
