"""
Agent protocol and base implementations.

This module defines the interface that all agents must implement
to work with the AgentRunner.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from ..benchmark.types import AgentContext, AgentResponse


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol that all agents must implement.

    The agent is a pure decision function:
    - Receives AgentContext from the Runner
    - Returns AgentResponse indicating what action to take
    - Does NOT execute tools or manage state
    """

    def step(self, context: AgentContext) -> AgentResponse:
        """
        Given context, decide what action to take.

        Args:
            context: The current context provided by the Runner,
                    including budget, tool limits, available tools, history, etc.

        Returns:
            AgentResponse indicating:
            - action_type: "tool" or "finish"
            - tool_name/tool_params: for tool actions
            - is_rejection/rejection_reason: for finish actions
            - reasoning: explanation of the decision
        """
        ...


class BaseAgent(ABC):
    """
    Abstract base class for agents.

    Provides a cleaner interface than the Protocol for implementations
    that want to use inheritance.
    """

    @abstractmethod
    def step(self, context: AgentContext) -> AgentResponse:
        """
        Given context, decide what action to take.

        Args:
            context: The current context provided by the Runner

        Returns:
            AgentResponse indicating the next action
        """
        pass

    def reset(self):
        """
        Reset any internal state for a new session.

        Override if your agent maintains internal state.
        Default implementation does nothing.
        """
        pass


class SimpleAgent(BaseAgent):
    """
    A simple rule-based agent for testing.

    This agent follows a fixed strategy:
    1) Read knowledge once
    2) Reject the request

    Useful for testing the runner without an LLM.
    """

    def __init__(self):
        self._step_count = 0

    def step(self, context: AgentContext) -> AgentResponse:
        """Simple rule-based decision making."""
        from ..benchmark.types import ActionType

        self._step_count += 1

        if self._step_count == 1:
            return AgentResponse(
                action_type=ActionType.TOOL,
                reasoning="Reading business rules first",
                tool_name="read_knowledge",
                tool_params={},
            )

        return AgentResponse(
            action_type=ActionType.FINISH,
            reasoning="This is a test agent that always rejects",
            is_rejection=True,
            rejection_reason="SimpleAgent always rejects requests",
        )

    def reset(self):
        """Reset step counter."""
        self._step_count = 0
