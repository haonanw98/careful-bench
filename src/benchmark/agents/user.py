"""
User agent for realistic business stakeholder responses.

This agent represents a business user responding to database agent questions,
behaving like a real person who knows business context but not technical details.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from ...llm.factory import create_model
from ...llm.interface import Message, Role
from ...logger import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    """Represents a conversation turn."""
    role: Role
    content: str


class UserAgent:
    """Represents a realistic business stakeholder interacting with a database system."""
    
    def __init__(self, model_name: str = "sonnet4", system_prompt: str = None):
        """
        Initialize the user agent.
        
        Args:
            model_name: LLM model to use
            system_prompt: System prompt defining the user's role and knowledge
        """
        self.model_name = model_name
        self.llm = create_model(model_name)
        self.system_prompt = system_prompt or self._default_prompt()
        self.history: List[Turn] = []
        logger.info(f"UserAgent initialized with model: {model_name}")
    
    @staticmethod
    def _default_prompt() -> str:
        """Default prompt for a generic business user."""
        return """You are a non-technical business stakeholder.
• You understand business goals and context, not databases or SQL.
• Answer the agent's specific questions briefly based on your knowledge, only provide the specific answer for the question, don't provide any other information in the knowledge or hallucinate.
• If a question is vague, say you don't know and ask them to be specific."""
    
    def answer(self, question: str) -> str:
        """
        Answer a clarification question from the database agent.
        
        Args:
            question: The clarification question
            
        Returns:
            The user's response
        """
        messages: List[Message] = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        
        # Add conversation history
        for turn in self.history:
            messages.append(Message(role=turn.role, content=turn.content))
        
        # Add current question
        messages.append(Message(role=Role.USER, content=question))
        
        # Get response
        response = self.llm.generate(messages)
        reply = response.content
        
        # Update history
        self.history.append(Turn(Role.USER, question))
        self.history.append(Turn(Role.ASSISTANT, reply))
        
        return reply
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()


def build_user_system_prompt(
    request_text: str,
    intent: Optional[str] = None,
    user_knowledge: Optional[Union[str, Dict[str, Any]]] = None,
    extra_context: Optional[str] = None,
) -> str:
    """
    Build a scenario-aware system prompt for the business user persona.

    Includes:
    - Default persona with strict brevity and no invention
    - Original request for topical grounding
    - KNOWN FACTS from the scenario (string or dict)
    - Clarification guidance tailored for common flows (e.g., confirm names, transfers)
    """
    parts: list[str] = [UserAgent._default_prompt()]

    # Include the original request so answers stay on-topic
    if request_text:
        parts.append("")
        parts.append("ORIGINAL REQUEST:")
        parts.append(str(request_text))

    if user_knowledge:
        parts.append("")
        parts.append("KNOWN FACTS:")

        # Handle list format (new structure with question/answer/answer_type)
        if isinstance(user_knowledge, list):
            for item in user_knowledge:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    question = item['question']
                    answer = item['answer']

                    # Format based on question
                    if question.lower() == 'default':
                        # Default case - general instruction
                        parts.append(f"- For questions not covered: please answer EXACTLY this {answer}")
                    else:
                        parts.append(f"- If the question is semantically similar to \"{question}\": answer \"{answer}\"")
                else:
                    # Fallback for list items that aren't in the expected format
                    parts.append(f"- {str(item)}")

        # Handle dict format (legacy)
        elif isinstance(user_knowledge, dict):
            for k, v in user_knowledge.items():
                parts.append(f"- {k}: {v}")

        # Handle string format (legacy)
        else:
            parts.append(str(user_knowledge))

    if extra_context:
        parts.append("")
        parts.append("EXTRA CONTEXT:")
        parts.append(str(extra_context))
    return "\n".join(parts)
