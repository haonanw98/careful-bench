"""
🧠 Anthropic Provider for DB-AgentSafe

Direct access to Claude models via Anthropic API.
Supports Claude 3 Opus, Sonnet, and Haiku.
"""

from typing import List, Optional
from ..interface import ModelInterface, ModelResponse, ModelCapabilities, Message, ToolChoice, ToolSchema
from ...logger import get_logger

logger = get_logger(__name__)


class AnthropicProvider(ModelInterface):
    """Provider for Anthropic Claude models."""
    
    def _initialize(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            # Only pass api_key if explicitly set, otherwise let library auto-detect from env
            if self.config.api_key:
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
            else:
                self.client = anthropic.Anthropic()  # Will use ANTHROPIC_API_KEY from env
            logger.info(f"Initialized Anthropic client for {self.config.model_name}")
            
        except ImportError:
            raise ImportError(
                "anthropic library required for Anthropic provider. "
                "Install with: pip install anthropic"
            )
    
    def _raw_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> ModelResponse:
        """Generate response using Anthropic API from chat messages."""
        if tools:
            raise ValueError("Native tools are not implemented for direct AnthropicProvider yet")
        try:
            # Separate system message if present (Anthropic supports system)
            system_text = None
            anth_messages = []
            for m in messages:
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                if role == "system":
                    system_text = m.content
                    continue
                if role not in ("user", "assistant"):
                    role = "user"
                anth_messages.append({"role": role, "content": m.content})

            kwargs = dict(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                messages=anth_messages,
                timeout=self.config.timeout,
            )
            if system_text:
                kwargs["system"] = system_text

            response = self.client.messages.create(**kwargs)
            
            content = response.content[0].text
            
            # Calculate cost estimate
            capabilities = self.get_capabilities()
            cost_estimate = (
                (response.usage.input_tokens / 1000) * capabilities.cost_per_1k_input_tokens +
                (response.usage.output_tokens / 1000) * capabilities.cost_per_1k_output_tokens
            )
            
            return ModelResponse(
                content=content,
                model_name=self.config.model_name,
                provider="anthropic",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                cost_estimate=cost_estimate,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason,
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def get_capabilities(self) -> ModelCapabilities:
        """Get Anthropic model capabilities."""
        model_configs = {
            "claude-3-opus-20240229": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=4096,
                supports_json_mode=False,
                supports_function_calling=False,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075,
                reasoning_capable=True
            ),
            "claude-3-sonnet-20240229": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=4096,
                supports_json_mode=False,
                supports_function_calling=False,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                reasoning_capable=True
            ),
            "claude-3-haiku-20240307": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=4096,
                supports_json_mode=False,
                supports_function_calling=False,
                cost_per_1k_input_tokens=0.00025,
                cost_per_1k_output_tokens=0.00125,
                reasoning_capable=True
            ),
        }
        
        return model_configs.get(
            self.config.model_name,
            ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=4096,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015
            )
        )
