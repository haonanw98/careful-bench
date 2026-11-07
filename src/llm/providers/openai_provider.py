"""
🔥 OpenAI provider.

Supports chat models plus reasoning-oriented OpenAI models such as GPT-5.x and
o-series models.
"""

import json
from typing import Any, List, Dict, Optional
from ..interface import (
    ModelInterface,
    ModelResponse,
    ModelCapabilities,
    Message,
    ToolCall,
    ToolChoice,
    ToolSchema,
)
from ...logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(ModelInterface):
    """Provider for OpenAI models."""

    @staticmethod
    def _normalize_messages(messages: List[Message]) -> List[Dict[str, str]]:
        """Convert internal messages to OpenAI-compatible role/content dictionaries."""
        oa_messages: List[Dict[str, str]] = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            if role not in ("system", "user", "assistant"):
                role = "user"
            oa_messages.append({"role": role, "content": OpenAIProvider._content_to_text(m.content)})
        return oa_messages

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        try:
            import json
            return json.dumps(content, default=str)
        except Exception:
            return str(content)

    @staticmethod
    def _usage_int(value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    @staticmethod
    def _cached_input_tokens(usage: Any) -> int:
        if usage is None:
            return 0
        details = (
            getattr(usage, "input_tokens_details", None)
            or getattr(usage, "prompt_tokens_details", None)
        )
        return OpenAIProvider._usage_int(getattr(details, "cached_tokens", 0))

    def _apply_prompt_cache_params(self, params: Dict[str, Any], *, include_retention: bool) -> None:
        retention = str(self.config.provider_settings.get("prompt_cache_retention") or "").strip()
        cache_key = str(self.config.provider_settings.get("prompt_cache_key") or "").strip()
        if include_retention and retention:
            params["prompt_cache_retention"] = retention
        if cache_key:
            params["prompt_cache_key"] = cache_key

    @staticmethod
    def _is_reasoning_series(model_name: str) -> bool:
        """Return True for OpenAI reasoning-oriented model families."""
        return model_name.startswith("o") or model_name.startswith("gpt-5")

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any, tool_name: str = "") -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if raw_arguments in (None, ""):
            return {}
        try:
            parsed = json.loads(str(raw_arguments))
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and tool_name in {"explore_database", "test_query", "write_db"}:
                return {"queries": parsed}
            return {}
        except Exception:
            return {}

    @staticmethod
    def _responses_tool_schema(tool: ToolSchema) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
            "strict": False,
        }

    @staticmethod
    def _chat_tool_schema(tool: ToolSchema) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
                "strict": False,
            },
        }

    @staticmethod
    def _tool_choice_value(tool_choice: Optional[ToolChoice]) -> Optional[str]:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, ToolChoice):
            return tool_choice.value
        return str(tool_choice)
    
    def _initialize(self):
        """Initialize OpenAI client."""
        try:
            import openai
            
            # Configure client with newer SDK style
            client_kwargs = {}
            # Only pass api_key if explicitly set; otherwise let the SDK read OPENAI_API_KEY from env.
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base
            # Optional organization/project (picked up from env via factory)
            org = self.config.provider_settings.get("organization")
            proj = self.config.provider_settings.get("project")
            if org:
                client_kwargs["organization"] = org
            if proj:
                client_kwargs["project"] = proj

            self.client = openai.OpenAI(**client_kwargs)
            
            logger.info(f"Initialized OpenAI client for {self.config.model_name}")
            
        except ImportError:
            raise ImportError(
                "OpenAI library required for OpenAI provider. "
                "Install with: pip install openai"
            )
    
    def _raw_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> ModelResponse:
        """Generate response using OpenAI API from chat messages."""
        try:
            oa_messages = self._normalize_messages(messages)
            model_name = self.config.model_name

            if self._is_reasoning_series(model_name):
                reasoning: Dict[str, str] = {}
                effort = self.config.provider_settings.get("reasoning_effort")
                summary = self.config.provider_settings.get("reasoning_summary")
                if effort:
                    reasoning["effort"] = str(effort)
                if summary:
                    reasoning["summary"] = str(summary)

                params = {
                    "model": model_name,
                    "input": oa_messages,
                    "timeout": self.config.timeout,
                    "max_output_tokens": self.config.max_tokens,
                }
                if tools:
                    params["tools"] = [self._responses_tool_schema(tool) for tool in tools]
                    params["parallel_tool_calls"] = False
                    choice_value = self._tool_choice_value(tool_choice)
                    if choice_value:
                        params["tool_choice"] = choice_value
                if reasoning:
                    params["reasoning"] = reasoning
                self._apply_prompt_cache_params(params, include_retention=True)

                response = self.client.responses.create(**params)
                usage = response.usage
                cached_input_tokens = self._cached_input_tokens(usage)
                input_tokens = self._usage_int(getattr(usage, "input_tokens", 0))
                output_tokens = self._usage_int(getattr(usage, "output_tokens", 0))
                total_tokens = self._usage_int(getattr(usage, "total_tokens", 0)) or input_tokens + output_tokens
                reasoning_tokens = None
                if usage and getattr(usage, "output_tokens_details", None) is not None:
                    reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", None)
                incomplete_details = getattr(response, "incomplete_details", None)
                if incomplete_details is not None:
                    if hasattr(incomplete_details, "model_dump"):
                        incomplete_details = incomplete_details.model_dump()
                    elif not isinstance(incomplete_details, (str, int, float, bool, dict, list)):
                        incomplete_details = str(incomplete_details)

                capabilities = self.get_capabilities()
                cached_input_rate = (
                    capabilities.cost_per_1k_cache_read_input_tokens
                    if capabilities.cost_per_1k_cache_read_input_tokens is not None
                    else capabilities.cost_per_1k_input_tokens * 0.1
                )
                uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
                cost_estimate = (
                    (uncached_input_tokens / 1000) * capabilities.cost_per_1k_input_tokens +
                    (cached_input_tokens / 1000) * cached_input_rate +
                    (output_tokens / 1000) * capabilities.cost_per_1k_output_tokens
                )

                tool_calls: List[ToolCall] = []
                for item in getattr(response, "output", []) or []:
                    item_type = getattr(item, "type", None)
                    if item_type != "function_call":
                        continue
                    raw_arguments = getattr(item, "arguments", None)
                    tool_calls.append(
                        ToolCall(
                            id=str(getattr(item, "call_id", None) or getattr(item, "id", "") or ""),
                            name=str(getattr(item, "name", "") or ""),
                            arguments=self._parse_tool_arguments(
                                raw_arguments,
                                str(getattr(item, "name", "") or ""),
                            ),
                            raw_arguments=raw_arguments,
                        )
                    )

                return ModelResponse(
                    content=response.output_text or "",
                    model_name=self.config.model_name,
                    provider="openai",
                    tokens_used=total_tokens,
                    cost_estimate=cost_estimate,
                    tool_calls=tool_calls,
                    metadata={
                        "response_id": getattr(response, "id", None),
                        "status": getattr(response, "status", None),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "incomplete_details": incomplete_details,
                    }
                )

            params = {
                "model": model_name,
                "messages": oa_messages,
                "timeout": self.config.timeout,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
            }
            if tools:
                params["tools"] = [self._chat_tool_schema(tool) for tool in tools]
                params["parallel_tool_calls"] = False
                choice_value = self._tool_choice_value(tool_choice)
                if choice_value:
                    params["tool_choice"] = choice_value
            self._apply_prompt_cache_params(params, include_retention=False)
            response = self.client.chat.completions.create(**params)

            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls: List[ToolCall] = []
            for call in getattr(choice.message, "tool_calls", []) or []:
                function = getattr(call, "function", None)
                raw_arguments = getattr(function, "arguments", None) if function else None
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(call, "id", "") or ""),
                        name=str(getattr(function, "name", "") if function else ""),
                        arguments=self._parse_tool_arguments(
                            raw_arguments,
                            str(getattr(function, "name", "") if function else ""),
                        ),
                        raw_arguments=raw_arguments,
                    )
                )

            capabilities = self.get_capabilities()
            cached_input_tokens = self._cached_input_tokens(response.usage)
            prompt_tokens = self._usage_int(getattr(response.usage, "prompt_tokens", 0))
            completion_tokens = self._usage_int(getattr(response.usage, "completion_tokens", 0))
            total_tokens = (
                self._usage_int(getattr(response.usage, "total_tokens", 0))
                or prompt_tokens + completion_tokens
            )
            cached_input_rate = (
                capabilities.cost_per_1k_cache_read_input_tokens
                if capabilities.cost_per_1k_cache_read_input_tokens is not None
                else capabilities.cost_per_1k_input_tokens * 0.1
            )
            uncached_prompt_tokens = max(prompt_tokens - cached_input_tokens, 0)
            cost_estimate = (
                (uncached_prompt_tokens / 1000) * capabilities.cost_per_1k_input_tokens +
                (cached_input_tokens / 1000) * cached_input_rate +
                (completion_tokens / 1000) * capabilities.cost_per_1k_output_tokens
            )

            return ModelResponse(
                content=content,
                model_name=self.config.model_name,
                provider="openai",
                tokens_used=total_tokens,
                cost_estimate=cost_estimate,
                tool_calls=tool_calls,
                metadata={
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_input_tokens": cached_input_tokens,
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_capabilities(self) -> ModelCapabilities:
        """Get OpenAI model capabilities."""
        # Keep a minimal set of current OpenAI models
        model_configs = {
            # GPT-4o (multi-modal, chat-compatible)
            "gpt-4o": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.005,   # indicative
                cost_per_1k_output_tokens=0.015,  # indicative
                reasoning_capable=True
            ),
            "gpt-4o-mini": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.00015,
                cost_per_1k_output_tokens=0.0006,
                reasoning_capable=True
            ),

            # o3 series - Advanced reasoning models (2025)
            "o3": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=100000,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.060,
                reasoning_capable=True
            ),
            "o3-2025-04-16": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=100000,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.060,
                reasoning_capable=True
            ),
            "o3-mini": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=65536,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=False,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.012,
                reasoning_capable=True
            ),
            "o3-pro": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=100000,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.030,
                cost_per_1k_output_tokens=0.120,
                reasoning_capable=True
            ),

            # gpt-5 family
            "gpt-5": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=8192,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                reasoning_capable=True
            ),
            "gpt-5.2": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                reasoning_capable=True
            ),
            "gpt-5-mini": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.002,
                cost_per_1k_output_tokens=0.008,
                reasoning_capable=True
            ),
            "gpt-5.4": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                reasoning_capable=True
            ),
            "gpt-5.3-codex": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                reasoning_capable=True
            ),
            "gpt-5.4-mini": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.002,
                cost_per_1k_output_tokens=0.008,
                reasoning_capable=True
            ),
            "gpt-5.4-pro": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.060,
                reasoning_capable=True
            ),
            "gpt-5.5": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=16384,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.060,
                reasoning_capable=True
            ),

            # o4-mini (advanced reasoning, lighter variant)
            "o4-mini-2025-04-16": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=65536,
                supports_json_mode=True,
                supports_function_calling=True,
                supports_vision=False,
                cost_per_1k_input_tokens=0.004,
                cost_per_1k_output_tokens=0.016,
                reasoning_capable=True
            ),
        }
        
        if self.config.model_name in model_configs:
            capabilities = model_configs[self.config.model_name]
            capabilities.supports_native_tools = capabilities.supports_function_calling
            capabilities.supports_parallel_tool_calls = capabilities.supports_function_calling
            return capabilities

        # Handle dated/versioned aliases within the same family.
        family_prefixes = (
            "gpt-5.5",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.3-codex",
            "gpt-5-mini",
            "gpt-5.2",
            "gpt-5",
            "gpt-4o-mini",
            "gpt-4o",
            "o3-pro",
            "o3-mini",
            "o3",
            "o4-mini",
        )
        for prefix in family_prefixes:
            if self.config.model_name.startswith(prefix) and prefix in model_configs:
                capabilities = model_configs[prefix]
                capabilities.supports_native_tools = capabilities.supports_function_calling
                capabilities.supports_parallel_tool_calls = capabilities.supports_function_calling
                return capabilities

        return ModelCapabilities(  # Default capabilities
            max_context_length=4096,
            max_output_tokens=2048,
            supports_json_mode=True,
            cost_per_1k_input_tokens=0.002,
            cost_per_1k_output_tokens=0.004
        )
