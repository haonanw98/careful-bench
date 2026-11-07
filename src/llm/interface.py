"""
🧠 Advanced Model Interface for DB-AgentSafe

Core abstraction layer providing:
- Unified API across all model providers
- Intelligent response parsing and validation
- Automatic retry logic with exponential backoff
- Response caching and deduplication
- Performance monitoring and metrics
- Safety filtering and content validation
"""

import json
import time
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from ..logger import get_logger

logger = get_logger(__name__)


class Role(Enum):
    """Represents the role of a message's author."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a conversation, with a role and content."""
    role: Role
    content: Any
    tool_call_id: Optional[str] = None # For tool responses


@dataclass(frozen=True)
class ToolSchema:
    """Provider-neutral schema for one model-callable tool."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """Provider-neutral representation of a model-requested tool call."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: Any = None


class ToolChoice(Enum):
    """Provider-neutral native tool choice policy."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


class ProviderType(Enum):
    """Supported providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    AZURE = "azure"
    GOOGLE = "google"


@dataclass
class ModelCapabilities:
    """Model capabilities and limitations."""
    max_context_length: int = 4096
    max_output_tokens: int = 1024
    supports_json_mode: bool = False
    supports_function_calling: bool = False
    supports_native_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    cost_per_1k_cache_read_input_tokens: Optional[float] = None
    cost_per_1k_cache_write_input_tokens: Optional[float] = None
    safety_filtered: bool = True
    reasoning_capable: bool = True


@dataclass
class ModelConfig:
    """Configuration for model access and behavior."""
    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    enable_safety_filter: bool = True
    max_prompt_length: int = 50000
    validate_json: bool = True
    provider_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Standardized model response."""
    content: str
    model_name: str
    provider: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    response_time: Optional[float] = None
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)


class ResponseCache:
    """Simple in-memory response cache."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> str:
        """Generate cache key from messages and config."""
        provider_settings = self._serialize_provider_settings(config.provider_settings)
        config_str = (
            f"{config.model_name}_{config.temperature}_{config.max_tokens}_"
            f"{provider_settings}"
        )
        message_content = "".join([
            f"{m.role.value}:{self._serialize_content(m.content)}"
            for m in messages
        ])
        tools_content = self._serialize_content(tools or [])
        choice_content = tool_choice.value if isinstance(tool_choice, ToolChoice) else str(tool_choice or "")
        content = f"{message_content}_{config_str}_{tools_content}_{choice_content}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _serialize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, ToolSchema):
            return json.dumps(
                {
                    "name": content.name,
                    "description": content.description,
                    "input_schema": content.input_schema,
                },
                sort_keys=True,
                default=str,
            )
        if isinstance(content, ToolCall):
            return json.dumps(
                {
                    "id": content.id,
                    "name": content.name,
                    "arguments": content.arguments,
                    "raw_arguments": content.raw_arguments,
                },
                sort_keys=True,
                default=str,
            )
        if isinstance(content, list):
            return json.dumps(
                [json.loads(ResponseCache._serialize_content(item)) if not isinstance(item, str) else item for item in content],
                sort_keys=True,
                default=str,
            )
        try:
            return json.dumps(content, sort_keys=True, default=str)
        except Exception:
            return str(content)

    @staticmethod
    def _serialize_provider_settings(settings: Optional[Dict[str, Any]]) -> str:
        if not settings:
            return "{}"

        sanitized: Dict[str, Any] = {}
        sensitive_markers = ("secret", "token", "api_key", "access_key", "password", "prompt_cache_key")
        for key, value in sorted(settings.items()):
            key_l = str(key).lower()
            if any(marker in key_l for marker in sensitive_markers):
                digest = hashlib.sha256(str(value).encode()).hexdigest()[:12]
                sanitized[key] = f"<sha256:{digest}>"
            else:
                sanitized[key] = value
        return ResponseCache._serialize_content(sanitized)
    
    def get(
        self,
        messages: List[Message],
        config: ModelConfig,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Optional[ModelResponse]:
        """Get cached response if available and not expired."""
        key = self._generate_key(messages, config, tools, tool_choice)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < config.cache_ttl:
                self.access_times[key] = time.time()
                response.cached = True
                return response
            else:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(
        self,
        messages: List[Message],
        config: ModelConfig,
        response: ModelResponse,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ):
        """Cache a response."""
        if not config.enable_caching:
            return
        
        key = self._generate_key(messages, config, tools, tool_choice)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = (response, time.time())
        self.access_times[key] = time.time()


class ModelInterface(ABC):
    """Abstract base class for all model providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.cache = ResponseCache()
        self.call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.provider_cache_read_input_tokens = 0
        self.provider_cache_write_input_tokens = 0
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize provider-specific client."""
        pass
    
    @abstractmethod
    def _raw_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> ModelResponse:
        """Generate response using provider API."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        pass
    
    def generate(
        self,
        messages: List[Message],
        expect_json: bool = False,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> ModelResponse:
        """
        Generate response with caching, retries, and safety checks.
        
        Args:
            messages: A list of Message objects representing the conversation history.
            expect_json: Whether to expect a JSON response.
            
        Returns:
            ModelResponse with generated content.
        """
        # Validate input
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if tools and not self.get_capabilities().supports_native_tools:
            raise ValueError(f"Model/provider {self.config.model_name} does not support native tools")
        
        # Check cache first
        if self.config.enable_caching:
            cached_response = self.cache.get(messages, self.config, tools, tool_choice)
            if cached_response:
                logger.debug(f"Cache hit for {self.config.model_name}")
                return cached_response
        
        # Generate with retries
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                response = self._raw_generate(messages, tools=tools, tool_choice=tool_choice)
                response.response_time = time.time() - start_time
                
                # Post-process response
                if expect_json:
                    response = self._extract_json_response(response)
                
                # Update metrics
                self.call_count += 1
                self.total_tokens += response.tokens_used or 0
                self.total_cost += response.cost_estimate or 0.0
                cache_read = self._metadata_int(
                    response.metadata,
                    "cache_read_input_tokens",
                    "cached_input_tokens",
                )
                cache_write = self._metadata_int(
                    response.metadata,
                    "cache_write_input_tokens",
                    "cache_creation_input_tokens",
                )
                self.provider_cache_read_input_tokens += cache_read
                self.provider_cache_write_input_tokens += cache_write
                input_tokens = self._metadata_int(
                    response.metadata,
                    "input_tokens",
                    "prompt_tokens",
                )
                output_tokens = self._metadata_int(
                    response.metadata,
                    "output_tokens",
                    "completion_tokens",
                )
                
                # Cache successful response
                if self.config.enable_caching and not response.cached:
                    self.cache.set(messages, self.config, response, tools, tool_choice)
                
                cache_note = ""
                if input_tokens or output_tokens or cache_read or cache_write:
                    cache_ratio = (cache_read / input_tokens) if input_tokens else 0.0
                    cache_note = (
                        f", input={input_tokens}, output={output_tokens}, "
                        f"cache_read={cache_read}, cache_write={cache_write}, "
                        f"prompt_prefix_cache_ratio={cache_ratio:.4f}"
                    )
                logger.info(
                    f"Generated response: {len(response.content or '')} chars, "
                    f"{response.tokens_used} tokens{cache_note}, {response.response_time:.2f}s"
                )
                
                return response
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed: {e}")
        
        raise Exception(f"Model generation failed after {self.config.max_retries + 1} attempts: {last_exception}")

    @staticmethod
    def _metadata_int(metadata: Optional[Dict[str, Any]], *keys: str) -> int:
        if not metadata:
            return 0
        for key in keys:
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    @staticmethod
    def _is_repaired_state_changing_payload(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False

        tool_name = str(parsed.get("tool_name") or "").strip()
        if tool_name == "write_db":
            return True

        func_call = parsed.get("function_call")
        if isinstance(func_call, dict):
            func_name = str(func_call.get("function") or func_call.get("name") or "").strip()
            if func_name == "write_db":
                return True

        decision = str(parsed.get("decision") or "").strip().lower()
        return decision == "execute"

    @staticmethod
    def _find_json_key_outside_string(candidate: str, key: str) -> int:
        needle = f'"{key}"'
        in_string = False
        escape = False
        idx = 0
        while idx < len(candidate):
            ch = candidate[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                idx += 1
                continue

            if ch == '"':
                if candidate.startswith(needle, idx):
                    after = idx + len(needle)
                    while after < len(candidate) and candidate[after].isspace():
                        after += 1
                    if after < len(candidate) and candidate[after] == ":":
                        return idx
                in_string = True
            idx += 1
        return -1

    def _extract_json_response(self, response: ModelResponse) -> ModelResponse:
        """Extract and validate JSON from response."""
        content = response.content
        
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
        else:
            json_start = content.find('{')
            if json_start == -1:
                logger.warning("No JSON found in response")
                response.metadata['json_extraction_failed'] = True
                return response
            # Use the full suffix; we'll parse via raw_decode to tolerate trailing text.
            json_str = content[json_start:].strip()

        # If provider metadata leaks into the text (e.g., guardrails/stop_reason),
        # strip it out before attempting JSON repair.
        stop_reason_idx = self._find_json_key_outside_string(json_str, "stop_reason")
        if stop_reason_idx != -1:
            json_str = json_str[:stop_reason_idx].rstrip()

        def _repair_truncated_json(candidate: str) -> Optional[str]:
            """Best-effort repair for truncated JSON objects (missing closing delimiters)."""
            if not candidate:
                return None

            stack: List[str] = []
            in_string = False
            escape = False

            for ch in candidate:
                if in_string:
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch in "{[":
                    stack.append(ch)
                    continue
                if ch in "}]":
                    if not stack:
                        return None
                    opener = stack.pop()
                    if opener == "{" and ch != "}":
                        return None
                    if opener == "[" and ch != "]":
                        return None

            fixed = candidate.rstrip()

            # If we ended mid-string due to truncation, close the quote.
            if in_string:
                fixed += '"'

            fixed = fixed.rstrip()
            while fixed and fixed[-1] in {",", ":"}:
                fixed = fixed[:-1].rstrip()

            for opener in reversed(stack):
                fixed += "}" if opener == "{" else "]"
            return fixed

        try:
            # First attempt: parse a single JSON object from the start (tolerates trailing text).
            decoder = json.JSONDecoder()
            parsed_json, end_idx = decoder.raw_decode(json_str)
            if end_idx and end_idx < len(json_str):
                response.metadata["json_trailing_text"] = json_str[end_idx:end_idx + 200]
            response.metadata['parsed_json'] = parsed_json
            response.metadata['json_valid'] = True
            logger.debug(f"Successfully extracted JSON: {json_str[:100]}...")
        except json.JSONDecodeError as e:
            # Second attempt: try to fix common issues
            try:
                # Fix unescaped newlines/tabs within strings
                # This regex finds strings and escapes control chars within them
                def escape_string_contents(match):
                    s = match.group(0)
                    # Keep the quotes, escape the contents
                    return s[0] + s[1:-1].replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + s[-1]
                
                # Apply to all quoted strings
                fixed_json = re.sub(r'"[^"]*"', escape_string_contents, json_str)
                decoder = json.JSONDecoder()
                parsed_json, _ = decoder.raw_decode(fixed_json)
                if self._is_repaired_state_changing_payload(parsed_json):
                    raise ValueError("Refusing to fix malformed state-changing action JSON")
                
                response.metadata['parsed_json'] = parsed_json
                response.metadata['json_valid'] = True
                response.metadata['json_fixed'] = True
                logger.debug(f"Fixed and extracted JSON successfully")
            except Exception as e2:
                try:
                    repaired = _repair_truncated_json(json_str)
                    if repaired:
                        parsed_json = json.loads(repaired)
                        if self._is_repaired_state_changing_payload(parsed_json):
                            raise ValueError("Refusing to repair truncated state-changing action JSON")
                        response.metadata["parsed_json"] = parsed_json
                        response.metadata["json_valid"] = True
                        response.metadata["json_repaired"] = True
                        logger.debug("Repaired and extracted JSON successfully")
                    else:
                        raise ValueError("Could not repair JSON")
                except Exception as e3:
                    logger.warning(
                        f"Invalid JSON extracted: {e} (fix attempt also failed: {e2}; repair failed: {e3})"
                    )
                    response.metadata['json_extraction_failed'] = True
                    response.metadata['invalid_json'] = json_str[:500]
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'cache_size': len(self.cache.cache),
            'provider_cache_read_input_tokens': self.provider_cache_read_input_tokens,
            'provider_cache_write_input_tokens': self.provider_cache_write_input_tokens,
            'model_name': self.config.model_name,
            'provider': self.config.provider.value
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self.cache.cache.clear()
        self.cache.access_times.clear()
        logger.info("Model response cache cleared")
    
    def estimate_cost(self, messages: List[Message], response_length: int = None) -> float:
        """Estimate cost for a conversation."""
        capabilities = self.get_capabilities()
        
        # Rough token estimation
        input_tokens = sum(len(ResponseCache._serialize_content(m.content)) for m in messages) // 4
        output_tokens = (response_length or self.config.max_tokens) // 4
        
        cost = (
            (input_tokens / 1000) * capabilities.cost_per_1k_input_tokens +
            (output_tokens / 1000) * capabilities.cost_per_1k_output_tokens
        )
        
        return cost
