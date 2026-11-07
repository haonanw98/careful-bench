"""
AWS Bedrock Provider

Supports Claude, DeepSeek, Qwen, Llama models via AWS Bedrock.
Use us. prefix for cross-region inference.

Supported Models:
- anthropic.claude-opus-4-5-20251101-v1:0
- anthropic.claude-sonnet-4-5-20250929-v1:0
- anthropic.claude-haiku-4-5-20251001-v1:0
- anthropic.claude-opus-4-1-20250805-v1:0
- anthropic.claude-opus-4-20250514-v1:0
- anthropic.claude-sonnet-4-20250514-v1:0
- anthropic.claude-3-7-sonnet-20250219-v1:0
- anthropic.claude-3-5-sonnet-20241022-v2:0
- anthropic.claude-3-5-haiku-20241022-v1:0
- anthropic.claude-3-haiku-20240307-v1:0
- deepseek.r1-v1:0
- deepseek.v3-v1:0
- qwen.qwen3-coder-480b-a35b-v1:0
- qwen.qwen3-235b-a22b-v1:0
- qwen.qwen3-32b-v1:0
- meta.llama3-8b-instruct-v1:0
- meta.llama3-70b-instruct-v1:0
"""

import json
import os
from typing import Any, Dict, List, Optional
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


class BedrockProvider(ModelInterface):
    """Provider for AWS Bedrock models."""

    @staticmethod
    def _claude_sampling_controls_deprecated(model_name: str) -> bool:
        """Latest Bedrock Claude Opus/Sonnet variants reject sampling controls."""
        return any(
            marker in model_name
            for marker in (
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
            )
        )

    @staticmethod
    def _strip_sampling_controls(body: dict) -> dict:
        """Remove sampling controls for Bedrock Claude compatibility fallback."""
        return {
            key: value
            for key, value in body.items()
            if key not in {"temperature", "top_p", "top_k"}
        }

    @staticmethod
    def _is_sampling_compatibility_error(error_message: str) -> bool:
        lowered = (error_message or "").lower()
        normalized = lowered.replace("`", "").replace('"', "").replace("'", "")
        return any(
            marker in normalized
            for marker in (
                "temperature is deprecated",
                "top_p is deprecated",
                "top_k is deprecated",
                "temperature is not supported",
                "top_p is not supported",
                "top_k is not supported",
                "temperature must not be provided",
                "top_p must not be provided",
                "top_k must not be provided",
            )
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text is not None:
                        parts.append(str(text))
                    elif "cache_control" not in block:
                        parts.append(json.dumps(block, default=str))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        try:
            return json.dumps(content, default=str)
        except Exception:
            return str(content)

    @staticmethod
    def _openai_like_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [
            {
                "role": msg.get("role", "user"),
                "content": BedrockProvider._content_to_text(msg.get("content", "")),
            }
            for msg in messages
        ]

    @staticmethod
    def _usage_int(usage: Dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    @staticmethod
    def _claude_tool_schema(tool: ToolSchema) -> Dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema or {"type": "object", "properties": {}},
        }

    @staticmethod
    def _claude_tool_choice(tool_choice: Optional[ToolChoice]) -> Optional[Dict[str, Any]]:
        if tool_choice is None or tool_choice == ToolChoice.AUTO:
            return {"type": "auto", "disable_parallel_tool_use": True}
        if tool_choice == ToolChoice.REQUIRED:
            return {"type": "any", "disable_parallel_tool_use": True}
        if tool_choice == ToolChoice.NONE:
            return {"type": "none"}
        return {"type": "auto", "disable_parallel_tool_use": True}

    @staticmethod
    def _converse_tool_schema(tool: ToolSchema) -> Dict[str, Any]:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": tool.input_schema or {"type": "object", "properties": {}},
                },
            },
        }

    @staticmethod
    def _converse_tool_choice(tool_choice: Optional[ToolChoice]) -> Optional[Dict[str, Any]]:
        if tool_choice is None or tool_choice == ToolChoice.AUTO:
            return {"auto": {}}
        if tool_choice == ToolChoice.REQUIRED:
            return {"any": {}}
        if tool_choice == ToolChoice.NONE:
            return None
        return {"auto": {}}

    @staticmethod
    def _claude_supports_native_tools(model_name: str) -> bool:
        normalized = (model_name or "").lower()
        if "anthropic.claude" not in normalized:
            return False
        return any(
            marker in normalized
            for marker in (
                "claude-3",
                "claude-opus",
                "claude-sonnet",
                "claude-haiku",
            )
        )

    @staticmethod
    def _converse_supports_native_tools(model_name: str) -> bool:
        """Models whose Bedrock Runtime Converse API returns toolUse blocks."""
        base_model_id = BedrockProvider._base_model_id(model_name)
        return base_model_id.startswith("openai.gpt-oss")

    @staticmethod
    def _base_model_id(model_name: str) -> str:
        normalized = (model_name or "").strip().lower()
        for prefix in ("us.", "eu.", "jp.", "global."):
            if normalized.startswith(prefix):
                return normalized[len(prefix):]
        return normalized

    @staticmethod
    def _is_openai_like_model(model_name: str) -> bool:
        base_model_id = BedrockProvider._base_model_id(model_name)
        return (
            base_model_id.startswith("deepseek.")
            or base_model_id.startswith("moonshot.")
            or base_model_id.startswith("moonshotai.")
            or base_model_id.startswith("minimax.")
            or base_model_id.startswith("zai.")
            or base_model_id.startswith("openai.")
            or base_model_id.startswith("gpt-oss")
        )

    @staticmethod
    def _is_qwen_model(model_name: str) -> bool:
        return BedrockProvider._base_model_id(model_name).startswith("qwen.")

    @staticmethod
    def _uses_one_hour_prompt_cache(value: Any) -> bool:
        if isinstance(value, dict):
            cache_control = value.get("cache_control")
            if isinstance(cache_control, dict) and cache_control.get("ttl") == "1h":
                return True
            return any(BedrockProvider._uses_one_hour_prompt_cache(item) for item in value.values())
        if isinstance(value, list):
            return any(BedrockProvider._uses_one_hour_prompt_cache(item) for item in value)
        return False
    
    def _initialize(self):
        """Initialize Bedrock client with proper AWS configuration."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            from botocore.exceptions import ClientError, NoCredentialsError
            
            # Get AWS credentials from multiple sources
            aws_config = {}
            
            # Priority 1: Direct config parameters
            if self.config.provider_settings.get("aws_region"):
                aws_config["region_name"] = self.config.provider_settings["aws_region"]
            elif self.config.provider_settings.get("AWS_REGION"):
                aws_config["region_name"] = self.config.provider_settings["AWS_REGION"]
            elif os.getenv("AWS_REGION"):
                aws_config["region_name"] = os.getenv("AWS_REGION")
            else:
                aws_config["region_name"] = "us-east-1"  # Default region
            
            profile_name = (
                self.config.provider_settings.get("aws_profile")
                or self.config.provider_settings.get("AWS_PROFILE")
                or os.getenv("AWS_PROFILE")
            )

            # A named profile is an explicit credential choice. Do not let stale
            # environment/provider_settings access keys override it.
            if not profile_name:
                if self.config.provider_settings.get("aws_access_key_id"):
                    aws_config["aws_access_key_id"] = self.config.provider_settings["aws_access_key_id"]
                elif self.config.provider_settings.get("AWS_ACCESS_KEY_ID"):
                    aws_config["aws_access_key_id"] = self.config.provider_settings["AWS_ACCESS_KEY_ID"]
                elif os.getenv("AWS_ACCESS_KEY_ID"):
                    aws_config["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")

                if self.config.provider_settings.get("aws_secret_access_key"):
                    aws_config["aws_secret_access_key"] = self.config.provider_settings["aws_secret_access_key"]
                elif self.config.provider_settings.get("AWS_SECRET_ACCESS_KEY"):
                    aws_config["aws_secret_access_key"] = self.config.provider_settings["AWS_SECRET_ACCESS_KEY"]
                elif os.getenv("AWS_SECRET_ACCESS_KEY"):
                    aws_config["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

                if self.config.provider_settings.get("aws_session_token"):
                    aws_config["aws_session_token"] = self.config.provider_settings["aws_session_token"]
                elif self.config.provider_settings.get("AWS_SESSION_TOKEN"):
                    aws_config["aws_session_token"] = self.config.provider_settings["AWS_SESSION_TOKEN"]
                elif os.getenv("AWS_SESSION_TOKEN"):
                    aws_config["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")

            def _int_setting(key: str, default: int) -> int:
                raw = self.config.provider_settings.get(key)
                if raw is None:
                    raw = os.getenv(key.upper())
                if raw is None:
                    return default
                try:
                    return int(raw)
                except Exception:
                    return default

            # Bedrock client timeouts: prevent hung invoke_model calls from stalling long eval batches.
            read_timeout_s = _int_setting("bedrock_read_timeout_s", int(self.config.timeout or 120))
            connect_timeout_s = _int_setting("bedrock_connect_timeout_s", min(20, read_timeout_s))
            aws_retry_attempts = _int_setting("bedrock_retry_attempts", 3)

            aws_config["config"] = BotoConfig(
                connect_timeout=connect_timeout_s,
                read_timeout=read_timeout_s,
                retries={"max_attempts": aws_retry_attempts, "mode": "standard"},
            )
            
            if profile_name:
                session = boto3.Session(
                    profile_name=profile_name,
                    region_name=aws_config.get("region_name"),
                )
                self.bedrock = session.client(
                    "bedrock-runtime",
                    config=aws_config["config"],
                )
                auth_mode = f"profile:{profile_name}"
            else:
                self.bedrock = boto3.client("bedrock-runtime", **aws_config)
                if aws_config.get("aws_access_key_id"):
                    auth_mode = "explicit-env-or-settings"
                else:
                    auth_mode = "default-sdk-chain"
            self.ClientError = ClientError
            self.NoCredentialsError = NoCredentialsError
            
            # Store config for debugging
            self.aws_region = aws_config.get("region_name", "us-east-1")
            
            logger.info(
                "Initialized Bedrock client for %s in region %s using %s (connect_timeout=%ss read_timeout=%ss retries=%s)",
                self.config.model_name,
                self.aws_region,
                auth_mode,
                connect_timeout_s,
                read_timeout_s,
                aws_retry_attempts,
            )
            
        except ImportError:
            raise ImportError(
                "boto3 required for Bedrock provider. "
                "Install with: pip install boto3"
            )
        except NoCredentialsError:
            raise ValueError(
                "AWS credentials not found. Set AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                "or configure the default AWS credential chain."
            )

    def _raw_generate_converse(
        self,
        messages: List[Message],
        tools: List[ToolSchema],
        tool_choice: Optional[ToolChoice] = None,
    ) -> ModelResponse:
        """Generate with Bedrock Converse when a non-Claude model supports toolUse blocks."""
        converse_messages: List[Dict[str, Any]] = []
        system: List[Dict[str, str]] = []

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role).lower()
            content = self._content_to_text(msg.content if hasattr(msg, "content") else str(msg))
            if role == "system":
                if content:
                    system.append({"text": content})
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            converse_messages.append({"role": role, "content": [{"text": content}]})

        if not converse_messages:
            converse_messages.append({"role": "user", "content": [{"text": ""}]})

        inference_config: Dict[str, Any] = {"maxTokens": self.config.max_tokens}
        if self.config.temperature is not None:
            inference_config["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            inference_config["topP"] = self.config.top_p

        tool_config: Dict[str, Any] = {
            "tools": [self._converse_tool_schema(tool) for tool in tools],
        }
        converse_tool_choice = self._converse_tool_choice(tool_choice)
        if converse_tool_choice:
            tool_config["toolChoice"] = converse_tool_choice

        params: Dict[str, Any] = {
            "modelId": self.config.model_name,
            "messages": converse_messages,
            "inferenceConfig": inference_config,
            "toolConfig": tool_config,
        }
        if system:
            params["system"] = system

        try:
            response_body = self.bedrock.converse(**params)
        except self.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            should_retry_without_sampling = (
                error_code == "ValidationException"
                and any(key in inference_config for key in ("temperature", "topP"))
                and self._is_sampling_compatibility_error(error_message)
            )
            if not should_retry_without_sampling:
                raise
            retry_params = dict(params)
            retry_params["inferenceConfig"] = {"maxTokens": self.config.max_tokens}
            logger.warning(
                "Retrying Bedrock Converse request for %s without sampling controls after validation error: %s",
                self.config.model_name,
                error_message,
            )
            response_body = self.bedrock.converse(**retry_params)

        output_message = response_body.get("output", {}).get("message", {})
        blocks = output_message.get("content", [])
        content_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    content_parts.append(str(block.get("text") or ""))
                elif "reasoningContent" in block:
                    reasoning = block.get("reasoningContent") or {}
                    reasoning_text = reasoning.get("reasoningText") if isinstance(reasoning, dict) else None
                    if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                        content_parts.append(str(reasoning_text["text"]))
                elif "toolUse" in block:
                    raw_tool = block.get("toolUse") or {}
                    raw_input = raw_tool.get("input")
                    tool_calls.append(
                        ToolCall(
                            id=str(raw_tool.get("toolUseId", "") or ""),
                            name=str(raw_tool.get("name", "") or ""),
                            arguments=raw_input if isinstance(raw_input, dict) else {},
                            raw_arguments=raw_input,
                        )
                    )

        usage = response_body.get("usage", {})
        input_tokens = self._usage_int(usage, "inputTokens", "input_tokens")
        output_tokens = self._usage_int(usage, "outputTokens", "output_tokens")
        total_tokens = self._usage_int(usage, "totalTokens", "total_tokens") or input_tokens + output_tokens

        capabilities = self.get_capabilities()
        cost_estimate = (
            (input_tokens / 1000) * capabilities.cost_per_1k_input_tokens
            + (output_tokens / 1000) * capabilities.cost_per_1k_output_tokens
        )

        return ModelResponse(
            content="\n".join(part for part in content_parts if part),
            model_name=self.config.model_name,
            provider="bedrock",
            tokens_used=total_tokens,
            cost_estimate=cost_estimate,
            tool_calls=tool_calls,
            metadata={
                "stop_reason": response_body.get("stopReason"),
                "usage": usage,
                "metrics": response_body.get("metrics", {}),
                "aws_region": self.aws_region,
                "model_id": self.config.model_name,
                "api": "converse",
            },
        )
    
    def _raw_generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> ModelResponse:
        """Generate response using Bedrock API with correct format."""
        try:
            if tools and self._converse_supports_native_tools(self.config.model_name):
                return self._raw_generate_converse(messages, tools, tool_choice)

            if tools and not self._claude_supports_native_tools(self.config.model_name):
                raise ValueError("Native tools are only implemented for Bedrock Claude models")

            # Convert Message objects to dict format and separate system messages
            formatted_messages = []
            system_prompt = None
            
            for msg in messages:
                # Handle different role formats
                if hasattr(msg, 'role'):
                    role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role).lower()
                else:
                    role = 'user'
                
                # Preserve Claude content blocks for provider-native prompt caching.
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # Anthropic API: system messages go in a separate parameter
                if role == 'system':
                    system_prompt = self._content_to_text(content)
                else:
                    # Only user and assistant roles go in messages array
                    if role not in ['user', 'assistant']:
                        role = 'user'  # Default to user for unknown roles
                    formatted_messages.append({
                        "role": role,
                        "content": content
                    })
            
            # For single message, extract the prompt for backwards compatibility
            if len(messages) == 1:
                prompt = self._content_to_text(messages[0].content)
            else:
                # Build conversation prompt for multiple messages
                prompt = "\n".join([
                    f"{msg.role}: {self._content_to_text(msg.content)}"
                    for msg in messages
                ])
            
            # Prepare request body based on model type
            if "anthropic.claude" in self.config.model_name:
                # Anthropic Claude format for Bedrock
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.config.max_tokens,
                    "messages": formatted_messages
                }

                if not self._claude_sampling_controls_deprecated(self.config.model_name):
                    # Older Bedrock Claude variants reject requests that specify both temperature
                    # and top_p. Prefer temperature when both are available.
                    if self.config.temperature is not None:
                        body["temperature"] = self.config.temperature
                    elif self.config.top_p is not None:
                        body["top_p"] = self.config.top_p

                # Add system prompt if present
                if system_prompt:
                    body["system"] = system_prompt

                if tools:
                    body["tools"] = [self._claude_tool_schema(tool) for tool in tools]
                    claude_tool_choice = self._claude_tool_choice(tool_choice)
                    if claude_tool_choice:
                        body["tool_choice"] = claude_tool_choice

                # Add top_k if specified
                if self.config.top_k and not self._claude_sampling_controls_deprecated(self.config.model_name):
                    body["top_k"] = self.config.top_k
                
            elif "llama4" in self.config.model_name:
                # Meta Llama 4 format for Bedrock (requires prompt with chat template)
                # Build Llama 4 chat template: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
                llama4_prompt = "<|begin_of_text|>"
                if system_prompt:
                    llama4_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                for msg in formatted_messages:
                    role = msg.get("role", "user")
                    content = self._content_to_text(msg.get("content", ""))
                    llama4_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                # Add assistant header to prompt response
                llama4_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

                body = {
                    "prompt": llama4_prompt,
                    "max_gen_len": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p
                }
                if self.config.top_k:
                    body["top_k"] = self.config.top_k

            elif "meta.llama" in self.config.model_name:
                # Meta Llama 3.x format for Bedrock (legacy prompt format)
                body = {
                    "prompt": prompt,
                    "max_gen_len": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p
                }

                if self.config.top_k:
                    body["top_k"] = self.config.top_k
                    
            elif "amazon.titan" in self.config.model_name:
                # Amazon Titan format
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "topP": self.config.top_p,
                        "stopSequences": []
                    }
                }
            elif self._is_openai_like_model(self.config.model_name):
                # DeepSeek/OpenAI GPT-OSS format (OpenAI-like messages)
                openai_messages = self._openai_like_messages(formatted_messages)
                body = {
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "messages": openai_messages
                }
                if system_prompt:
                    body["messages"].insert(0, {"role": "system", "content": system_prompt})
            elif self._is_qwen_model(self.config.model_name):
                # Qwen format (similar to OpenAI)
                qwen_messages = self._openai_like_messages(formatted_messages)
                body = {
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "messages": qwen_messages
                }
                if system_prompt:
                    body["messages"].insert(0, {"role": "system", "content": system_prompt})
            else:
                # Generic format for other models
                body = {
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p
                }
            
            logger.debug(f"Bedrock request body: {json.dumps(body, indent=2)}")
            
            # Make API call
            cache_read_input_tokens = 0
            cache_write_input_tokens = 0
            tool_calls: List[ToolCall] = []

            try:
                response = self.bedrock.invoke_model(
                    body=json.dumps(body),
                    modelId=self.config.model_name,
                    accept="application/json",
                    contentType="application/json"
                )
            except self.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                should_retry_without_sampling = (
                    "anthropic.claude" in self.config.model_name
                    and error_code == "ValidationException"
                    and any(key in body for key in ("temperature", "top_p", "top_k"))
                    and self._is_sampling_compatibility_error(error_message)
                )
                if not should_retry_without_sampling:
                    raise

                retry_body = self._strip_sampling_controls(body)
                logger.warning(
                    "Retrying Bedrock Claude request for %s without sampling controls after validation error: %s",
                    self.config.model_name,
                    error_message,
                )
                response = self.bedrock.invoke_model(
                    body=json.dumps(retry_body),
                    modelId=self.config.model_name,
                    accept="application/json",
                    contentType="application/json"
                )
            
            # Parse response
            raw_body = response.get('body')
            response_text = raw_body.read() if hasattr(raw_body, 'read') else raw_body
            response_body = json.loads(response_text)
            logger.debug(f"Bedrock response: {json.dumps(response_body, indent=2)}")
            
            # Extract content based on model type
            if "anthropic.claude" in self.config.model_name:
                # Claude (Anthropic) via Bedrock: content is a list of content blocks
                content_parts: List[str] = []
                content = ""
                tokens_used = 0
                usage = response_body.get('usage', {})
                try:
                    blocks = response_body.get('content', [])
                    if isinstance(blocks, list):
                        for block in blocks:
                            if not isinstance(block, dict):
                                continue
                            block_type = block.get("type")
                            if block_type == "text":
                                text = block.get("text") or block.get("content") or ""
                                if text:
                                    content_parts.append(str(text))
                            elif block_type == "tool_use":
                                raw_input = block.get("input")
                                tool_calls.append(
                                    ToolCall(
                                        id=str(block.get("id", "") or ""),
                                        name=str(block.get("name", "") or ""),
                                        arguments=raw_input if isinstance(raw_input, dict) else {},
                                        raw_arguments=raw_input,
                                    )
                                )
                    content = "\n".join(content_parts)
                    cache_read_input_tokens = self._usage_int(
                        usage,
                        "cache_read_input_tokens",
                        "CacheReadInputTokens",
                        "cached_input_tokens",
                    )
                    cache_write_input_tokens = self._usage_int(
                        usage,
                        "cache_creation_input_tokens",
                        "cache_write_input_tokens",
                        "CacheWriteInputTokens",
                    )
                    tokens_used = (
                        usage.get('input_tokens', 0)
                        + usage.get('output_tokens', 0)
                        + cache_read_input_tokens
                        + cache_write_input_tokens
                    )
                except Exception:
                    # Fallbacks for unexpected shapes
                    content = response_body.get('message', "") or response_body.get('completion', "") or json.dumps(response_body)
                
            elif "llama4" in self.config.model_name or "meta.llama" in self.config.model_name:
                # Llama 3.x response format (legacy)
                content = response_body.get('generation', '')
                tokens_used = response_body.get('prompt_token_count', 0) + response_body.get('generation_token_count', 0)
                
            elif "amazon.titan" in self.config.model_name:
                # Titan response format
                results = response_body.get('results', [])
                content = results[0].get('outputText', '') if results else ''
                tokens_used = response_body.get('inputTextTokenCount', 0) + response_body.get('outputTextTokenCount', 0)

            elif self._is_openai_like_model(self.config.model_name) or self._is_qwen_model(self.config.model_name):
                # DeepSeek/Qwen/OpenAI GPT-OSS response format (OpenAI-like)
                choices = response_body.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                else:
                    content = response_body.get('content', '') or response_body.get('text', '')
                usage = response_body.get('usage', {})
                tokens_used = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)

            else:
                # Generic extraction with robust fallbacks
                content = (
                    response_body.get('completion') or
                    response_body.get('text') or
                    response_body.get('message') or
                    response_body.get('outputText') or
                    str(response_body)
                )
                tokens_used = response_body.get('token_count', 0) or (
                    response_body.get('usage', {}).get('total_tokens', 0)
                )
            
            # Calculate cost estimate
            capabilities = self.get_capabilities()
            if "anthropic.claude" in self.config.model_name and 'usage' in response_body:
                usage = response_body['usage']
                cache_read_input_tokens = self._usage_int(
                    usage,
                    "cache_read_input_tokens",
                    "CacheReadInputTokens",
                    "cached_input_tokens",
                )
                cache_write_input_tokens = self._usage_int(
                    usage,
                    "cache_creation_input_tokens",
                    "cache_write_input_tokens",
                    "CacheWriteInputTokens",
                )
                cache_read_rate = (
                    capabilities.cost_per_1k_cache_read_input_tokens
                    if capabilities.cost_per_1k_cache_read_input_tokens is not None
                    else capabilities.cost_per_1k_input_tokens * 0.1
                )
                cache_write_rate = (
                    capabilities.cost_per_1k_cache_write_input_tokens
                    if capabilities.cost_per_1k_cache_write_input_tokens is not None
                    else capabilities.cost_per_1k_input_tokens * (
                        2.0 if self._uses_one_hour_prompt_cache(body) else 1.25
                    )
                )
                cost_estimate = (
                    (usage.get('input_tokens', 0) / 1000) * capabilities.cost_per_1k_input_tokens +
                    (cache_read_input_tokens / 1000) * cache_read_rate +
                    (cache_write_input_tokens / 1000) * cache_write_rate +
                    (usage.get('output_tokens', 0) / 1000) * capabilities.cost_per_1k_output_tokens
                )
            else:
                # Estimate for other models
                cost_estimate = (tokens_used / 1000) * (
                    capabilities.cost_per_1k_input_tokens + capabilities.cost_per_1k_output_tokens
                ) / 2
            
            return ModelResponse(
                content=content,
                model_name=self.config.model_name,
                provider="bedrock",
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                tool_calls=tool_calls,
                metadata={
                    "stop_reason": response_body.get("stop_reason"),
                    "stop_sequence": response_body.get("stop_sequence"),
                    "usage": response_body.get("usage", {}),
                    "content_block_count": (
                        len(response_body.get("content", []))
                        if isinstance(response_body.get("content"), list)
                        else None
                    ),
                    "aws_region": self.aws_region,
                    "model_id": self.config.model_name,
                    "cache_read_input_tokens": cache_read_input_tokens if "anthropic.claude" in self.config.model_name else 0,
                    "cache_write_input_tokens": cache_write_input_tokens if "anthropic.claude" in self.config.model_name else 0,
                }
            )
            
        except self.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Bedrock API error [{error_code}]: {error_message}")
            
            if error_code == 'AccessDeniedException':
                raise ValueError(f"AWS access denied. Check your credentials and model access permissions for {self.config.model_name}")
            elif error_code == 'ValidationException':
                raise ValueError(f"Invalid request format for model {self.config.model_name}: {error_message}")
            else:
                raise Exception(f"Bedrock API error [{error_code}]: {error_message}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Bedrock response as JSON: {e}")
            raise Exception("Invalid JSON response from Bedrock")
            
        except Exception as e:
            logger.error(f"Bedrock generation error: {e}")
            raise
    
    def get_capabilities(self) -> ModelCapabilities:
        """Get Bedrock model capabilities."""
        model_configs = {
            # Claude 4.7 / 4.6 models
            "us.anthropic.claude-opus-4-7": ModelCapabilities(
                max_context_length=1000000,
                max_output_tokens=128000,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.025,
            ),
            "us.anthropic.claude-opus-4-6-v1": ModelCapabilities(
                max_context_length=1000000,
                max_output_tokens=128000,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.025,
            ),
            "us.anthropic.claude-sonnet-4-6": ModelCapabilities(
                max_context_length=1000000,
                max_output_tokens=64000,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
            ),

            # Claude 4.5 models
            "us.anthropic.claude-opus-4-5-20251101-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=32000,
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.025,
            ),
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=64000,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
            ),
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=64000,
                cost_per_1k_input_tokens=0.00025,
                cost_per_1k_output_tokens=0.00125,
            ),
            # Claude 4.1 models
            "us.anthropic.claude-opus-4-1-20250805-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=32000,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075,
            ),
            # Claude 4 models
            "us.anthropic.claude-opus-4-20250514-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=32000,
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075,
            ),
            "us.anthropic.claude-sonnet-4-20250514-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=64000,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
            ),
            # Claude 3.7 Sonnet
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=64000,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
            ),
            # Claude 3.5 models
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
            ),
            "us.anthropic.claude-3-5-haiku-20241022-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.00025,
                cost_per_1k_output_tokens=0.00125,
            ),
            # Claude 3 models
            "us.anthropic.claude-3-haiku-20240307-v1:0": ModelCapabilities(
                max_context_length=200000,
                max_output_tokens=4096,
                cost_per_1k_input_tokens=0.00025,
                cost_per_1k_output_tokens=0.00125,
            ),
            # DeepSeek models
            "us.deepseek.r1-v1:0": ModelCapabilities(
                max_context_length=64000,
                max_output_tokens=8192,
                # indicative
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "us.deepseek.v3-v1:0": ModelCapabilities(
                max_context_length=64000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "deepseek.v3-v1:0": ModelCapabilities(
                max_context_length=64000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "deepseek.v3.2": ModelCapabilities(
                max_context_length=64000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            # Qwen models
            "us.qwen.qwen3-coder-480b-a35b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                # indicative
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "qwen.qwen3-coder-480b-a35b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "qwen.qwen3-coder-30b-a3b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "us.qwen.qwen3-235b-a22b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "qwen.qwen3-235b-a22b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "qwen.qwen3-235b-a22b-2507-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "qwen.qwen3-next-80b-a3b": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "us.qwen.qwen3-32b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "qwen.qwen3-32b-v1:0": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            # Bedrock third-party frontier baselines. Pricing is left indicative
            # until exact account/model-card pricing is pulled into this repo.
            "moonshotai.kimi-k2.5": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "minimax.minimax-m2.5": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "zai.glm-5": ModelCapabilities(
                max_context_length=32000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            # OpenAI GPT-OSS (via Bedrock)
            "openai.gpt-oss-120b-1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                # indicative
                cost_per_1k_input_tokens=0.002,
                cost_per_1k_output_tokens=0.004,
            ),
            "openai.gpt-oss-20b-1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.002,
            ),
            "us.openai.gpt-oss-120b-1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.002,
                cost_per_1k_output_tokens=0.004,
            ),
            # Llama models
            "meta.llama3-8b-instruct-v1:0": ModelCapabilities(
                max_context_length=8192,
                max_output_tokens=2048,
                # indicative
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "meta.llama3-70b-instruct-v1:0": ModelCapabilities(
                max_context_length=8192,
                max_output_tokens=2048,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "meta.llama3-1-8b-instruct-v1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "us.meta.llama3-1-8b-instruct-v1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "meta.llama3-1-70b-instruct-v1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "meta.llama3-1-405b-instruct-v1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "meta.llama3-3-70b-instruct-v1:0": ModelCapabilities(
                max_context_length=128000,
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            # Llama 4 models
            "us.meta.llama4-maverick-17b-instruct-v1:0": ModelCapabilities(
                max_context_length=1000000,  # 1M context
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
            "us.meta.llama4-scout-17b-instruct-v1:0": ModelCapabilities(
                max_context_length=3500000,  # 3.5M context (Bedrock limit)
                max_output_tokens=8192,
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
            ),
        }

        model_name = (self.config.model_name or "").strip()
        lookup_candidates = [model_name]
        for prefix in ("us.", "eu.", "jp.", "global."):
            if model_name.startswith(prefix):
                lookup_candidates.append(model_name[len(prefix):])
        lookup_candidates.extend(
            candidate + ":0"
            for candidate in list(lookup_candidates)
            if candidate and not candidate.endswith(":0")
        )

        for candidate in lookup_candidates:
            if candidate in model_configs:
                capabilities = model_configs[candidate]
                if self._claude_supports_native_tools(model_name) or self._converse_supports_native_tools(model_name):
                    capabilities.supports_function_calling = True
                    capabilities.supports_native_tools = True
                    capabilities.supports_parallel_tool_calls = self._claude_supports_native_tools(model_name)
                return capabilities

        capabilities = ModelCapabilities(
            max_context_length=32000,
            max_output_tokens=8192,
        )
        if self._claude_supports_native_tools(model_name) or self._converse_supports_native_tools(model_name):
            capabilities.supports_function_calling = True
            capabilities.supports_native_tools = True
            capabilities.supports_parallel_tool_calls = self._claude_supports_native_tools(model_name)
        return capabilities
