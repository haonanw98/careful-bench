"""Model factory for creating LLM instances."""

import os
from typing import Dict, Type, Optional

from .interface import ModelInterface, ProviderType
from .providers import OpenAIProvider, AnthropicProvider, BedrockProvider


class ModelFactory:
    """Factory for creating model interfaces with optimal configurations."""

    # Provider class mappings
    PROVIDERS: Dict[ProviderType, Type[ModelInterface]] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.BEDROCK: BedrockProvider,
    }

    # Model presets with optimal configurations
    MODEL_PRESETS = {
        # OpenAI Models
        "gpt-4": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "gpt-4-turbo": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-4-turbo-preview",
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "gpt-4o": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "gpt-4o-mini": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "gpt-3.5-turbo": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "gpt-5.2": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5.2",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "medium"},
        },
        "gpt-5-mini": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5-mini",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "medium"},
        },
        "gpt-5.4": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5.4",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "medium"},
        },
        "gpt-5.3-codex": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5.3-codex",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "medium"},
        },
        "gpt-5.4-mini": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5.4-mini",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "medium"},
        },
        "gpt-5.4-pro": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5.4-pro",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "high"},
        },
        "gpt-5.5": {
            "provider": ProviderType.OPENAI,
            "model_name": "gpt-5.5",
            "temperature": 0.1,
            "max_tokens": 16384,
            "provider_settings": {"reasoning_effort": "medium"},
        },
        "gpt-oss-120b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "openai.gpt-oss-120b-1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "gpt-oss-20b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "openai.gpt-oss-20b-1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "o3": {
            "provider": ProviderType.OPENAI,
            "model_name": "o3",
            "temperature": 0.1,
            "max_tokens": 16384,
        },

        # Bedrock Claude 4.7 / 4.6 Models
        "bedrock/claude-4.7-opus": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-opus-4-7",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "bedrock/claude-4.6-opus": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-opus-4-6-v1",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "bedrock/claude-4.6-sonnet": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-sonnet-4-6",
            "temperature": 0.1,
            "max_tokens": 16384,
        },

        # Bedrock Claude 4.5 Models
        "bedrock/claude-4.5-opus": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "bedrock/claude-4.5-sonnet": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "bedrock/claude-4.5-haiku": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        # Bedrock Claude 4.1 Models
        "bedrock/claude-4.1-opus": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-opus-4-1-20250805-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        # Bedrock Claude 4 Models
        "bedrock/claude-4-opus": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-opus-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "bedrock/claude-4-sonnet": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        # Bedrock Claude 3.x Models
        "bedrock/claude-3.7-sonnet": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        "bedrock/claude-3.5-sonnet": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/claude-3.5-haiku": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/claude-3-haiku": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.anthropic.claude-3-haiku-20240307-v1:0",
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        # Bedrock DeepSeek Models
        "bedrock/deepseek-r1": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.deepseek.r1-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/deepseek-v3": {
            "provider": ProviderType.BEDROCK,
            "model_name": "deepseek.v3-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/deepseek-v3.2": {
            "provider": ProviderType.BEDROCK,
            "model_name": "deepseek.v3.2",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        # Bedrock Qwen Models
        "bedrock/qwen3-coder-480b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "qwen.qwen3-coder-480b-a35b-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/qwen3-coder-30b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "qwen.qwen3-coder-30b-a3b-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/qwen3-235b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "qwen.qwen3-235b-a22b-2507-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/qwen3-next-80b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "qwen.qwen3-next-80b-a3b",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/qwen3-32b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "qwen.qwen3-32b-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        # Bedrock third-party frontier baselines
        "bedrock/kimi-k2.5": {
            "provider": ProviderType.BEDROCK,
            "model_name": "moonshotai.kimi-k2.5",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/minimax-m2.5": {
            "provider": ProviderType.BEDROCK,
            "model_name": "minimax.minimax-m2.5",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/glm-5": {
            "provider": ProviderType.BEDROCK,
            "model_name": "zai.glm-5",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        # Bedrock Llama Models
        "bedrock/llama3-8b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "meta.llama3-8b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "bedrock/llama3-70b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "meta.llama3-70b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "bedrock/llama3.1-8b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.meta.llama3-1-8b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/llama3.1-70b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "meta.llama3-1-70b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/llama3.1-405b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "meta.llama3-1-405b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/llama3.3-70b": {
            "provider": ProviderType.BEDROCK,
            "model_name": "meta.llama3-3-70b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        # Bedrock Llama 4 Models
        "bedrock/llama4-maverick": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.meta.llama4-maverick-17b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "bedrock/llama4-scout": {
            "provider": ProviderType.BEDROCK,
            "model_name": "us.meta.llama4-scout-17b-instruct-v1:0",
            "temperature": 0.1,
            "max_tokens": 8192,
        },
    }

    @classmethod
    def create(
        cls,
        model_name: str,
        provider: Optional[ProviderType] = None,
        **kwargs
    ) -> ModelInterface:
        """Create a model interface instance.

        Args:
            model_name: Name of the model or preset key
            provider: Optional provider type override
            **kwargs: Additional configuration to override defaults

        Returns:
            ModelInterface instance
        """
        # Check if it's a preset
        if model_name in cls.MODEL_PRESETS:
            preset = cls.MODEL_PRESETS[model_name].copy()
            provider = preset.pop("provider")
            actual_model_name = preset.pop("model_name", model_name)
            config = {**preset, **kwargs}
        else:
            actual_model_name = model_name
            config = kwargs
            provider = None

        # Determine provider if not specified
        if provider is None:
            provider = cls._infer_provider(actual_model_name)

        cls._apply_eval_overrides(config, provider)

        # Get provider class
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_class = cls.PROVIDERS[provider]

        # Create ModelConfig instance
        from .interface import ModelConfig
        config['provider'] = provider
        model_config = ModelConfig(model_name=actual_model_name, **config)

        # Create instance
        return provider_class(model_config)

    @staticmethod
    def _apply_eval_overrides(config: Dict, provider: ProviderType) -> None:
        """
        Apply opt-in evaluation cost controls.

        These environment variables are intentionally caps/overrides rather
        than preset changes, so normal model behavior stays unchanged unless a
        caller explicitly requests a cheaper run.
        """
        max_tokens_raw = os.getenv("EVAL_MODEL_MAX_TOKENS", "").strip()
        if max_tokens_raw:
            try:
                max_tokens_cap = int(max_tokens_raw)
            except ValueError:
                max_tokens_cap = 0
            if max_tokens_cap > 0:
                current_max = int(config.get("max_tokens", max_tokens_cap))
                config["max_tokens"] = min(current_max, max_tokens_cap)

        reasoning_effort = os.getenv("EVAL_OPENAI_REASONING_EFFORT", "").strip()
        if reasoning_effort and provider == ProviderType.OPENAI:
            provider_settings = dict(config.get("provider_settings") or {})
            provider_settings["reasoning_effort"] = reasoning_effort
            config["provider_settings"] = provider_settings

        if provider == ProviderType.OPENAI:
            timeout_raw = os.getenv("EVAL_OPENAI_TIMEOUT_SECONDS", "").strip()
            if timeout_raw:
                try:
                    timeout_seconds = int(timeout_raw)
                except ValueError:
                    timeout_seconds = 0
                if timeout_seconds > 0:
                    config["timeout"] = timeout_seconds

            prompt_cache_retention = os.getenv("EVAL_OPENAI_PROMPT_CACHE_RETENTION", "").strip()
            prompt_cache_key = os.getenv("EVAL_OPENAI_PROMPT_CACHE_KEY", "").strip()
            if prompt_cache_retention or prompt_cache_key:
                provider_settings = dict(config.get("provider_settings") or {})
                if prompt_cache_retention:
                    provider_settings["prompt_cache_retention"] = prompt_cache_retention
                if prompt_cache_key:
                    provider_settings["prompt_cache_key"] = prompt_cache_key
                config["provider_settings"] = provider_settings

    @classmethod
    def _infer_provider(cls, model_name: str) -> ProviderType:
        """Infer provider from model name."""
        name_lower = model_name.lower()

        bedrock_prefixes = (
            "anthropic.",
            "us.anthropic.",
            "us.deepseek.",
            "deepseek.",
            "us.qwen.",
            "qwen.",
            "moonshot.",
            "moonshotai.",
            "minimax.",
            "zai.",
            "meta.llama",
            "us.meta.",
            "openai.",
            "us.openai.",
        )

        if model_name.startswith(bedrock_prefixes):
            return ProviderType.BEDROCK
        elif "claude" in name_lower:
            return ProviderType.ANTHROPIC
        elif "gpt" in name_lower:
            return ProviderType.OPENAI
        else:
            return ProviderType.OPENAI


def create_model(model_name: str, **kwargs) -> ModelInterface:
    """Convenience function to create a model.

    Args:
        model_name: Name of the model or preset key
        **kwargs: Additional configuration

    Returns:
        ModelInterface instance
    """
    return ModelFactory.create(model_name, **kwargs)
