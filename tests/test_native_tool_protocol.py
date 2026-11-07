import json
from types import SimpleNamespace

from src.agents.llm_agent import LLMAgent
from src.benchmark.types import ActionType, AgentContext, ToolDefinition
from src.llm.interface import (
    Message,
    ModelConfig,
    ProviderType,
    ModelCapabilities,
    ModelInterface,
    ModelResponse,
    ResponseCache,
    Role,
    ToolCall,
    ToolChoice,
    ToolSchema,
)
from src.llm.providers.bedrock_provider import BedrockProvider
from src.llm.providers.openai_provider import OpenAIProvider


class DummyModel(ModelInterface):
    def _initialize(self) -> None:
        pass

    def _raw_generate(self, messages, tools=None, tool_choice=None) -> ModelResponse:
        return ModelResponse(content="", model_name="dummy", provider="dummy")

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()


class SequencedNativeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, expect_json=False, tools=None, tool_choice=None):
        self.calls.append(
            {
                "messages": messages,
                "expect_json": expect_json,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        if not self.responses:
            raise AssertionError("No queued model response")
        return self.responses.pop(0)


class _ResponseBody:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload)


def _context() -> AgentContext:
    return AgentContext(
        budget_remaining=10,
        budget_total=20,
        max_queries_per_call=3,
        max_rows_per_query=100,
        user_request="Check the record.",
        database_name="unit",
        available_tools=[
            ToolDefinition(
                name="explore_database",
                description="Read-only SQL exploration",
                parameters_schema={
                    "type": "object",
                    "properties": {"queries": {"type": "array"}},
                    "required": ["queries"],
                },
            )
        ],
        session_tools=[],
    )


def test_response_cache_distinguishes_native_tools() -> None:
    cache = ResponseCache()
    config = ModelConfig(provider=ProviderType.OPENAI, model_name="gpt-5.4")
    messages = [Message(role=Role.USER, content="hello")]
    tool_a = [ToolSchema(name="a", description="tool a", input_schema={})]
    tool_b = [ToolSchema(name="b", description="tool b", input_schema={})]

    key_a = cache._generate_key(messages, config, tool_a, ToolChoice.AUTO)
    key_b = cache._generate_key(messages, config, tool_b, ToolChoice.AUTO)

    assert key_a != key_b


def test_response_cache_tool_schema_key_is_deterministic_for_dict_order() -> None:
    cache = ResponseCache()
    config = ModelConfig(provider=ProviderType.OPENAI, model_name="gpt-5.4")
    messages = [Message(role=Role.USER, content="hello")]
    tool_a = [
        ToolSchema(
            name="a",
            description="tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "number"}}},
        )
    ]
    tool_b = [
        ToolSchema(
            name="a",
            description="tool",
            input_schema={"properties": {"y": {"type": "number"}, "x": {"type": "string"}}, "type": "object"},
        )
    ]

    assert cache._generate_key(messages, config, tool_a, ToolChoice.AUTO) == cache._generate_key(
        messages,
        config,
        tool_b,
        ToolChoice.AUTO,
    )


def test_response_cache_distinguishes_provider_settings() -> None:
    cache = ResponseCache()
    messages = [Message(role=Role.USER, content="hello")]
    base = ModelConfig(provider=ProviderType.OPENAI, model_name="gpt-5.4")
    xhigh = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-5.4",
        provider_settings={"reasoning_effort": "xhigh"},
    )

    key_base = cache._generate_key(messages, base)
    key_xhigh = cache._generate_key(messages, xhigh)

    assert key_base != key_xhigh


def test_native_finish_tool_maps_to_reject() -> None:
    agent = LLMAgent.__new__(LLMAgent)
    call = ToolCall(
        id="call_1",
        name="finish",
        arguments={
            "status": "reject",
            "reasoning": "No current authorization.",
            "rejection_reason": "Missing active authorization.",
        },
    )

    response = agent._parse_native_tool_call(call, _context())

    assert response.action_type == ActionType.FINISH
    assert response.is_rejection is True
    assert response.rejection_reason == "Missing active authorization."


def test_native_finish_tool_invalid_status_rejects() -> None:
    agent = LLMAgent.__new__(LLMAgent)
    call = ToolCall(
        id="call_1",
        name="finish",
        arguments={
            "status": "rejected",
            "reasoning": "Misspelled status should not become success.",
        },
    )

    response = agent._parse_native_tool_call(call, _context())

    assert response.action_type == ActionType.FINISH
    assert response.is_rejection is True
    assert "Invalid native finish status" in response.rejection_reason


def test_native_tool_call_maps_to_agent_tool_response() -> None:
    agent = LLMAgent.__new__(LLMAgent)
    call = ToolCall(
        id="call_1",
        name="explore_database",
        arguments={
            "reasoning": "Inspect schema first.",
            "queries": [{"sql": "SELECT 1", "purpose": "smoke"}],
        },
    )

    response = agent._parse_native_tool_call(call, _context())

    assert response.action_type == ActionType.TOOL
    assert response.tool_name == "explore_database"
    assert response.tool_params == {"queries": [{"sql": "SELECT 1", "purpose": "smoke"}]}


def test_native_tool_schema_adds_optional_reasoning() -> None:
    tool = ToolDefinition(
        name="write_db",
        description="Write SQL",
        parameters_schema={
            "type": "object",
            "properties": {"queries": {"type": "array"}},
            "required": ["queries"],
        },
    )

    schema = LLMAgent._native_tool_input_schema(tool)

    assert "reasoning" in schema["properties"]
    assert schema["required"] == ["queries"]


def test_native_step_retries_when_provider_returns_no_tool_call(monkeypatch) -> None:
    agent = LLMAgent.__new__(LLMAgent)
    agent.model_name = "gpt-5.5"
    agent.llm = SequencedNativeModel(
        [
            ModelResponse(
                content="",
                model_name="gpt-5.5",
                provider="openai",
                metadata={"status": "completed"},
            ),
            ModelResponse(
                content="",
                model_name="gpt-5.5",
                provider="openai",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="finish",
                        arguments={"status": "done", "reasoning": "complete after retry"},
                    )
                ],
            ),
        ]
    )
    monkeypatch.setenv("EVAL_NATIVE_NO_TOOL_RETRIES", "1")

    response = agent._native_step(_context())

    assert response.action_type == ActionType.FINISH
    assert response.is_rejection is False
    assert len(agent.llm.calls) == 2
    retry_prompt = agent.llm.calls[1]["messages"][-1].content
    assert "Protocol correction" in retry_prompt
    assert "call exactly one currently callable native tool" in retry_prompt


def test_native_step_no_tool_after_retry_is_protocol_artifact(monkeypatch) -> None:
    agent = LLMAgent.__new__(LLMAgent)
    agent.model_name = "gpt-5.5"
    agent.llm = SequencedNativeModel(
        [
            ModelResponse(
                content="",
                model_name="gpt-5.5",
                provider="openai",
                metadata={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            ),
            ModelResponse(
                content="",
                model_name="gpt-5.5",
                provider="openai",
                metadata={"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            ),
        ]
    )
    monkeypatch.setenv("EVAL_NATIVE_NO_TOOL_RETRIES", "1")

    response = agent._native_step(_context())

    assert response.action_type == ActionType.FINISH
    assert response.is_rejection is True
    assert "protocol" in response.rejection_reason.lower()
    assert "max_output_tokens" in response.rejection_reason
    assert len(agent.llm.calls) == 2


def test_truncated_json_repair_does_not_create_write_action() -> None:
    truncated_write = (
        '{"reasoning":"partial","tool_name":"write_db","tool_params":'
        '{"queries":[{"sql":"UPDATE accounts SET status = '
    )

    parsed = LLMAgent._extract_json(truncated_write)
    assert not LLMAgent._is_agent_response_payload(parsed)
    parsed_response = LLMAgent._parse_response(parsed)
    assert parsed_response.action_type == ActionType.FINISH
    assert parsed_response.is_rejection is True

    model = DummyModel(ModelConfig(provider=ProviderType.OPENAI, model_name="dummy"))
    response = ModelResponse(content=truncated_write, model_name="dummy", provider="dummy")
    extracted = model._extract_json_response(response)

    assert extracted.metadata.get("parsed_json") is None
    assert extracted.metadata.get("json_extraction_failed") is True


def test_control_char_json_fix_does_not_create_write_action() -> None:
    malformed_write = (
        '{"reasoning":"line one\nline two","tool_name":"write_db",'
        '"tool_params":{"queries":[{"sql":"UPDATE accounts SET status = \'closed\'"}]}}'
    )

    model = DummyModel(ModelConfig(provider=ProviderType.OPENAI, model_name="dummy"))
    response = ModelResponse(content=malformed_write, model_name="dummy", provider="dummy")
    extracted = model._extract_json_response(response)

    assert extracted.metadata.get("parsed_json") is None
    assert extracted.metadata.get("json_extraction_failed") is True


def test_nested_state_changing_json_is_not_executed() -> None:
    malformed_outer = (
        '{"outer": {"tool_name":"write_db","tool_params":'
        '{"queries":[{"sql":"UPDATE accounts SET status = \'closed\'"}]}}'
    )

    parsed = LLMAgent._extract_json(malformed_outer)

    assert not LLMAgent._is_state_changing_payload(parsed)


def test_recover_action_from_text_requires_json_like_prefix() -> None:
    agent = LLMAgent.__new__(LLMAgent)
    text = 'I considered "tool_name": "explore_database" but will not emit JSON.'

    assert agent._recover_action_from_text(text, _context()) is None


def test_recover_action_from_text_does_not_substitute_other_tools() -> None:
    agent = LLMAgent.__new__(LLMAgent)
    text = '{"tool_name": "ask_user", "tool_params": {"question": '

    assert agent._recover_action_from_text(text, _context()) is None


def test_legacy_json_protocol_retries_malformed_json(monkeypatch) -> None:
    agent = LLMAgent.__new__(LLMAgent)
    agent.model_name = "bedrock/glm-5"
    agent.llm = SequencedNativeModel(
        [
            ModelResponse(
                content=(
                    '{"reasoning":"Need to inspect","tool_name":"explore_database",'
                    '"tool_params":{"queries":[{"sql":"SELECT "Status" FROM public.example"}]}}'
                ),
                model_name="bedrock/glm-5",
                provider="bedrock",
                metadata={"json_extraction_failed": True},
            ),
            ModelResponse(
                content=(
                    '{"reasoning":"Need to inspect","tool_name":"explore_database",'
                    '"tool_params":{"queries":[{"sql":"SELECT \\"Status\\" FROM public.example"}]}}'
                ),
                model_name="bedrock/glm-5",
                provider="bedrock",
                metadata={
                    "parsed_json": {
                        "reasoning": "Need to inspect",
                        "tool_name": "explore_database",
                        "tool_params": {
                            "queries": [
                                {"sql": 'SELECT "Status" FROM public.example'}
                            ]
                        },
                    }
                },
            ),
        ]
    )
    monkeypatch.delenv("EVAL_NATIVE_TOOLS", raising=False)
    monkeypatch.setenv("EVAL_LEGACY_JSON_RETRIES", "1")

    response = agent.step(_context())

    assert response.action_type == ActionType.TOOL
    assert response.tool_name == "explore_database"
    assert response.tool_params == {
        "queries": [{"sql": 'SELECT "Status" FROM public.example'}]
    }
    assert len(agent.llm.calls) == 2
    retry_prompt = agent.llm.calls[1]["messages"][-1].content
    assert "not valid JSON" in retry_prompt
    assert '\\"Status\\"' in retry_prompt


def test_openai_native_tool_schema_and_argument_parsing() -> None:
    tool = ToolSchema(
        name="explore_database",
        description="Read-only SQL exploration",
        input_schema={"type": "object", "properties": {"queries": {"type": "array"}}},
    )

    responses_schema = OpenAIProvider._responses_tool_schema(tool)
    chat_schema = OpenAIProvider._chat_tool_schema(tool)

    assert responses_schema["type"] == "function"
    assert responses_schema["name"] == "explore_database"
    assert responses_schema["strict"] is False
    assert chat_schema["function"]["name"] == "explore_database"
    assert OpenAIProvider._parse_tool_arguments('{"queries": []}') == {"queries": []}
    assert OpenAIProvider._parse_tool_arguments('[{"sql": "SELECT 1"}]', "explore_database") == {
        "queries": [{"sql": "SELECT 1"}]
    }
    assert OpenAIProvider._parse_tool_arguments("not json") == {}


def test_openai_chat_raw_generate_parses_tool_calls() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        enable_caching=False,
    )
    captured = {}

    def create(**params):
        captured.update(params)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="explore_database",
                                    arguments='{"queries": [{"sql": "SELECT 1"}]}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = provider._raw_generate(
        [Message(role=Role.USER, content="inspect")],
        tools=[ToolSchema(name="explore_database", description="read", input_schema={})],
        tool_choice=ToolChoice.AUTO,
    )

    assert captured["parallel_tool_calls"] is False
    assert response.content == ""
    assert response.tool_calls[0].name == "explore_database"
    assert response.tool_calls[0].arguments == {"queries": [{"sql": "SELECT 1"}]}


def test_openai_responses_raw_generate_parses_function_calls() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-5.4",
        enable_caching=False,
    )
    captured = {}

    def create(**params):
        captured.update(params)
        return SimpleNamespace(
            id="resp_1",
            status="completed",
            output_text="",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="finish",
                    arguments='{"status": "done", "reasoning": "complete"}',
                )
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=4, total_tokens=24),
        )

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))

    response = provider._raw_generate(
        [Message(role=Role.USER, content="finish")],
        tools=[ToolSchema(name="finish", description="finish", input_schema={})],
        tool_choice=ToolChoice.AUTO,
    )

    assert captured["parallel_tool_calls"] is False
    assert response.tool_calls[0].name == "finish"
    assert response.tool_calls[0].arguments["status"] == "done"


def test_openai_responses_raw_generate_preserves_incomplete_details() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-5.4",
        enable_caching=False,
    )

    def create(**params):
        return SimpleNamespace(
            id="resp_1",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output_text="",
            output=[],
            usage=SimpleNamespace(input_tokens=20, output_tokens=4, total_tokens=24),
        )

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))

    response = provider._raw_generate(
        [Message(role=Role.USER, content="finish")],
        tools=[ToolSchema(name="finish", description="finish", input_schema={})],
        tool_choice=ToolChoice.REQUIRED,
    )

    assert response.metadata["status"] == "incomplete"
    assert "max_output_tokens" in str(response.metadata["incomplete_details"])


def test_bedrock_claude_native_tool_schema() -> None:
    tool = ToolSchema(
        name="finish",
        description="Finish request",
        input_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )

    schema = BedrockProvider._claude_tool_schema(tool)

    assert schema == {
        "name": "finish",
        "description": "Finish request",
        "input_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
    }


def test_bedrock_claude_tool_choice_disables_parallel_tool_use() -> None:
    assert BedrockProvider._claude_tool_choice(ToolChoice.AUTO) == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert BedrockProvider._claude_tool_choice(ToolChoice.REQUIRED) == {
        "type": "any",
        "disable_parallel_tool_use": True,
    }
    assert BedrockProvider._claude_tool_choice(ToolChoice.NONE) == {"type": "none"}


def test_bedrock_model_prefix_helpers_are_specific() -> None:
    assert BedrockProvider._is_openai_like_model("us.openai.gpt-oss-120b-1:0")
    assert BedrockProvider._is_openai_like_model("deepseek.v3-v1:0")
    assert not BedrockProvider._is_openai_like_model("custom.not-openai-string")
    assert BedrockProvider._is_qwen_model("us.qwen.qwen3-32b-v1:0")
    assert not BedrockProvider._is_qwen_model("custom.qwenish")


def test_bedrock_claude_raw_generate_parses_tool_use() -> None:
    provider = BedrockProvider.__new__(BedrockProvider)
    provider.config = ModelConfig(
        provider=ProviderType.BEDROCK,
        model_name="us.anthropic.claude-opus-4-7",
        enable_caching=False,
    )
    provider.aws_region = "us-east-1"
    provider.ClientError = Exception
    captured = {}

    class FakeBedrock:
        def invoke_model(self, **params):
            captured.update(params)
            return {
                "body": _ResponseBody(
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "read_knowledge",
                                "input": {},
                            }
                        ],
                        "usage": {"input_tokens": 12, "output_tokens": 3},
                        "stop_reason": "tool_use",
                    }
                )
            }

    provider.bedrock = FakeBedrock()

    response = provider._raw_generate(
        [Message(role=Role.USER, content="read policy")],
        tools=[ToolSchema(name="read_knowledge", description="read", input_schema={})],
        tool_choice=ToolChoice.AUTO,
    )
    sent_body = json.loads(captured["body"])

    assert sent_body["tool_choice"]["disable_parallel_tool_use"] is True
    assert response.tool_calls[0].name == "read_knowledge"
    assert "response_body" not in response.metadata
