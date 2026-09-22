"""Bounded long-term memory retrieval (`TurnBasedSessionManager.retrieve_customer_context`).

The SDK retrieves long-term memory on every user message, awaited before the
model call, through its shared read/write boto client with default retries —
so a throttled RetrieveMemoryRecords put retry backoff on first-token latency.
The override keeps the SDK's contract (namespaces from `retrieval_config`,
relevance filter, `<context_tag>` block prepended to the last user message)
and swaps in a dedicated one-attempt, short-timeout client.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from agents.main_agent.session import turn_based_session_manager as tbsm

PREFS = "/strategies/pref-1/actors/{actorId}"
FACTS = "/strategies/sem-1/actors/{actorId}"


class FakeRetrievalClient:
    def __init__(self, results=None, raise_with=None):
        self.results = results or {}
        self.raise_with = raise_with
        self.calls = []

    def retrieve_memory_records(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_with is not None:
            raise self.raise_with
        return {"memoryRecordSummaries": self.results.get(kwargs["namespacePath"], [])}


def _throttle():
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "RetrieveMemoryRecords")


def _record(text, score=0.9):
    return {"content": {"text": text}, "score": score}


def _event(text="what is my name?"):
    agent = SimpleNamespace(messages=[
        {"role": "user", "content": [{"text": "earlier"}]},
        {"role": "assistant", "content": [{"text": "ok"}]},
        {"role": "user", "content": [{"text": text}]},
    ])
    return SimpleNamespace(agent=agent)


def _manager(make_session_manager, client, namespaces=(PREFS, FACTS), relevance=0.7):
    mgr = make_session_manager()
    mgr.config.retrieval_config = {
        ns: SimpleNamespace(top_k=5, relevance_score=relevance, strategy_id=None) for ns in namespaces
    }
    mgr.config.context_tag = "user_context"
    mgr._retrieval_client = client
    return mgr


class TestRetrieval:
    def test_prepends_context_from_every_namespace_and_resolves_templates(self, make_session_manager):
        client = FakeRetrievalClient(results={
            "/strategies/pref-1/actors/test-actor": [_record("likes short answers")],
            "/strategies/sem-1/actors/test-actor": [_record("name is Ada")],
        })
        mgr = _manager(make_session_manager, client)
        mgr.config.actor_id = "test-actor"
        event = _event()

        mgr.retrieve_customer_context(event)

        assert len(client.calls) == 2
        assert {c["namespacePath"] for c in client.calls} == {
            "/strategies/pref-1/actors/test-actor", "/strategies/sem-1/actors/test-actor",
        }
        assert all(c["searchCriteria"] == {"searchQuery": "what is my name?", "topK": 5} for c in client.calls)
        assert all(c["memoryId"] == mgr.config.memory_id for c in client.calls)
        first_block = event.agent.messages[-1]["content"][0]["text"]
        assert first_block.startswith("<user_context>") and first_block.endswith("</user_context>")
        assert "likes short answers" in first_block and "name is Ada" in first_block
        # The user's own words stay last.
        assert event.agent.messages[-1]["content"][-1] == {"text": "what is my name?"}

    def test_relevance_filter_applies(self, make_session_manager):
        client = FakeRetrievalClient(results={
            "/strategies/pref-1/actors/test-actor": [_record("weak", score=0.2), _record("strong", score=0.95)],
        })
        mgr = _manager(make_session_manager, client, namespaces=(PREFS,))
        mgr.config.actor_id = "test-actor"
        event = _event()

        mgr.retrieve_customer_context(event)

        block = event.agent.messages[-1]["content"][0]["text"]
        assert "strong" in block and "weak" not in block

    def test_nothing_retrieved_means_message_untouched(self, make_session_manager):
        mgr = _manager(make_session_manager, FakeRetrievalClient())
        event = _event()
        before = [dict(b) for b in event.agent.messages[-1]["content"]]

        mgr.retrieve_customer_context(event)

        assert event.agent.messages[-1]["content"] == before

    def test_skips_when_last_message_is_not_a_text_user_message(self, make_session_manager):
        client = FakeRetrievalClient()
        mgr = _manager(make_session_manager, client)
        assistant_last = SimpleNamespace(agent=SimpleNamespace(messages=[{"role": "assistant", "content": [{"text": "x"}]}]))
        tool_result = SimpleNamespace(agent=SimpleNamespace(messages=[{"role": "user", "content": [{"toolResult": {"toolUseId": "t", "content": []}}]}]))

        mgr.retrieve_customer_context(assistant_last)
        mgr.retrieve_customer_context(tool_result)
        mgr.retrieve_customer_context(SimpleNamespace(agent=SimpleNamespace(messages=[])))

        assert client.calls == []

    def test_no_retrieval_config_means_no_client(self, make_session_manager):
        mgr = make_session_manager()
        mgr.config.retrieval_config = {}
        with patch.object(tbsm.TurnBasedSessionManager, "_get_retrieval_client") as get_client:
            mgr.retrieve_customer_context(_event())
        get_client.assert_not_called()


class TestBounded:
    def test_throttle_costs_one_attempt_and_the_turn_proceeds_without_context(self, make_session_manager):
        client = FakeRetrievalClient(raise_with=_throttle())
        mgr = _manager(make_session_manager, client)
        event = _event()

        mgr.retrieve_customer_context(event)  # must not raise

        assert len(client.calls) == 2  # one per namespace, no retries at this layer
        assert event.agent.messages[-1]["content"] == [{"text": "what is my name?"}]

    def test_one_failing_namespace_does_not_sink_the_other(self, make_session_manager):
        class HalfBroken(FakeRetrievalClient):
            def retrieve_memory_records(self, **kwargs):
                self.calls.append(kwargs)
                if "pref-1" in kwargs["namespacePath"]:
                    raise RuntimeError("boom")
                return {"memoryRecordSummaries": [_record("name is Ada")]}

        mgr = _manager(make_session_manager, HalfBroken())
        event = _event()

        mgr.retrieve_customer_context(event)

        assert "name is Ada" in event.agent.messages[-1]["content"][0]["text"]

    def test_dedicated_client_is_lazy_and_bounded(self, make_session_manager, monkeypatch):
        monkeypatch.delenv(tbsm.MEMORY_RETRIEVAL_TIMEOUT_ENV, raising=False)
        monkeypatch.delenv(tbsm.MEMORY_RETRIEVAL_MAX_ATTEMPTS_ENV, raising=False)
        mgr = make_session_manager()
        assert getattr(mgr, "_retrieval_client", None) is None

        with patch("boto3.client") as boto_client:
            boto_client.return_value = object()
            first = mgr._get_retrieval_client()
            second = mgr._get_retrieval_client()

        assert first is second
        boto_client.assert_called_once()
        assert boto_client.call_args.args == ("bedrock-agentcore",)
        cfg = boto_client.call_args.kwargs["config"]
        assert cfg.retries == {"total_max_attempts": 1, "mode": "standard"}
        assert cfg.read_timeout == 2.0 and cfg.connect_timeout == 2.0
        assert boto_client.call_args.kwargs["region_name"] == mgr.region_name

    def test_env_overrides_and_garbage(self, monkeypatch):
        monkeypatch.setenv(tbsm.MEMORY_RETRIEVAL_TIMEOUT_ENV, "0.75")
        monkeypatch.setenv(tbsm.MEMORY_RETRIEVAL_MAX_ATTEMPTS_ENV, "2")
        assert tbsm.memory_retrieval_timeout_seconds() == 0.75
        assert tbsm.memory_retrieval_max_attempts() == 2

        monkeypatch.setenv(tbsm.MEMORY_RETRIEVAL_TIMEOUT_ENV, "later")
        monkeypatch.setenv(tbsm.MEMORY_RETRIEVAL_MAX_ATTEMPTS_ENV, "0")
        assert tbsm.memory_retrieval_timeout_seconds() == 2.0
        assert tbsm.memory_retrieval_max_attempts() == 1


class TestSdkWiring:
    def test_sdk_register_hooks_dispatches_to_our_override(self):
        """The SDK registers `self.retrieve_customer_context` (both modes), so
        the override is what runs. Guards against an SDK bump that starts
        registering a private method instead."""
        import inspect

        from bedrock_agentcore.memory.integrations.strands.session_manager import (
            AgentCoreMemorySessionManager,
        )

        src = inspect.getsource(AgentCoreMemorySessionManager.register_hooks)
        assert src.count("self.retrieve_customer_context") >= 2
        assert tbsm.TurnBasedSessionManager.retrieve_customer_context is not AgentCoreMemorySessionManager.retrieve_customer_context
