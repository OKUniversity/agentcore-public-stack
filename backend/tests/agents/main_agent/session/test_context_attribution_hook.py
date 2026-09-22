"""Tests for ContextAttributionHook — per-turn system/tools/messages breakdown.

The hook computes the split once at cold start (caching the stable
system/tools tokens on the agent) and derives the messages partition each turn
from the authoritative projected total. Tool overhead deliberately absorbs the
tool-use scaffolding (full - count(system + messages, no tools)).
"""

import pytest
from strands.hooks import BeforeModelCallEvent

from agents.main_agent.session.hooks.context_attribution import (
    ContextAttributionHook,
    clear_probe_baselines,
    clear_split_memo,
    get_context_breakdown,
)


class FakeModel:
    """Async count_tokens returning system + per-message + (tools→overhead).

    Mirrors the real behavior the hook relies on: tool overhead is only counted
    when tool_specs are supplied (the empty-messages / no-tools baselines never
    include it).
    """

    def __init__(self, system=100, per_msg=10, tool_overhead=500, raise_on_count=False):
        self.system = system
        self.per_msg = per_msg
        self.tool_overhead = tool_overhead
        self.raise_on_count = raise_on_count
        self.calls = []

    async def count_tokens(self, messages, tool_specs=None, system_prompt=None, system_prompt_content=None):
        self.calls.append({"n_messages": len(messages), "has_tools": bool(tool_specs)})
        if self.raise_on_count:
            raise RuntimeError("count failed")
        total = self.system if system_prompt else 0
        total += len(messages) * self.per_msg
        total += self.tool_overhead if tool_specs else 0
        return total


class FakeToolRegistry:
    def __init__(self, specs):
        self._specs = specs

    def get_all_tool_specs(self):
        return self._specs


class FakeAgent:
    def __init__(self, model, messages, system_prompt="SYSTEM-PROMPT", tool_specs=None):
        self.model = model
        self.messages = messages
        self.system_prompt = system_prompt
        self._system_prompt_content = None
        self.tool_registry = FakeToolRegistry(tool_specs if tool_specs is not None else [{"name": "t"}])


def _event(agent, projected):
    return BeforeModelCallEvent(agent=agent, projected_input_tokens=projected)


def _parts(breakdown):
    return {p["key"]: p["tokens"] for p in breakdown["partitions"]}


class TestColdStart:
    @pytest.mark.asyncio
    async def test_computes_partitions_that_sum_to_total(self):
        model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=650))

        bd = get_context_breakdown(agent)
        parts = _parts(bd)
        # system_only=100; no_tools(1 msg)=110; toolTokens=650-110=540; messages=650-100-540=10
        assert parts == {"system": 100, "tools": 540, "messages": 10}
        assert bd["total"] == 650
        assert sum(parts.values()) == bd["total"]

    @pytest.mark.asyncio
    async def test_makes_exactly_three_count_calls_none_with_tools_and_none_empty(self):
        model = FakeModel()
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=650))

        assert model.calls == [
            {"n_messages": 1, "has_tools": False},  # probe only (per-model baseline)
            {"n_messages": 1, "has_tools": False},  # probe + system
            {"n_messages": 1, "has_tools": False},  # system + messages, no tools
        ]
        # Bedrock rejects an empty conversation; the hook must never send one.
        assert all(c["n_messages"] >= 1 for c in model.calls)

    @pytest.mark.asyncio
    async def test_tool_partition_absorbs_scaffolding(self):
        # full (projected) deliberately exceeds the no-tools baseline by more
        # than bare schemas would — the surplus is the scaffolding, and it must
        # land in `tools`, not `messages`.
        model = FakeModel(system=100, per_msg=10)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=900))

        parts = _parts(get_context_breakdown(agent))
        assert parts["tools"] == 900 - 110  # = 790, all of it on tools
        assert parts["messages"] == 10      # the actual single message, not inflated


class TestWarmTurn:
    @pytest.mark.asyncio
    async def test_reuses_cached_split_without_recounting(self):
        model = FakeModel(system=100, per_msg=10)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        hook = ContextAttributionHook()
        await hook._on_before_model_call(_event(agent, projected=650))
        calls_after_cold = len(model.calls)

        # Conversation grows; only the projected total changes.
        agent.messages = agent.messages + [{"role": "assistant", "content": [{"text": "ok"}]}]
        await hook._on_before_model_call(_event(agent, projected=700))

        assert len(model.calls) == calls_after_cold  # no new CountTokens calls
        parts = _parts(get_context_breakdown(agent))
        assert parts["system"] == 100
        assert parts["tools"] == 540
        assert parts["messages"] == 700 - 100 - 540  # grows with the turn


class TestProjectedUnavailable:
    @pytest.mark.asyncio
    async def test_cold_start_counts_full_with_tools_when_projected_none(self):
        model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=None))

        bd = get_context_breakdown(agent)
        parts = _parts(bd)
        # full counted with tools = 100 + 10 + 500 = 610; tools = 610-110 = 500
        assert bd["total"] == 610
        assert parts == {"system": 100, "tools": 500, "messages": 10}
        # 4 calls: probe baseline, probe+system, no-tools, then full WITH tools
        assert len(model.calls) == 4
        assert model.calls[3]["has_tools"] is True

    @pytest.mark.asyncio
    async def test_warm_turn_without_projected_leaves_breakdown_untouched(self):
        model = FakeModel()
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        hook = ContextAttributionHook()
        await hook._on_before_model_call(_event(agent, projected=650))
        bd_before = get_context_breakdown(agent)

        await hook._on_before_model_call(_event(agent, projected=None))
        assert get_context_breakdown(agent) == bd_before


class TestRobustness:
    @pytest.mark.asyncio
    async def test_count_failure_is_swallowed_and_yields_no_breakdown(self):
        model = FakeModel(raise_on_count=True)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        # Must not raise.
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=650))
        assert get_context_breakdown(agent) is None

    def test_get_context_breakdown_is_none_when_absent(self):
        agent = FakeAgent(FakeModel(), messages=[])
        assert get_context_breakdown(agent) is None


class TestInlineAttachmentGuard:
    """``toolTokens`` is a residual between two independently sourced counts
    (``full`` from Strands' projection, ``no_tools`` from our CountTokens call),
    so any disagreement about how a content block is counted lands wholly in it.

    Measured on dev 2026-09-16 (session ``61de2256``): a call reported
    ``toolTokens`` of 106,756 where the session's real tools prefix was 12,516 —
    a 94,240 difference against a document measured at ~94,485, i.e. the entire
    document attributed to tools. The split is therefore not computed while
    inline bytes are in context.
    """

    def _doc_message(self):
        return {
            "role": "user",
            "content": [
                {"text": "what does it say?"},
                {"document": {"format": "pdf", "name": "d_pdf", "source": {"bytes": b"%PDF-1.4"}}},
            ],
        }

    @pytest.mark.asyncio
    async def test_no_split_is_computed_while_a_document_is_inline(self):
        model = FakeModel()
        agent = FakeAgent(model, messages=[self._doc_message()])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=100_000))

        assert get_context_breakdown(agent) is None, "a contaminated split must not be published"
        assert model.calls == [], "and it must not pay for CountTokens to compute one"

    @pytest.mark.asyncio
    async def test_an_image_counts_too(self):
        model = FakeModel()
        agent = FakeAgent(model, messages=[{
            "role": "user",
            "content": [{"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}],
        }])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=50_000))
        assert get_context_breakdown(agent) is None

    @pytest.mark.asyncio
    async def test_a_digest_is_not_an_attachment_so_the_split_is_taken(self):
        """The offload turns the document into text, which counts normally —
        so an attachment session still gets ``prefixTokens`` from turn 2."""
        model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        agent = FakeAgent(model, messages=[{
            "role": "user",
            "content": [{"text": '<document-digest name="d.pdf" upload_id="u1" pages="60">'}],
        }])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=650))
        assert _parts(get_context_breakdown(agent)) == {"system": 100, "tools": 540, "messages": 10}

    @pytest.mark.asyncio
    async def test_a_later_clean_turn_computes_the_split(self):
        """Deferred, not abandoned: the same agent takes the split once the
        attachment has left the live context."""
        model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        agent = FakeAgent(model, messages=[self._doc_message()])
        hook = ContextAttributionHook()
        await hook._on_before_model_call(_event(agent, projected=100_000))
        assert get_context_breakdown(agent) is None

        agent.messages = [{"role": "user", "content": [{"text": "follow-up"}]}]
        await hook._on_before_model_call(_event(agent, projected=650))
        assert _parts(get_context_breakdown(agent)) == {"system": 100, "tools": 540, "messages": 10}

    @pytest.mark.asyncio
    async def test_a_cached_split_is_still_used_when_a_document_arrives_later(self):
        """The guard only defers the *computation*. An agent that already has a
        trustworthy split keeps reporting against it."""
        model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        hook = ContextAttributionHook()
        await hook._on_before_model_call(_event(agent, projected=650))

        agent.messages = [{"role": "user", "content": [{"text": "hi"}]}, self._doc_message()]
        await hook._on_before_model_call(_event(agent, projected=95_000))
        parts = _parts(get_context_breakdown(agent))
        assert parts["system"] == 100 and parts["tools"] == 540
        assert parts["messages"] == 95_000 - 640, "the document lands in messages, where it belongs"


class TestSessionSplitMemo:
    """A rebuilt Agent for the same session + configuration adopts the split
    its predecessor measured instead of paying two CountTokens calls again.

    Sessions whose injected tools keep them out of the agent cache rebuild
    their Agent every turn; without the memo that was 2 extra counts per turn
    (docs/specs/load-test-assessment-2026-09.md §1 fix 1)."""

    def setup_method(self):
        clear_split_memo()

    def teardown_method(self):
        clear_split_memo()

    @pytest.mark.asyncio
    async def test_second_agent_for_same_session_makes_no_count_calls(self):
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        first_model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        first = FakeAgent(first_model, messages=list(msgs))
        await ContextAttributionHook(session_id="s1")._on_before_model_call(_event(first, projected=650))
        assert len(first_model.calls) == 3

        second_model = FakeModel(system=100, per_msg=10, tool_overhead=500)
        second = FakeAgent(second_model, messages=list(msgs) + [{"role": "assistant", "content": [{"text": "yo"}]}])
        await ContextAttributionHook(session_id="s1")._on_before_model_call(_event(second, projected=660))

        assert second_model.calls == []
        assert _parts(get_context_breakdown(second)) == {"system": 100, "tools": 540, "messages": 20}

    @pytest.mark.asyncio
    async def test_different_session_recounts(self):
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        await ContextAttributionHook(session_id="s1")._on_before_model_call(
            _event(FakeAgent(FakeModel(), messages=list(msgs)), projected=650)
        )
        other_model = FakeModel()
        await ContextAttributionHook(session_id="s2")._on_before_model_call(
            _event(FakeAgent(other_model, messages=list(msgs)), projected=650)
        )
        assert len(other_model.calls) == 3

    @pytest.mark.asyncio
    async def test_changed_tools_or_prompt_recounts(self):
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        await ContextAttributionHook(session_id="s1")._on_before_model_call(
            _event(FakeAgent(FakeModel(), messages=list(msgs)), projected=650)
        )

        tools_changed = FakeModel()
        await ContextAttributionHook(session_id="s1")._on_before_model_call(
            _event(FakeAgent(tools_changed, messages=list(msgs), tool_specs=[{"name": "t"}, {"name": "u"}]), projected=700)
        )
        assert len(tools_changed.calls) == 3

        prompt_changed = FakeModel()
        await ContextAttributionHook(session_id="s1")._on_before_model_call(
            _event(FakeAgent(prompt_changed, messages=list(msgs), system_prompt="OTHER"), projected=650)
        )
        assert len(prompt_changed.calls) == 3

    @pytest.mark.asyncio
    async def test_no_session_id_means_instance_only(self):
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        await ContextAttributionHook()._on_before_model_call(
            _event(FakeAgent(FakeModel(), messages=list(msgs)), projected=650)
        )
        again = FakeModel()
        await ContextAttributionHook()._on_before_model_call(
            _event(FakeAgent(again, messages=list(msgs)), projected=650)
        )
        assert len(again.calls) == 3

    @pytest.mark.asyncio
    async def test_a_deferred_split_is_not_memoised(self):
        # Inline attachment → the split is skipped, so nothing must be stored
        # for a later clean agent to adopt.
        pdf = {"role": "user", "content": [{"document": {"format": "pdf", "name": "d", "source": {"bytes": b"%PDF"}}}]}
        skipped = FakeModel()
        await ContextAttributionHook(session_id="s1")._on_before_model_call(
            _event(FakeAgent(skipped, messages=[pdf]), projected=650)
        )
        assert skipped.calls == []

        clean = FakeModel()
        await ContextAttributionHook(session_id="s1")._on_before_model_call(
            _event(FakeAgent(clean, messages=[{"role": "user", "content": [{"text": "hi"}]}]), projected=650)
        )
        assert len(clean.calls) == 3

    def test_memo_is_bounded(self):
        from agents.main_agent.session.hooks import context_attribution as ca

        for i in range(ca._SPLIT_MEMO_MAX + 50):
            ca._memo_put((f"s{i}", "p", "t"), {"systemTokens": 1, "toolTokens": 1})
        assert len(ca._split_memo) == ca._SPLIT_MEMO_MAX
        assert ca._memo_get(("s0", "p", "t")) is None
        assert ca._memo_get((f"s{ca._SPLIT_MEMO_MAX + 49}", "p", "t")) is not None


class TestProbeBaseline:
    """The system prompt is counted against a fixed probe user message because
    Bedrock refuses an empty conversation. The probe's own weight is a
    per-model constant, measured once per model id per process."""

    def setup_method(self):
        clear_probe_baselines()

    def teardown_method(self):
        clear_probe_baselines()

    class ConfiguredModel(FakeModel):
        def __init__(self, model_id, **kw):
            super().__init__(**kw)
            self.config = {"model_id": model_id}

    @pytest.mark.asyncio
    async def test_probe_weight_is_measured_once_per_model_id(self):
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        first = self.ConfiguredModel("us.anthropic.x", system=100, per_msg=10)
        await ContextAttributionHook()._on_before_model_call(_event(FakeAgent(first, messages=list(msgs)), projected=650))
        assert len(first.calls) == 3

        second = self.ConfiguredModel("us.anthropic.x", system=100, per_msg=10)
        second_agent = FakeAgent(second, messages=list(msgs))
        await ContextAttributionHook()._on_before_model_call(_event(second_agent, projected=650))
        # Baseline reused: probe+system and no-tools only.
        assert len(second.calls) == 2
        assert _parts(get_context_breakdown(second_agent))["system"] == 100

        other = self.ConfiguredModel("us.anthropic.y", system=100, per_msg=10)
        await ContextAttributionHook()._on_before_model_call(_event(FakeAgent(other, messages=list(msgs)), projected=650))
        assert len(other.calls) == 3

    @pytest.mark.asyncio
    async def test_system_partition_is_the_difference_not_the_probe(self):
        # A probe that weighs 24 tokens on its own must not leak into system.
        model = self.ConfiguredModel("m", system=7, per_msg=24)
        agent = FakeAgent(model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
        await ContextAttributionHook()._on_before_model_call(_event(agent, projected=600))
        assert _parts(get_context_breakdown(agent))["system"] == 7

    @pytest.mark.asyncio
    async def test_a_model_without_a_config_counts_the_probe_each_time(self):
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        a, b = FakeModel(), FakeModel()
        await ContextAttributionHook()._on_before_model_call(_event(FakeAgent(a, messages=list(msgs)), projected=650))
        await ContextAttributionHook()._on_before_model_call(_event(FakeAgent(b, messages=list(msgs)), projected=650))
        assert len(a.calls) == len(b.calls) == 3
