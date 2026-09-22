"""Hook that computes a per-turn context-token attribution breakdown.

Splits the authoritative projected input-token count (Bedrock-native via
``CountTokensBedrockModel``) into ``system`` / ``tools`` / ``messages``
partitions and stashes the result on the agent. The stream coordinator reads
it via :func:`get_context_breakdown` and attaches it to the turn's final
``metadata`` SSE event as ``contextBreakdown`` — answering "what is filling the
context window?" without an aggregate-only guess.

Decomposition (convention validated live against Bedrock CountTokens):

- ``systemTokens`` = ``count(system only)``
- ``toolTokens``   = ``full - count(system + messages, no tools)`` — the tool
  schemas **plus** the tool-use scaffolding Bedrock injects only when tools and
  a conversation coexist (~400 tokens). Folded into Tools by design: it is the
  true marginal cost of having tools enabled. An empty-messages baseline would
  miss the scaffolding and mis-attribute it to messages.
- ``messageTokens`` = ``full - systemTokens - toolTokens`` (residual; grows with
  the conversation, scaffolding-free). Partitions sum to ``full`` by
  construction.

``systemTokens`` / ``toolTokens`` are stable across a session (the tool
overhead is constant as the conversation grows — verified), so they are
computed once per agent at cold start (two extra CountTokens calls, plus a
once-per-model probe baseline — see ``_probe_baseline``) and cached; every
turn afterward is pure arithmetic against the free, authoritative
``projected_input_tokens``. ``count(system only)`` is really
``count(probe + system) - count(probe)``: Bedrock refuses an empty message
list, so a bare system count silently degraded to the chars/4 heuristic.

**Why the split is not computed while an attachment is in context.**
``toolTokens`` is a *residual* between two independently sourced numbers —
``full`` (Strands' projection for the upcoming request) and ``no_tools`` (our own
CountTokens call) — so any disagreement between them about how a content block
is counted lands wholly in it. Bedrock understands a PDF page as an image *and*
a text layer; when the two sources do not agree on that, the document's entire
weight is attributed to tools. Measured on dev 2026-09-16 (session
``61de2256``): a call reported ``toolTokens`` of **106,756** where the session's
real tools prefix was **12,516** — a difference of 94,240 against a document
measured at ~94,485, i.e. the whole document. The split is therefore skipped on
any turn whose context carries inline document or image bytes, and taken on a
later clean turn instead. An absent ``prefixTokens`` reads "not tracked" (the
ledger's convention); a wrong one silently corrupts every share computed from
it.

Best-effort: any failure is swallowed so context attribution can never break a
model call. For non-Bedrock models ``count_tokens`` falls back to a heuristic,
so the numbers are approximate there.
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)

# Stashed on the per-session Strands agent instance.
_SPLIT_ATTR = "_context_attribution_split"          # cached stable {systemTokens, toolTokens}
_BREAKDOWN_ATTR = "_context_attribution_breakdown"  # latest per-turn breakdown dict

# Process-level memo of the stable split, keyed by *session and configuration*
# rather than by ``Agent`` instance. The instance attribute above is enough
# only while one Agent serves a session for its whole life; it does not, for
# every session whose injected tools keep it out of the agent cache (the
# spreadsheet-analysis family — see `apis/shared/tools/injected.py`), for
# `@`-mention turns, and for any Memory-Space binding. Those rebuild the Agent
# every turn and, without this memo, paid the two cold-start CountTokens calls
# every turn as well — on top of the SDK's own pre-call count, which is the
# "counted three times per turn" of docs/specs/load-test-assessment-2026-09.md
# §1 fix 1. The key carries a digest of the system prompt and of the full tool
# specs, so any configuration change that would move the split misses cleanly.
_SPLIT_MEMO_MAX = 512
_split_memo: "OrderedDict[Tuple[str, str, str], Dict[str, int]]" = OrderedDict()
_split_memo_lock = threading.Lock()


def clear_split_memo() -> None:
    """Drop every memoised split (tests, and any admin reset path)."""
    with _split_memo_lock:
        _split_memo.clear()


def _digest(payload: Any) -> str:
    try:
        raw = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = repr(payload)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _memo_key(session_id: str, agent: Any) -> Optional[Tuple[str, str, str]]:
    """``(session, prompt digest, tool-spec digest)`` or None if any part is
    unavailable — in which case the memo is simply not consulted."""
    try:
        prompt_payload = getattr(agent, "_system_prompt_content", None) or getattr(agent, "system_prompt", None)
        specs = agent.tool_registry.get_all_tool_specs()
    except Exception:  # noqa: BLE001 - never let key construction break a turn
        return None
    return (session_id, _digest(prompt_payload), _digest(specs))


def _memo_get(key: Tuple[str, str, str]) -> Optional[Dict[str, int]]:
    with _split_memo_lock:
        split = _split_memo.get(key)
        if split is not None:
            _split_memo.move_to_end(key)
        return dict(split) if split is not None else None


def _memo_put(key: Tuple[str, str, str], split: Dict[str, int]) -> None:
    with _split_memo_lock:
        _split_memo[key] = dict(split)
        _split_memo.move_to_end(key)
        while len(_split_memo) > _SPLIT_MEMO_MAX:
            _split_memo.popitem(last=False)


# The fixed user message the system prompt is counted against. Its own weight
# (message scaffolding + the two-letter text) is a per-model constant, so it is
# measured once per model id per process and subtracted. Keep it short and
# never change it casually: a different probe changes every systemTokens
# figure that follows, so the ledger's shares stop being comparable across the
# deploy.
_PROBE_MESSAGES = [{"role": "user", "content": [{"text": "hi"}]}]
_probe_baselines: Dict[str, int] = {}
_probe_lock = threading.Lock()


def clear_probe_baselines() -> None:
    """Drop the per-model probe weights (tests)."""
    with _probe_lock:
        _probe_baselines.clear()


def _probe_model_key(model: Any) -> Optional[str]:
    """Memo key for the probe baseline: the model id when the model exposes
    one, else None (count every time — a test double, not a Bedrock model)."""
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        model_id = config.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None


async def _probe_baseline(model: Any) -> int:
    """Token weight of ``_PROBE_MESSAGES`` alone on ``model``, memoised per model id."""
    key = _probe_model_key(model)
    if key is not None:
        with _probe_lock:
            cached = _probe_baselines.get(key)
        if cached is not None:
            return cached
    baseline = int(await model.count_tokens(messages=list(_PROBE_MESSAGES)))
    if key is not None:
        with _probe_lock:
            _probe_baselines[key] = baseline
    return baseline


def _has_inline_attachment(messages: Any) -> bool:
    """Whether any message carries inline ``document`` / ``image`` bytes.

    The condition under which ``toolTokens`` cannot be trusted — see the module
    docstring. Cheap: a walk over content blocks, no decoding."""
    if not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            for key in ("document", "image"):
                payload = block.get(key)
                if isinstance(payload, dict):
                    source = payload.get("source")
                    if isinstance(source, dict) and isinstance(
                        source.get("bytes"), (bytes, bytearray)
                    ):
                        return True
    return False


def get_context_breakdown(agent: Any) -> Optional[dict]:
    """Return the latest context breakdown stashed on ``agent``, or ``None``.

    Used by the stream coordinator to enrich the final ``metadata`` SSE event.
    """
    return getattr(agent, _BREAKDOWN_ATTR, None)


def get_prefix_token_split(agent: Any) -> Optional[Dict[str, int]]:
    """The stable ``{"system": n, "tools": n}`` split for this agent, or ``None``.

    Persisted on each call's cost row (as ``prefixTokens``) so the static
    prefix a session carries — and which part of it is tool schemas — is a
    stored fact rather than a scan-and-guess. Same numbers the SSE breakdown
    reports; this just reads the cached split without re-counting.
    """
    split = getattr(agent, _SPLIT_ATTR, None)
    if not isinstance(split, dict):
        return None
    try:
        return {
            "system": int(split.get("systemTokens") or 0),
            "tools": int(split.get("toolTokens") or 0),
        }
    except (TypeError, ValueError):
        return None


class ContextAttributionHook(HookProvider):
    """Compute the system / tools / messages token breakdown each turn.

    ``session_id`` enables the process-level split memo (see the module
    comment): a rebuilt Agent for the same session and configuration adopts
    the split its predecessor measured instead of re-counting. Without it the
    split lives on the Agent instance only.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self._session_id = session_id or None

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)

    async def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        try:
            await self._compute(event)
        except Exception as e:  # noqa: BLE001 - attribution must never break a turn
            logger.debug("Context attribution skipped: %s", e)

    async def _compute(self, event: BeforeModelCallEvent) -> None:
        agent = event.agent
        model = agent.model
        system_prompt = getattr(agent, "system_prompt", None)
        system_prompt_content = getattr(agent, "_system_prompt_content", None)
        full = event.projected_input_tokens

        split = getattr(agent, _SPLIT_ATTR, None)
        memo_key = _memo_key(self._session_id, agent) if (split is None and self._session_id) else None
        if split is None and memo_key is not None:
            split = _memo_get(memo_key)
            if split is not None:
                # A predecessor Agent for this session + configuration already
                # measured it; adopt without spending two CountTokens calls.
                setattr(agent, _SPLIT_ATTR, split)
                logger.debug("Context attribution split adopted from session memo")
        if split is None and _has_inline_attachment(agent.messages):
            # Untrustworthy residual (see module docstring) — leave the split
            # uncomputed and try again on a turn without inline bytes.
            logger.debug("Context attribution deferred: inline attachment in context")
            return
        if split is None:
            # Bedrock CountTokens rejects an empty message list ("A
            # conversation must start with a user message" — verified live
            # against dev 2026-09-18), and Strands swallows that into the
            # chars/4 heuristic, so `count(messages=[])` was never the
            # authoritative system count it looked like. Count the system
            # prompt against a fixed probe user message and subtract the
            # probe's own weight, which is a per-model constant measured once
            # per process.
            probe_only = await _probe_baseline(model)
            system_with_probe = await model.count_tokens(
                messages=list(_PROBE_MESSAGES),
                system_prompt=system_prompt,
                system_prompt_content=system_prompt_content,
            )
            system_tokens = max(0, system_with_probe - probe_only)
            # system + the current conversation, WITHOUT tools — so the
            # difference from `full` captures tool schemas + the tool-use
            # scaffolding (present only when tools and messages coexist).
            no_tools = await model.count_tokens(
                messages=agent.messages,
                system_prompt=system_prompt,
                system_prompt_content=system_prompt_content,
            )
            if full is None:
                # projected estimate unavailable — count the full request once
                # so cold start can still establish the split.
                tool_specs = agent.tool_registry.get_all_tool_specs()
                full = await model.count_tokens(
                    messages=agent.messages,
                    tool_specs=tool_specs,
                    system_prompt=system_prompt,
                    system_prompt_content=system_prompt_content,
                )
            split = {
                "systemTokens": system_tokens,
                "toolTokens": max(0, full - no_tools),
            }
            setattr(agent, _SPLIT_ATTR, split)
            if memo_key is not None:
                _memo_put(memo_key, split)

        if full is None:
            # No authoritative total this turn — can't place the messages
            # partition. Leave the previous breakdown (if any) untouched.
            return

        message_tokens = max(0, full - split["systemTokens"] - split["toolTokens"])
        breakdown = {
            "total": full,
            "partitions": [
                {"key": "system", "label": "System prompt", "tokens": split["systemTokens"]},
                {"key": "tools", "label": "Tools", "tokens": split["toolTokens"]},
                {"key": "messages", "label": "Messages", "tokens": message_tokens},
            ],
        }
        setattr(agent, _BREAKDOWN_ATTR, breakdown)
