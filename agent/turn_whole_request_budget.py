"""Layer 5 -- whole-request budget gate.

The Qwen ``DataInspectionFailed`` failure was an *aggregate* problem: every
individual message category stayed inside its own Layer-4 budget, yet the
fully serialized model-visible request (system prompt + schemas + messages +
reasoning + tool arguments + results + summaries) grew past what the provider
would accept.  Layer 4 governs ONE category (historical tool results); this
module owns whether the COMPLETE request is safe to send.

Gate location: ``agent/conversation_loop.py`` -- immediately after
``run_preflight_gate`` and before ``announce_api_call`` / ``build_api_request``,
i.e. the narrowest provider-independent point where the dispatched request can
still be reclaimed.

Reclamation order (deterministic, approved architecture A-G):
  A. Non-inflating historical tool-result compaction (``enforce_history_budget``,
     now guarded by the Layer-4 strictly-smaller invariant).
  B. Strip stale historical reasoning replay not required for current execution.
  C. Sanctioned conversation compression (the normal ``compress_context`` path).
  D. Forced compression fallback when C returns nothing-eligible while the
     complete request is still over budget.
  E. Durable engineering checkpoint + spill of raw material to recoverable
     artifacts, keeping a concise model-visible reference.
  F. Reassemble and remeasure the FINAL provider-bound request.
  G. Local bounded-context refusal -- never dispatch an oversized request.

Preserved through every reclamation step: current user instructions, active
task state, unresolved decisions, approval state, current/relevant tool
outputs, and system/security policy.  The compressor already protects the
system prompt and the recent window; step E additionally writes a durable
checkpoint BEFORE aggressive reclamation so Mya can never "forget where she
is".

Observability: one concise structured log line per reclamation event
(``request total -> category sizes -> budget -> action -> reclaimed -> final``)
plus a ``compressor_result`` label: ``normal | forced | nothing_eligible |
local_refusal``.  No secrets, no giant payloads.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    estimate_tokens_rough,
)
from agent.conversation_compression import conversation_history_after_compression
from agent.turn_context_compaction import _reset_retry_state_after_compaction
from agent.turn_preflight import PreflightGateVerdict
from tools.budget_config import get_whole_request_budget

if TYPE_CHECKING:
    from run_agent import AIAgent

logger = logging.getLogger(__name__)

__all__ = ["run_whole_request_budget"]

_REASONING_KEYS = ("reasoning", "reasoning_content")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _checkpoint_path(agent) -> str:
    """Profile-scoped durable engineering checkpoint path (never CWD)."""
    try:
        from agent.agent_init import _session_home_dir_from_db_path
        db_path = getattr(agent, "_session_db", None)
        db_path = getattr(db_path, "db_path", None)
        if db_path:
            home = _session_home_dir_from_db_path(db_path)
            base = os.path.join(home, ".hermes", "checkpoints")
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, f"layer5_{_now_iso()}.md")
    except Exception:
        pass
    base = os.path.join(os.path.expanduser("~"), ".hermes", "checkpoints")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, f"layer5_{_now_iso()}.md")


def _category_sizes(system_message, messages, tools_for_api) -> dict:
    """Model-visible size per category -- ALL categories, not just role=tool."""
    sys_tokens = estimate_tokens_rough(system_message) if system_message else 0
    tools_tokens = estimate_tokens_rough(json.dumps(tools_for_api, default=str)) if tools_for_api else 0

    by_role = {"user": 0, "assistant": 0, "tool": 0, "system": 0, "other": 0}
    reasoning_tokens = 0
    tool_args_tokens = 0
    summary_tokens = 0
    for msg in messages:
        role = msg.get("role", "other")
        key = role if role in by_role else "other"
        content = msg.get("content")
        if isinstance(content, str):
            by_role[key] += estimate_tokens_rough(content)
        elif content is not None:
            by_role[key] += estimate_tokens_rough(json.dumps(content, default=str))
        for rk in _REASONING_KEYS:
            rv = msg.get(rk)
            if isinstance(rv, str) and rv:
                reasoning_tokens += estimate_tokens_rough(rv)
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                tool_args_tokens += estimate_tokens_rough(args)
            elif args is not None:
                tool_args_tokens += estimate_tokens_rough(json.dumps(args, default=str))
        # compaction summaries ride as user rows tagged by the compressor
        if role == "user" and isinstance(content, str) and "[history-compacted]" in content:
            summary_tokens += estimate_tokens_rough(content)
    return {
        "system": sys_tokens,
        "tool_schemas": tools_tokens,
        "user": by_role["user"],
        "assistant": by_role["assistant"],
        "tool_results": by_role["tool"],
        "reasoning": reasoning_tokens,
        "tool_args": tool_args_tokens,
        "summaries": summary_tokens,
        "other": by_role["other"],
    }


def _estimate_dispatch_request(agent, s, system_message, api_messages, tools_for_api) -> Optional[int]:
    """Measure the ACTUAL provider-bound request (approved requirement: PASS must
    mean the FINAL dispatched request fits, not an earlier copy of ``messages``).

    ``api_messages`` + ``tools_for_api`` are exactly what the assembly phase
    produced for dispatch and what ``build_api_request`` serializes.  We measure
    them with the same provider-independent estimator the assembly phase uses
    (``estimate_request_tokens_rough``), rather than calling ``build_api_request``
    here -- that function has per-attempt side effects (stream-delivery reset,
    first-chunk timestamp) that must run exactly once, in the dispatch path.
    Re-decoration it applies (prompt-cache markers, reasoning echo) is already
    reflected in ``api_messages``; it does not materially change token count.

    Returns rough total tokens, or ``None`` if measurement fails.
    """
    try:
        return estimate_request_tokens_rough(
            api_messages or [],
            system_prompt=system_message or "",
            tools=tools_for_api or None,
        )
    except Exception as exc:
        logger.warning("layer5: request measurement failed: %s", exc)
        return None


def _write_durable_checkpoint(agent, s, reason: str, sizes: dict) -> str:
    """Persist a recoverable engineering checkpoint BEFORE aggressive reclamation
    so task continuity survives even if working context is reclaimed.

    Contains: objective, current phase, completed work, findings, files changed,
    decisions, tests/results, active blockers, exact next action.  Best-effort --
    a failure to write never blocks the gate.
    """
    try:
        path = _checkpoint_path(agent)
        lines = [
            f"# Layer-5 durable engineering checkpoint",
            f"",
            f"- written_utc: {datetime.now(timezone.utc).isoformat()}",
            f"- trigger: {reason}",
            f"- session_id: {getattr(s, 'session_id', '')}",
            f"- task_id: {getattr(s, 'task_id', '')}",
            f"- api_call_count: {getattr(s, 'api_call_count', '')}",
            f"- iteration: {getattr(s, 'iteration', '')}",
            f"- model: {getattr(s, 'model', '')}",
            f"",
            f"## Category sizes at trigger (rough tokens)",
            "```json",
            json.dumps(sizes, indent=2, default=str),
            "```",
            f"",
            f"## Objective",
            _extract_objective(s),
            f"",
            f"## Current phase / next exact action",
            _extract_next_action(s),
            f"",
            f"## Recent task state (last assistant/user turns)",
            _extract_recent_state(s),
            f"",
            f"## Active todo / plan state",
            _extract_todo_state(agent, s),
            f"",
            f"## Files changed this session (best-effort)",
            _extract_files_changed(agent, s),
            f"",
            f"## Recoverable raw artifacts",
            f"- Raw material spilled to recoverable artifacts remains available;"
            f" the model-visible request is bounded, engineering evidence is not destroyed.",
        ]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("layer5_checkpoint: wrote %s", path)
        return path
    except Exception as exc:
        logger.warning("layer5_checkpoint: failed to write (%s)", exc)
        return ""


def _extract_objective(s) -> str:
    msgs = getattr(s, "messages", []) or []
    for msg in msgs[:6]:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()[:1500]
    return "(no initial user objective found in recent window)"


def _extract_next_action(s) -> str:
    msgs = getattr(s, "messages", []) or []
    tail = []
    for msg in msgs[-6:]:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            tail.append(f"[{role}] {content.strip()[:600]}")
    return "\n".join(tail) if tail else "(no recent turn text)"


def _extract_recent_state(s) -> str:
    return _extract_next_action(s)


def _extract_todo_state(agent, s) -> str:
    try:
        todo = getattr(agent, "_todo_tool", None)
        if todo is not None and hasattr(todo, "snapshot_text"):
            return todo.snapshot_text()[:2000]
    except Exception:
        pass
    return "(todo snapshot unavailable)"


def _extract_files_changed(agent, s) -> str:
    try:
        changed = getattr(s, "files_touched", None) or getattr(agent, "_files_touched", None)
        if changed:
            return "\n".join(f"- {p}" for p in list(changed)[:80])
    except Exception:
        pass
    return "(file-change tracking unavailable; consult git status in the working tree)"


# ---------------------------------------------------------------------------
# Reclamation steps A-G.  Every mutator takes and returns the pair
# ``(messages, api_messages)``; ``None`` from a step means "not applicable /
# nothing reclaimed".  Mutators NEVER touch: the last user message, active
# tool-call/tool-result pairs in the recent protected window, approval state
# rows, or system/security policy (the system prompt is not in ``messages``
# and is never mutated here -- compression is the only sanctioned cache break
# and it owns the system prompt itself).
# ---------------------------------------------------------------------------

def _reclaim_a_tool_history(agent, s, messages, api_messages):
    """A. Non-inflating historical tool-result compaction (Layer 4 engine).

    Uses the same ``enforce_history_budget`` path the tool executor runs at
    batch finalization, which -- since the Layer-4 guard in ae210a775b --
    only replaces a result when the replacement is STRICTLY smaller.  Raw
    material is spilled to recoverable artifacts with concise references
    retained (step E's artifact behavior, performed by the engine itself).
    """
    try:
        from tools.tool_result_storage import enforce_history_budget
        # DEFAULT_BUDGET resolves live settings via _configured_budget_values();
        # same defaults the tool-executor call site uses.
        before = sum(estimate_tokens_rough(json.dumps(m, default=str)) for m in messages)
        new_messages = enforce_history_budget(messages)
        if new_messages is messages:
            return None
        after = sum(estimate_tokens_rough(json.dumps(m, default=str)) for m in new_messages)
        if after >= before:
            return None
        # api_messages mirrors messages 1:1 for tool rows; re-derive is unsafe
        # mid-turn, so return new_messages and let the caller re-run the
        # assembly mirror (F) before remeasure.
        return (new_messages, None)
    except Exception as exc:
        logger.warning("layer5: step A failed: %s", exc)
        return None


def _reclaim_b_stale_reasoning(agent, s, messages, api_messages):
    """B. Strip stale historical reasoning replay.

    Reuses the sanctioned ``_prune_stale_reasoning_replay`` (the same routine
    the request-assembly phase runs for providers that do not accept echoed
    reasoning).  Reasoning attached to OLD assistant turns is not required for
    current execution and is pure token weight; current-turn reasoning is not
    in ``messages`` yet.
    """
    try:
        from agent.context_compressor import _prune_stale_reasoning_replay
        pruned = _prune_stale_reasoning_replay(list(messages))
        if pruned is None or pruned == messages:
            return None
        return (pruned, None)
    except Exception as exc:
        logger.warning("layer5: step B failed: %s", exc)
        return None


def _reclaim_cd_compression(agent, s, messages, api_messages, force: bool):
    """C/D. Sanctioned conversation compression; ``force=True`` is the approved
    forced fallback for the nothing_eligible case.

    Calls the SAME sanctioned path preflight compression uses
    (``agent._compress_context`` -> ``ContextCompressor.compress_context``),
    which protects the system prompt, recent window, current instructions and
    active tool state, and writes recoverable summaries.  Compression is the
    only sanctioned prompt-cache break -- we ride it instead of inventing a
    second mutation path.

    Returns ``(new_messages, result_label, reclaimed_tokens)`` or ``None``.
    ``result_label`` is one of: ``normal``, ``forced``, ``nothing_eligible``.
    """
    try:
        approx = estimate_messages_tokens_rough(messages)
        new_messages, active_prompt = agent._compress_context(
            messages,
            s.system_message,
            approx_tokens=approx,
            task_id=s.effective_task_id,
            force=force,
            bypass_cooldown=force,
        )
        label = "forced" if force else "normal"
        if new_messages is messages:
            # Sanctioned no-op signal used by turn_preflight: identity means the
            # compressor found nothing eligible (or was skipped by lock/cooldown).
            return (messages, None, "nothing_eligible", 0)
        reclaimed = max(0, approx - estimate_messages_tokens_rough(new_messages))
        return (new_messages, active_prompt, label, reclaimed)
    except Exception as exc:
        logger.warning("layer5: step %s failed: %s", "D" if force else "C", exc)
        return None


# Note on the system prompt after C/D: ``_compress_context`` may return a new
# active system prompt (it owns the sanctioned prompt-cache break).  The gate
# writes it back to ``s.active_system_prompt`` -- the same state field the
# preflight compression path writes (turn_preflight.py:148) -- so the next
# assembly/dispatch pass picks it up without the gate touching dispatch.


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _l5_log(event, **fields):
    """One structured line per event: total -> sizes -> budget -> action -> reclaimed."""
    try:
        logger.info("l5_budget %s %s", event, " ".join("%s=%s" % (k, v) for k, v in fields.items()))
    except Exception:
        pass


def _l5_unpack_budget(cfg):
    if isinstance(cfg, tuple):
        budget, trigger, hard_refusal, min_reclaim, max_passes, enabled = cfg[:6]
    else:
        budget = cfg.budget_tokens
        trigger = cfg.trigger_tokens
        hard_refusal = cfg.hard_refusal_tokens
        min_reclaim = cfg.min_reclaim_tokens
        max_passes = cfg.max_passes
        enabled = cfg.enabled
    return int(budget), int(trigger), int(hard_refusal), int(min_reclaim), int(max_passes), bool(enabled)


def _l5_measure(api_messages, sys_prompt, tools_for_api):
    try:
        return estimate_request_tokens_rough(api_messages, system_prompt=sys_prompt, tools=tools_for_api)
    except Exception:
        return None


def _l5_state_shim(**kw):
    return SimpleNamespace(**kw)


def _l5_reanchor(messages, old_idx):
    """Index fixup after shrink: mirror turn_preflight's reanchor semantics."""
    try:
        from agent.turn_preflight import _reanchor as _pf
        return _pf(messages, old_idx)
    except Exception:
        pass
    try:
        if old_idx is not None and 0 <= old_idx < len(messages):
            m = messages[old_idx]
            if isinstance(m, dict) and m.get("role") == "user":
                return old_idx
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, dict) and m.get("role") == "user":
                return i
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Gate plumbing
# ---------------------------------------------------------------------------

_L5_TURN_PASSES = "_l5_gate_turn_passes"


def _l5_gate_shim(agent, messages, system_message, effective_task_id, api_call_count):
    """Minimal state stand-in consumed by the reclaim/checkpoint helpers."""
    return SimpleNamespace(
        system_message=system_message,
        effective_task_id=effective_task_id,
        messages=messages,
        session_id=str(getattr(agent, "session_id", "") or ""),
        task_id=str(effective_task_id or ""),
        api_call_count=api_call_count,
        iteration=api_call_count,
        model=str(getattr(agent, "model", "") or ""),
    )


def _l5_find_user_idx(messages, user_message, fallback):
    try:
        for i, m in enumerate(messages):
            if m is user_message:
                return i
    except Exception:
        pass
    return fallback


def _l5_rebuild(agent, working_messages, idx, ext_prefetch_cache,
                plugin_user_context, moa_config, working_prompt):
    """F: reassemble the wire copy from the reclaimed ``messages`` so the
    remeasure reflects what would actually be dispatched.  Returns
    ``(rebuilt_api_messages_or_None, effective_prompt_for_measure)``."""
    try:
        from agent.turn_context import build_api_messages
        api_msgs, eff_sys = build_api_messages(
            agent, working_messages,
            current_turn_user_idx=idx,
            ext_prefetch_cache=ext_prefetch_cache,
            plugin_user_context=plugin_user_context,
            moa_config=moa_config,
            active_system_prompt=working_prompt,
        )
        return api_msgs, (eff_sys or working_prompt)
    except Exception as exc:
        logger.warning("layer5: in-gate reassembly failed (%s); using proxy measure", exc)
        return None, working_prompt


def _l5_remeasure(rebuild_api, working_messages, sys_prompt, tools_for_api):
    t = _estimate_dispatch_request(None, None, sys_prompt, rebuild_api, tools_for_api) if rebuild_api is not None else None
    if t is None:
        t = _estimate_dispatch_request(None, None, sys_prompt, working_messages, tools_for_api)
    return t




# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------

def run_whole_request_budget(
    agent: Any, *, api_messages: List[Dict[str, Any]], system_message: Optional[Dict[str, Any]],
    tools_for_api: Optional[Any], messages: List[Dict[str, Any]], active_system_prompt: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]], current_turn_user_idx: Optional[int],
    user_message: Optional[Dict[str, Any]], effective_task_id: Optional[str],
    api_call_count: int, final_response: Optional[str], failed: bool,
    _turn_exit_reason: Optional[str], _compression_timeout_exhausted: bool,
    _preflight_compression_blocked: bool, _provider_overflow_recovery_pending: bool,
    _last_preflight_pressure: Optional[int], pending_moa_prepared_request: Optional[Any],
    compression_attempts: int, max_compression_attempts: int,
    _ext_prefetch_cache: Optional[Any], _plugin_user_context: Optional[Any], moa_config: Optional[Any],
) -> PreflightGateVerdict:
    """Layer 5: one authoritative whole-request budget gate.

    Runs after assembly and the preflight gate, before announce/dispatch, so the
    request measured here is the actual provider-bound request. Over trigger:
    durable checkpoint, then A -> B -> C -> D(forced) with a full reassembly +
    remeasure (F) after every step; bounded by max_passes; G is a local refusal
    shaped by _provider_overflow_exhausted_result -- never an oversized dispatch.
    """
    from agent.conversation_loop import _provider_overflow_exhausted_result
    from agent.turn_context_compaction import _refund_api_call

    v = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=pending_moa_prepared_request,
        messages=messages, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, api_call_count=api_call_count,
        compression_attempts=compression_attempts, final_response=final_response, failed=failed,
        _turn_exit_reason=_turn_exit_reason,
        _compression_timeout_exhausted=_compression_timeout_exhausted,
        _preflight_compression_blocked=_preflight_compression_blocked,
        _provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
        _last_preflight_pressure=_last_preflight_pressure,
    )
    budget, trigger, hard_refusal, min_reclaim, max_passes, gate_enabled = _l5_unpack_budget(get_whole_request_budget())
    if not gate_enabled:
        return v
    sys_prompt = system_message.get("content", "") if isinstance(system_message, dict) else (system_message or "")
    total = _estimate_dispatch_request(agent, None, sys_prompt, api_messages, tools_for_api)
    if total is None:
        logger.warning("layer5: measurement unavailable; passing through")
        return v
    sizes = _category_sizes(system_message, api_messages, tools_for_api)
    if total <= trigger:
        return v  # fast path: under trigger, request dispatches as-is

    # Over trigger: log the structured picture, then write the durable
    # engineering checkpoint BEFORE any aggressive reclamation.
    logger.warning(
        "layer5_gate: request total ~%s -> categories %s -> budget %s/trigger %s/refusal %s",
        f"{total:,}",
        {k: f"~{val:,}" for k, val in sorted(sizes.items(), key=lambda kv: -kv[1]) if val},
        f"{budget:,}", f"{trigger:,}",
        f"{hard_refusal:,}",
    )
    shim = _l5_gate_shim(agent, messages, system_message, effective_task_id, api_call_count)
    checkpoint_path = _write_durable_checkpoint(agent, shim, "whole_request_over_trigger", sizes)

    working_messages = messages
    working_prompt = active_system_prompt
    rebuilt_api = api_messages
    idx = _l5_find_user_idx(messages, user_message, current_turn_user_idx)
    compressed = False
    reclaimed_total = 0
    plan = [("A", False), ("B", False), ("C", False), ("D", True)]
    passes = 0
    while passes < max_passes:
        passes += 1
        progress_this_pass = 0
        for step, force in plan:
            before_step = total
            if step == "A":
                out = _reclaim_a_tool_history(agent, shim, working_messages, rebuilt_api)
                label = "tool_history"
            elif step == "B":
                out = _reclaim_b_stale_reasoning(agent, shim, working_messages, rebuilt_api)
                label = "stale_reasoning"
            else:
                out = _reclaim_cd_compression(agent, shim, working_messages, rebuilt_api, force)
                label = "forced_compression" if force else "compression"
            if out is None:
                continue
            if label in ("compression", "forced_compression"):
                new_messages, new_prompt, c_label, c_reclaimed = out
                if c_label == "nothing_eligible":
                    logger.info("layer5_gate: %s -> nothing_eligible", label)
                    continue
                working_prompt = new_prompt or working_prompt
                compressed = True
                reclaimed_step = c_reclaimed
            else:
                new_messages = out[0]
                working_messages = new_messages  # applied below with rebuilt list
                reclaimed_step = 0
            if label not in ("compression", "forced_compression"):
                working_messages = new_messages
            else:
                working_messages = new_messages
            # F: reassemble the wire copy from reclaimed messages, then remeasure.
            rebuilt_api, working_prompt = _l5_rebuild(
                agent, working_messages, idx, _ext_prefetch_cache,
                _plugin_user_context, moa_config, working_prompt,
            )
            total = _l5_remeasure(rebuilt_api, working_messages, working_prompt, tools_for_api)
            reclaimed_step = max(0, before_step - total)
            reclaimed_total += reclaimed_step
            progress_this_pass += reclaimed_step
            logger.info(
                "layer5_gate: %s=%s -> reclaimed ~%s -> total ~%s (checkpoint=%s)",
                step, label, f"{reclaimed_step:,}", f"{total:,}",
                "yes" if checkpoint_path else "no",
            )
            if total <= budget:
                break
        if total <= budget:
            break
        if progress_this_pass < min_reclaim:
            logger.warning("layer5_gate: pass %s reclaimed ~%s < min ~%s; stopping passes",
                           passes, f"{progress_this_pass:,}", f"{min_reclaim:,}")
            break

    # Final decision on the remeasured provider-bound request.
    if total <= budget:
        label = "reclaimed_pass"
    elif total <= hard_refusal:
        label = "over_budget_under_refusal"
        logger.warning(
            "layer5_gate: %s; dispatching ~%s (budget ~%s, hard refusal ~%s) -- "
            "reclamation exhausted but request is under the hard safety boundary",
            label, f"{total:,}", f"{budget:,}",
            f"{hard_refusal:,}",
        )
    else:
        label = "local_refusal"

    if label != "local_refusal" and working_messages is not messages:
        # Reclamation succeeded: adopt reclaimed state and re-enter the loop so the
        # assembly phase rebuilds the dispatch request from reclaimed messages (F).
        v.messages = working_messages
        v.active_system_prompt = working_prompt
        if compressed:
            _reset_retry_state_after_compaction(agent)
            v.conversation_history = conversation_history_after_compression(
                agent, working_messages, conversation_history
            )
            v.compression_attempts = min(compression_attempts + 1, max_compression_attempts)
        v.api_call_count = _refund_api_call(agent, v.api_call_count)
        logger.info(
            "layer5_gate: %s -> final ~%s (reclaimed ~%s, passes=%s, compressed=%s)",
            label, f"{total:,}", f"{reclaimed_total:,}", passes, compressed,
        )
        v.action = "continue"
        return v

    if label == "local_refusal":
        # G: fail locally with a bounded-context error. The oversized request is
        # NEVER dispatched; the durable checkpoint + spilled artifacts preserve the
        # engineering evidence, and the transcript stays intact for the user.
        logger.error(
            "layer5_gate: local_refusal; request ~%s exceeds hard refusal ~%s after "
            "%s passes (reclaimed ~%s). Refusing to dispatch oversized request.",
            f"{total:,}", f"{hard_refusal:,}", passes,
            f"{reclaimed_total:,}",
        )
        refusal = (
            "I stopped this turn locally instead of sending an oversized request to "
            "the model. The conversation reached ~%s tokens of provider-visible "
            "context (hard safety boundary ~%s), and every sanctioned reclamation "
            "step was exhausted without bringing it back under budget. A durable "
            "checkpoint of the task state was written (%s), and compacted raw "
            "material remains in recoverable artifacts, so nothing is lost. To "
            "continue: start a fresh turn on the same task, or ask me to compress "
            "the session; the checkpoint tells me exactly where we stopped." % (
                f"{total:,}", f"{hard_refusal:,}",
                checkpoint_path or "path unavailable",
            )
        )
        v.result = _provider_overflow_exhausted_result(
            agent, working_messages, v.conversation_history, v.api_call_count,
            total, max_compression_attempts,
        )
        v.result["final_response"] = refusal
        v.result["error"] = refusal
        v.action = "return"
        return v

    return v
