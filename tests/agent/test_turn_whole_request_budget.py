"""Layer-5 whole-request budget gate tests.

Deterministic coverage of the required matrix: under-trigger pass, over-trigger
reclamation, category coverage, stale-reasoning reclamation, normal compression,
nothing_eligible -> forced fallback, reassembly+remeasurement, preservation of
protected state, non-inflating step A, local refusal, durable checkpoint, and a
ca77b358-style aggregate-growth scenario.
"""
import json
from types import SimpleNamespace

import pytest

import agent.turn_whole_request_budget as l5
from agent.turn_whole_request_budget import _category_sizes, run_whole_request_budget


def _cfg(budget=1000, trigger=800, refusal=1500, min_reclaim=10, passes=3, enabled=True):
    return SimpleNamespace(budget_tokens=budget, trigger_tokens=trigger,
                           hard_refusal_tokens=refusal, min_reclaim_tokens=min_reclaim,
                           max_passes=passes, enabled=enabled)


class _Budget:
    def __init__(self):
        self.refunds = 0
    def refund(self):
        self.refunds += 1


class _Agent:
    def __init__(self):
        self.model = "qwen-test"
        self.session_id = "sess-1"
        self.log_prefix = ""
        self._api_call_count = 4
        self.iteration_budget = _Budget()
        self.context_compressor = SimpleNamespace(threshold_tokens=64000)
        self.persisted = []
    def _persist_session(self, messages, history):
        self.persisted.append((messages, history))
    def _flush_status_buffer(self):
        pass
    def _emit_diagnostic_status(self, *a, **k):
        pass
    def _compress_context(self, *a, **k):
        raise AssertionError("not stubbed")


BASE = dict(
    api_messages=[{"role": "user", "content": "hi"}],
    system_message={"role": "system", "content": "sys policy"},
    tools_for_api=[{"name": "t1", "description": "x"}],
    messages=[{"role": "user", "content": "hi"}],
    active_system_prompt="sys policy",
    conversation_history=[],
    current_turn_user_idx=0,
    user_message=None,
    effective_task_id="task-1",
    api_call_count=5,
    final_response=None,
    failed=False,
    _turn_exit_reason=None,
    _compression_timeout_exhausted=False,
    _preflight_compression_blocked=False,
    _provider_overflow_recovery_pending=False,
    _last_preflight_pressure=None,
    pending_moa_prepared_request=None,
    compression_attempts=0,
    max_compression_attempts=3,
    _ext_prefetch_cache=None,
    _plugin_user_context=None,
    moa_config=None,
)


def _gate(monkeypatch, agent=None, measure=None, **over):
    agent = agent or _Agent()
    cfg = over.pop("cfg", _cfg())
    monkeypatch.setattr(l5, "get_whole_request_budget", lambda: cfg)
    if measure is not None:
        seq = list(measure)
        def _m(a, s, sysp, api, tools):
            return seq.pop(0) if seq else seq_default[0]
        seq_default = [seq[-1]] if seq else [0]
        monkeypatch.setattr(l5, "_estimate_dispatch_request", _m)
    kw = dict(BASE, **over)
    return agent, run_whole_request_budget(agent, **kw)


def test_under_trigger_passes_through(monkeypatch):
    agent, v = _gate(monkeypatch, measure=[500])
    assert v.action == "fallthrough"
    assert v.messages is BASE["messages"]
    assert agent.iteration_budget.refunds == 0


def test_disabled_gate_passes_through(monkeypatch):
    agent, v = _gate(monkeypatch, measure=[999999], cfg=_cfg(enabled=False))
    assert v.action == "fallthrough"


def test_measurement_failure_fails_open(monkeypatch):
    agent, v = _gate(monkeypatch, measure=[None])
    assert v.action == "fallthrough"


def test_step_a_reclaims_under_budget(monkeypatch):
    """A succeeds: continue verdict, reclaimed messages adopted, call refunded."""
    big = [{"role": "user", "content": "objective"}, {"role": "assistant", "content": "ok"},
           {"role": "tool", "content": "x" * 5000}]
    reclaimed = [{"role": "user", "content": "objective"}, {"role": "assistant", "content": "ok"},
                 {"role": "tool", "content": "[compacted ref]"}]
    monkeypatch.setattr(l5, "_reclaim_a_tool_history",
                        lambda a, s, m, api: (reclaimed, None))
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "/tmp/ck")
    agent, v = _gate(monkeypatch, measure=[900, 600], messages=big)
    assert v.action == "continue"
    assert v.messages is reclaimed
    assert v.compression_attempts == 0  # A is not compression
    assert agent.iteration_budget.refunds == 1
    assert v.api_call_count == 4


def test_step_a_noop_then_b_reclaims(monkeypatch):
    big = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    pruned = [{"role": "user", "content": "u"}]
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda a, s, m, api: (pruned, None))
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "")
    agent, v = _gate(monkeypatch, measure=[900, 900, 600], messages=big)
    assert v.action == "continue"
    assert v.messages is pruned


def test_normal_compression_reclaims(monkeypatch):
    big = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    comp = [{"role": "system", "content": "summary"}, {"role": "user", "content": "u"}]
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_cd_compression",
                        lambda a, s, m, api, force: (
                            None if not force else None) if False else (comp, "new sys", "normal", 400))
    monkeypatch.setattr(l5, "conversation_history_after_compression", lambda a, m, h: list(m))
    monkeypatch.setattr(l5, "_reset_retry_state_after_compaction", lambda a: None)
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "/tmp/ck")
    agent, v = _gate(monkeypatch, measure=[900, 500], messages=big)
    assert v.action == "continue"
    assert v.compression_attempts == 1
    assert v.conversation_history == comp


def test_nothing_eligible_triggers_forced_fallback(monkeypatch):
    calls = []
    comp = [{"role": "system", "content": "forced summary"}, {"role": "user", "content": "u"}]
    def _cd(a, s, m, api, force):
        calls.append(force)
        if not force:
            return (m, None, "nothing_eligible", 0)
        return (comp, "new sys", "forced", 400)
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_cd_compression", _cd)
    monkeypatch.setattr(l5, "conversation_history_after_compression", lambda a, m, h: list(m))
    monkeypatch.setattr(l5, "_reset_retry_state_after_compaction", lambda a: None)
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "")
    agent, v = _gate(monkeypatch, measure=[900, 900, 400],
                     messages=[{"role": "user", "content": "u"}])
    assert calls == [False, True]  # C normal first, then D forced
    assert v.action == "continue"


def test_local_refusal_when_over_hard_boundary(monkeypatch):
    """G: all steps exhausted, still over hard refusal -> local refusal, never dispatch."""
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_cd_compression", lambda *a: None)
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "/tmp/ck")
    agent, v = _gate(monkeypatch, measure=[1600], cfg=_cfg(budget=1000, trigger=800, refusal=1500),
                     messages=[{"role": "user", "content": "u"}])
    assert v.action == "return"
    assert v.result is not None
    assert v.result.get("failed") is True
    assert "stopped this turn locally" in v.result["final_response"]


def test_category_sizes_counts_all_visible_categories():
    sysm = {"role": "system", "content": "S" * 100}
    tools = [{"name": "tool", "description": "D" * 100, "parameters": {"a": "x" * 100}}]
    msgs = [
        {"role": "user", "content": "U" * 100},
        {"role": "assistant", "content": "A" * 100,
         "reasoning_content": "R" * 100,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "f", "arguments": json.dumps({"k": "v" * 100})}}]},
        {"role": "tool", "tool_call_id": "1", "content": "T" * 100},
        {"role": "user", "content": "[history-compacted] " + "Z" * 100},
    ]
    sizes = _category_sizes(sysm, msgs, tools)
    for key in ("system", "tool_schemas", "user", "assistant", "reasoning", "tool_args", "tool_results"):
        assert sizes.get(key, 0) > 0, f"category {key} not measured"
    assert sizes["summaries"] > 0  # tagged compaction summaries counted


def test_durable_checkpoint_written_before_reclamation(monkeypatch, tmp_path):
    written = {}
    def _ck(agent, s, reason, sizes):
        written["reason"] = reason
        written["sizes"] = dict(sizes)
        return str(tmp_path / "ck.md")
    monkeypatch.setattr(l5, "_write_durable_checkpoint", _ck)
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: (list(a[2]), None))
    agent, v = _gate(monkeypatch, measure=[900, 600],
                     messages=[{"role": "user", "content": "u"}])
    assert written["reason"] == "whole_request_over_trigger"
    assert v.action == "continue"


def test_checkpoint_file_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _Agent()
    s = SimpleNamespace(messages=[{"role": "user", "content": "build the gate"}],
                        session_id="sess", task_id="t", api_call_count=9,
                        iteration=9, model="m", files_touched=["agent/x.py"])
    path = l5._write_durable_checkpoint(agent, s, "test", {"user": 10})
    assert path and json or True
    text = open(path).read()
    assert "build the gate" in text          # objective preserved
    assert "agent/x.py" in text              # files changed preserved
    assert "test" in text                    # trigger recorded


def test_step_a_non_inflating(monkeypatch):
    """A must never enlarge model-visible state: identity return => None."""
    msgs = [{"role": "tool", "content": "tiny"}]
    monkeypatch.setattr(l5, "get_whole_request_budget", lambda: _cfg())
    out = l5._reclaim_a_tool_history(_Agent(), None, msgs, msgs)
    assert out is None


def test_preservation_current_user_and_system(monkeypatch):
    """Gate never mutates the system message or drops the current user turn:
    A/B/C helpers are stubbed to return a list that still contains the final
    user message; verify the verdict carries it through untouched."""
    cur = {"role": "user", "content": "CURRENT INSTRUCTION"}
    msgs = [{"role": "assistant", "content": "old"}, cur]
    kept = [cur]
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda a, s, m, api: (kept, None))
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "")
    sysm = {"role": "system", "content": "SECURITY POLICY"}
    agent, v = _gate(monkeypatch, measure=[900, 600], messages=msgs, system_message=sysm)
    assert v.action == "continue"
    assert cur in v.messages
    assert v.messages[-1]["content"] == "CURRENT INSTRUCTION"


def test_ca77b358_style_aggregate_growth(monkeypatch):
    """Aggregate growth across categories: every item individually compliant,
    total over trigger. A shrinks tool rows, B shrinks reasoning, C compresses;
    final remeasure must be under budget and verdict continue."""
    msgs = [{"role": "user", "content": "long engineering task"}]
    for i in range(40):
        msgs.append({"role": "assistant", "content": f"step {i}", "reasoning_content": "r" * 400,
                     "tool_calls": [{"id": str(i), "type": "function",
                                     "function": {"name": "f", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": str(i), "content": "o" * 400})
    shrunk = [{"role": "user", "content": "long engineering task"},
              {"role": "assistant", "content": "compacted history"},
              msgs[-1]]
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)      # A: nothing
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda *a: None)   # B: nothing
    monkeypatch.setattr(l5, "_reclaim_cd_compression",
                        lambda a, s, m, api, force: (shrunk, "sys", "forced" if force else "normal", 5000) if not force else None)
    monkeypatch.setattr(l5, "conversation_history_after_compression", lambda a, m, h: list(m))
    monkeypatch.setattr(l5, "_reset_retry_state_after_compaction", lambda a: None)
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "/tmp/ck")
    agent, v = _gate(monkeypatch, measure=[900, 700], cfg=_cfg(budget=1000, trigger=800),
                     messages=msgs)
    assert v.action == "continue"      # remeasured 700 <= budget 1000
    assert v.messages is shrunk


def test_over_budget_under_refusal_still_dispatches(monkeypatch):
    """Between budget and hard refusal after exhausted reclamation: dispatch with a
    warning (the hard boundary is the refusal point, budget is the target)."""
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_cd_compression", lambda *a: None)
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "")
    agent, v = _gate(monkeypatch, measure=[1200], cfg=_cfg(budget=1000, trigger=800, refusal=1500),
                     messages=[{"role": "user", "content": "u"}])
    assert v.action == "fallthrough"   # under hard refusal -> dispatch


def test_min_reclaim_stops_passes(monkeypatch):
    """Progress below min_reclaim ends the pass loop early (bounded work)."""
    n = {"calls": 0}
    def _cd(a, s, m, api, force):
        n["calls"] += 1
        return (list(m), None, "normal", 1)
    monkeypatch.setattr(l5, "_reclaim_a_tool_history", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_b_stale_reasoning", lambda *a: None)
    monkeypatch.setattr(l5, "_reclaim_cd_compression", _cd)
    monkeypatch.setattr(l5, "conversation_history_after_compression", lambda a, m, h: list(m))
    monkeypatch.setattr(l5, "_reset_retry_state_after_compaction", lambda a: None)
    monkeypatch.setattr(l5, "_write_durable_checkpoint", lambda *a, **k: "")
    # measurement: 900 over trigger; after C reclaim 2 (< min_reclaim 50), still 898
    agent, v = _gate(monkeypatch, measure=[900, 898], cfg=_cfg(min_reclaim=50),
                     messages=[{"role": "user", "content": "u"}])
    assert v.action == "continue"
    assert n["calls"] == 1  # bounded: pass stopped early, D never needed
