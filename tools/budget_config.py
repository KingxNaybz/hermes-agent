"""Configurable budgets for model-visible tool results.

Raw tool output is never discarded: :mod:`tools.tool_result_storage` spills output
that exceeds these limits to an artifact and gives the model a compact preview plus
artifact path.  Limits here govern only the representation injected into context.
"""

from dataclasses import dataclass, field
from typing import Dict

from agent.model_metadata import CHARS_PER_TOKEN

# Never overridden; read_file=inf prevents infinite persist->read->persist loops.
PINNED_THRESHOLDS: Dict[str, float] = {"read_file": float("inf")}

DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500

# Layer 4 — accumulated-session ceiling.  A long autonomous/cron session can pass every
# per-result and per-turn check and still accumulate tens of thousands of chars of
# inspection-sensitive tool output across many model calls; Qwen rejected one such request
# with InternalError.Algo.DataInspectionFailed after 32 successful calls (46 tool results,
# ~95K chars, zero individual oversize hits).  Oldest results are compacted to artifact
# references first; the most recent results stay inline as working context.
DEFAULT_HISTORY_BUDGET_CHARS: int = 60_000
DEFAULT_HISTORY_RECENT_PROTECT: int = 6

# Tools whose normal successful responses can contain whole pages, transcripts, search
# histories, or model-generated vision prose.  These are deliberately lower than the
# generic compatibility ceiling: provider content inspection can reject a request based
# on one large tool result even while the token window still has ample room.
DEFAULT_TOOL_RESULT_LIMITS: Dict[str, int] = {
    "vision_analyze": 12_000,
    "browser_exec": 20_000,
    "web_extract": 20_000,
    "web_search": 16_000,
    "x_search": 16_000,
    "session_search": 16_000,
    "hindsight_recall": 16_000,
    "hindsight_reflect": 16_000,
    "delegate_task": 20_000,
    "execute_code": 24_000,
    "terminal": 32_000,
    "search_files": 24_000,
}

# MCP servers frequently return unpaginated payloads.  Keep raw output as an artifact,
# but do not carry more than this in every subsequent inference request.
DEFAULT_MCP_RESULT_SIZE_CHARS: int = 20_000
MCP_TOOL_PREFIX: str = "mcp_"


def _positive_int(value, default: int) -> int:
    try:
        value = int(value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _configured_budget_values() -> tuple[int, int, int, int, Dict[str, int], int, int]:
    """Load ``tool_budget`` without making config availability a runtime dependency.

    Supported keys::

        tool_budget:
          default_result_size_chars: 100000
          turn_budget_chars: 200000
          preview_size_chars: 1500
          mcp_result_size_chars: 20000
          history_budget_chars: 60000
          history_recent_protect: 6
          per_tool_result_size_chars:
            vision_analyze: 12000
            browser_exec: 20000

    Invalid entries are ignored.  Per-tool values augment/override the safe defaults.
    """
    default_size = DEFAULT_RESULT_SIZE_CHARS
    turn_budget = DEFAULT_TURN_BUDGET_CHARS
    preview_size = DEFAULT_PREVIEW_SIZE_CHARS
    mcp_size = DEFAULT_MCP_RESULT_SIZE_CHARS
    history_budget = DEFAULT_HISTORY_BUDGET_CHARS
    history_protect = DEFAULT_HISTORY_RECENT_PROTECT
    overrides = dict(DEFAULT_TOOL_RESULT_LIMITS)
    try:
        from hermes_cli.config import load_config_readonly

        data = load_config_readonly()
        block = data.get("tool_budget") if isinstance(data, dict) else None
        if not isinstance(block, dict):
            return default_size, turn_budget, preview_size, mcp_size, overrides, history_budget, history_protect
        default_size = _positive_int(block.get("default_result_size_chars"), default_size)
        turn_budget = _positive_int(block.get("turn_budget_chars"), turn_budget)
        preview_size = _positive_int(block.get("preview_size_chars"), preview_size)
        mcp_size = _positive_int(block.get("mcp_result_size_chars"), mcp_size)
        history_budget = _positive_int(block.get("history_budget_chars"), history_budget)
        history_protect = _positive_int(block.get("history_recent_protect"), history_protect)
        configured = block.get("per_tool_result_size_chars")
        if isinstance(configured, dict):
            for name, value in configured.items():
                if isinstance(name, str) and name.strip():
                    parsed = _positive_int(value, 0)
                    if parsed > 0:
                        overrides[name.strip()] = parsed
    except Exception:
        pass
    return default_size, turn_budget, preview_size, mcp_size, overrides, history_budget, history_protect


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable model-visible tool-result budgets.

    Resolution priority is pinned safety rule, explicit/default per-tool limit,
    MCP limit, registry value, then generic limit.  Every non-pinned result is also
    capped by ``default_result_size`` so a small model's context-scaled budget wins.
    """

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    mcp_result_size: int = DEFAULT_MCP_RESULT_SIZE_CHARS
    tool_overrides: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TOOL_RESULT_LIMITS))
    history_budget: int = DEFAULT_HISTORY_BUDGET_CHARS
    history_recent_protect: int = DEFAULT_HISTORY_RECENT_PROTECT

    def resolve_threshold(self, tool_name: str) -> int | float:
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        if tool_name in self.tool_overrides:
            return min(self.tool_overrides[tool_name], self.default_result_size)
        if tool_name.startswith(MCP_TOOL_PREFIX):
            return min(self.mcp_result_size, self.default_result_size)
        from tools.registry import registry

        registry_value = registry.get_max_result_size(tool_name, default=self.default_result_size)
        if registry_value == float("inf"):
            return registry_value
        return min(registry_value, self.default_result_size)


DEFAULT_BUDGET = BudgetConfig()

_CHARS_PER_TOKEN: int = CHARS_PER_TOKEN
_PER_RESULT_WINDOW_FRACTION: float = 0.15
_PER_TURN_WINDOW_FRACTION: float = 0.30
_PER_HISTORY_WINDOW_FRACTION: float = 0.10
_MIN_RESULT_SIZE_CHARS: int = 8_000
_MIN_TURN_BUDGET_CHARS: int = 16_000
_MIN_HISTORY_BUDGET_CHARS: int = 8_000


def budget_for_context_window(context_length: int | None) -> BudgetConfig:
    """Return configured budgets capped for the active model's context window."""
    default_size, turn_budget, preview_size, mcp_size, overrides, history_budget, history_protect = _configured_budget_values()
    if not context_length or context_length <= 0:
        if (default_size == DEFAULT_RESULT_SIZE_CHARS
                and turn_budget == DEFAULT_TURN_BUDGET_CHARS
                and preview_size == DEFAULT_PREVIEW_SIZE_CHARS
                and mcp_size == DEFAULT_MCP_RESULT_SIZE_CHARS
                and overrides == DEFAULT_TOOL_RESULT_LIMITS
                and history_budget == DEFAULT_HISTORY_BUDGET_CHARS
                and history_protect == DEFAULT_HISTORY_RECENT_PROTECT):
            return DEFAULT_BUDGET
    else:
        window_chars = context_length * _CHARS_PER_TOKEN
        default_size = max(
            _MIN_RESULT_SIZE_CHARS,
            min(int(window_chars * _PER_RESULT_WINDOW_FRACTION), default_size),
        )
        turn_budget = max(
            _MIN_TURN_BUDGET_CHARS,
            min(int(window_chars * _PER_TURN_WINDOW_FRACTION), turn_budget),
        )
        history_budget = max(
            _MIN_HISTORY_BUDGET_CHARS,
            min(int(window_chars * _PER_HISTORY_WINDOW_FRACTION), history_budget),
        )
    return BudgetConfig(
        default_result_size=default_size,
        turn_budget=turn_budget,
        preview_size=preview_size,
        mcp_result_size=mcp_size,
        tool_overrides=overrides,
        history_budget=history_budget,
        history_recent_protect=history_protect,
    )
