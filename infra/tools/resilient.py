"""Resilient tool-call wrapper.

Wraps a `FunctionTool` so that exceptions raised by its `on_invoke_tool`
callback are caught and returned to the agent as a structured error
string, instead of propagating up through the agents SDK as `UserError`
and aborting the entire stage.

Motivation (Gate 3, runs 1+3+4):
- Sonnet run-1 hit `apply_patch Invalid Context` after debugging the
  wrong tenant set. UserError aborted the run.
- Opus run-3 hit the same `Invalid Context` mid-libarrow-API-migration.
  UserError aborted the run after only 6 calls.
- Opus run-4 issued `sudo tee /proc/sys/vm/drop_caches`. The shell
  sandbox raised `RuntimeError("sudo rejected")`. UserError aborted
  the run after 7 calls — even though the engine binary was already
  verifiably correct on all three tenants.

In all three cases a recoverable tool error was promoted to a fatal
runner error. With this wrapper, the agent sees:

    TOOL_ERROR (recoverable, no state change): RuntimeError: sudo rejected

and can choose a different approach. Sandbox boundaries are unchanged;
this only changes the error-return surface, not what is allowed.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool

logger = logging.getLogger("mews.resilient")


def make_resilient_tool(inner: FunctionTool) -> FunctionTool:
    """Wrap `inner.on_invoke_tool` so exceptions become tool-result strings.

    The wrapped tool keeps the same name, description, and JSON schema as
    the inner tool — drop-in replacement.
    """
    inner_invoke = inner.on_invoke_tool

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            return await inner_invoke(ctx, args_json)
        except Exception as e:
            msg = (
                f"TOOL_ERROR (recoverable, no state change): "
                f"{type(e).__name__}: {e}\n\n"
                f"The tool call did not execute. Workspace state is unchanged. "
                f"Choose a different approach — do not retry the same call. "
                f"If this is a sandbox refusal (sudo, write outside workspace, etc.), "
                f"the issue is in your approach, not the validator or test environment."
            )
            logger.warning(f"resilient[{inner.name}]: caught {type(e).__name__}: {e}")
            return msg

    return FunctionTool(
        name=inner.name,
        description=inner.description,
        params_json_schema=inner.params_json_schema,
        on_invoke_tool=on_invoke,
    )
