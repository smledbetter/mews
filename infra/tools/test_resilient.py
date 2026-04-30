"""Smoke test for tools.resilient.make_resilient_tool."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import os
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, os.environ.get("BESPOKE_ROOT", str(_REPO_ROOT / "upstream" / "BespokeOLAP")))
sys.path.insert(0, str(_REPO_ROOT))

from agents.tool import FunctionTool

from infra.tools.resilient import make_resilient_tool


def _make_raising_tool(exc: Exception) -> FunctionTool:
    async def on_invoke(ctx, args_json: str) -> str:
        raise exc

    return FunctionTool(
        name="boom",
        description="Always raises",
        params_json_schema={"type": "object", "properties": {}, "required": []},
        on_invoke_tool=on_invoke,
    )


def _make_ok_tool() -> FunctionTool:
    async def on_invoke(ctx, args_json: str) -> str:
        return "ok"

    return FunctionTool(
        name="ok",
        description="Always returns ok",
        params_json_schema={"type": "object", "properties": {}, "required": []},
        on_invoke_tool=on_invoke,
    )


async def main() -> int:
    failed = 0

    # 1. RuntimeError ("sudo rejected") becomes a recoverable string.
    t = make_resilient_tool(_make_raising_tool(RuntimeError("sudo rejected")))
    out = await t.on_invoke_tool(None, "{}")
    if "TOOL_ERROR" not in out or "sudo rejected" not in out:
        print(f"FAIL: expected TOOL_ERROR + sudo rejected, got: {out}")
        failed += 1
    else:
        print("PASS: RuntimeError becomes TOOL_ERROR string")

    # 2. ValueError (e.g. apply_patch context drift) becomes a recoverable string.
    t = make_resilient_tool(_make_raising_tool(ValueError("Invalid Context 167:")))
    out = await t.on_invoke_tool(None, "{}")
    if "TOOL_ERROR" not in out or "Invalid Context" not in out:
        print(f"FAIL: expected Invalid Context in output, got: {out}")
        failed += 1
    else:
        print("PASS: ValueError becomes TOOL_ERROR string")

    # 3. Successful tool calls pass through unchanged.
    t = make_resilient_tool(_make_ok_tool())
    out = await t.on_invoke_tool(None, "{}")
    if out != "ok":
        print(f"FAIL: expected 'ok', got: {out}")
        failed += 1
    else:
        print("PASS: success path unchanged")

    # 4. Schema and name are preserved.
    inner = _make_ok_tool()
    wrapped = make_resilient_tool(inner)
    if wrapped.name != "ok" or wrapped.params_json_schema != inner.params_json_schema:
        print(f"FAIL: name/schema not preserved")
        failed += 1
    else:
        print("PASS: name + schema preserved")

    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
