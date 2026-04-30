"""Smoke tests for lockin_apply_patch helpers + fast paths.

End-to-end tests (with engine compile + validate_all + revert) are deferred
to Gate 3 work, where the auto-revert path will fire naturally. Here we
verify the cheap testable surface: imports, gate predicates, marker
detection, snapshot path computation, and wrapper instantiation.

Run:
    cd ~/projects/mews/upstream/BespokeOLAP && \\
    UV_CACHE_DIR=/tmp/uv_cache ~/.local/bin/uv run python \\
        ~/projects/mews/infra/tools/test_lockin_apply_patch.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import os
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, os.environ.get("BESPOKE_ROOT", str(_REPO_ROOT / "upstream" / "BespokeOLAP")))
sys.path.insert(0, str(_REPO_ROOT))

from infra.tools.lockin_apply_patch import (  # noqa: E402
    _BEST_BENCH_FILE,
    _GATED_FILES,
    _LOCKIN_MARKER,
    _SNAPSHOT_DIR,
    BEST_KEEPER_TOLERANCE,
    _load_best_bench,
    _lockin_armed,
    _save_best_bench,
    _should_gate,
    _snapshot_path,
    make_lockin_apply_patch_tool,
)


def test_should_gate():
    assert _should_gate("engine.cpp") is True, "engine.cpp must be gated"
    assert _should_gate("./engine.cpp") is True, "relative path must work"
    assert _should_gate("subdir/engine.cpp") is True, "nested engine.cpp must be gated"
    assert _should_gate("notes.txt") is False, "txt files must not be gated"
    assert _should_gate("storage_plan.txt") is False, "storage plan must not be gated"
    assert _should_gate("Makefile") is False, "Makefile must not be gated"
    assert _should_gate("engine.hpp") is False, "engine.hpp not in default gate set"
    print("[test_should_gate] PASS")


def test_lockin_armed():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        assert _lockin_armed(ws) is False, "fresh dir should not be armed"
        (ws / _LOCKIN_MARKER).touch()
        assert _lockin_armed(ws) is True, "marker file should arm lockin"
        (ws / _LOCKIN_MARKER).unlink()
        assert _lockin_armed(ws) is False, "removing marker should disarm"
    print("[test_lockin_armed] PASS")


def test_snapshot_path():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        p = _snapshot_path(ws, "engine.cpp", "pre")
        assert p.parent.name == _SNAPSHOT_DIR, f"expected {_SNAPSHOT_DIR} dir, got {p.parent.name}"
        assert p.name == "engine.cpp.pre.snap", f"got {p.name}"
        assert p.parent.exists(), "snapshot dir should be created"
        # nested path: should still flatten to filename
        p2 = _snapshot_path(ws, "subdir/engine.cpp", "lastpass")
        assert p2.name == "engine.cpp.lastpass.snap", f"got {p2.name}"
    print("[test_snapshot_path] PASS")


def test_make_tool_instantiation():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        tool = make_lockin_apply_patch_tool(root=ws, wandb_metrics_hook=None)
        assert tool.name == "apply_patch", f"tool name should be apply_patch, got {tool.name}"
        assert "Applies a unified diff" in tool.description, "description should be preserved"
        assert tool.params_json_schema is not None, "params schema should be set"
        assert callable(tool.on_invoke_tool), "on_invoke_tool must be callable"
    print("[test_make_tool_instantiation] PASS")


def test_gated_files_constant():
    # Sanity check on what we're gating
    assert "engine.cpp" in _GATED_FILES, "engine.cpp must be in gated set"
    print(f"[test_gated_files_constant] PASS (gated: {sorted(_GATED_FILES)})")


def test_best_bench_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        assert _load_best_bench(ws) is None, "fresh dir should have no best"
        _save_best_bench(ws, 2.05)
        assert (ws / _BEST_BENCH_FILE).exists(), "best bench file should be written"
        loaded = _load_best_bench(ws)
        assert loaded is not None, "loaded best should not be None"
        assert abs(loaded - 2.05) < 1e-9, f"roundtrip mismatch: got {loaded}"
        # Overwrite + reload
        _save_best_bench(ws, 1.95)
        assert abs(_load_best_bench(ws) - 1.95) < 1e-9, "overwrite should persist"
    print("[test_best_bench_roundtrip] PASS")


def test_best_bench_corrupt_returns_none():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / _BEST_BENCH_FILE).write_text("not json")
        assert _load_best_bench(ws) is None, "corrupt JSON should return None"
        (ws / _BEST_BENCH_FILE).write_text('{"wrong_key": 1.0}')
        assert _load_best_bench(ws) is None, "missing key should return None"
        (ws / _BEST_BENCH_FILE).write_text('{"engine_s_median": "not-a-float"}')
        assert _load_best_bench(ws) is None, "non-float value should return None"
    print("[test_best_bench_corrupt_returns_none] PASS")


def test_best_keeper_tolerance_constant():
    assert BEST_KEEPER_TOLERANCE > 1.0, "tolerance must allow some slack"
    assert BEST_KEEPER_TOLERANCE < 1.5, "tolerance must not be unbounded"
    print(f"[test_best_keeper_tolerance_constant] PASS (tolerance={BEST_KEEPER_TOLERANCE})")


def main():
    print("=" * 70)
    print("lockin_apply_patch smoke tests")
    print("=" * 70)
    tests = [
        test_should_gate,
        test_lockin_armed,
        test_snapshot_path,
        test_make_tool_instantiation,
        test_gated_files_constant,
        test_best_bench_roundtrip,
        test_best_bench_corrupt_returns_none,
        test_best_keeper_tolerance_constant,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[{t.__name__}] FAIL: {type(e).__name__}: {e}")
            failed.append(t.__name__)
    print("=" * 70)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"ALL {len(tests)} SMOKE TESTS PASSED")
    print("Note: end-to-end auto-revert tests (silent-x1 catch + best-keeper "
          "regression revert) require live engine compile + cliff parquets; "
          "exercised via the Gate-5 synthesis runs.")


if __name__ == "__main__":
    main()
