#!/usr/bin/env python3
"""Mews cost watchdog.

Tails a synthesis log, parses per-call cost lines emitted by BespokeOLAPs
get_tokens_context_and_dollar_info(), and:
  - prints a running total at a fixed cadence
  - aborts the synthesis (SIGTERM the target PID) if total exceeds threshold
  - records each event to a JSONL ledger

Cost line format (from utils/token_usage.py):
  ... | Estimated cost: $0.001234 | LLM requests: N
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

COST_RE = re.compile(r"cached_litellm[^:]*:Cost:\s+\$(\d+\.\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TOKENS_RE = re.compile(
    r"Input tokens:\s+(\d+)\s+\(cached:\s+(\d+)\),\s+Output tokens:\s+(\d+)\s+\(reasoning:\s+(\d+)\)"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def watch(log_path: Path, threshold_usd: float, target_pid: int | None,
          ledger_path: Path, poll_seconds: int, idle_exit_seconds: int) -> int:
    print(f"[cost-watch] {now_iso()} starting; log={log_path} threshold=${threshold_usd:.2f} pid={target_pid}", flush=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    n_calls = 0
    inode = None
    pos = 0
    last_growth = time.time()

    while True:
        if not log_path.exists():
            print(f"[cost-watch] {now_iso()} waiting for log...", flush=True)
            time.sleep(poll_seconds)
            continue

        st = log_path.stat()
        if inode is None or st.st_ino != inode:
            if inode is not None:
                print(f"[cost-watch] {now_iso()} log rotated (new inode); restarting at offset 0", flush=True)
            inode = st.st_ino
            pos = 0

        if st.st_size > pos:
            with log_path.open("r", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                line_clean = ANSI_RE.sub("", line)
                m = COST_RE.search(line_clean)
                if not m:
                    continue
                cost = float(m.group(1))
                total += cost
                n_calls += 1
                tok_m = TOKENS_RE.search(line_clean)
                tokens = {}
                if tok_m:
                    tokens = {
                        "input_tokens": int(tok_m.group(1)),
                        "cached_tokens": int(tok_m.group(2)),
                        "output_tokens": int(tok_m.group(3)),
                        "reasoning_tokens": int(tok_m.group(4)),
                    }
                with ledger_path.open("a") as ld:
                    ld.write(json.dumps({
                        "ts": now_iso(),
                        "call_index": n_calls,
                        "call_cost_usd": cost,
                        "running_total_usd": round(total, 6),
                        **tokens,
                    }) + "\n")
            last_growth = time.time()

        # report
        print(f"[cost-watch] {now_iso()} total=${total:.4f} calls={n_calls} log_size={st.st_size}", flush=True)

        # threshold check
        if total > threshold_usd:
            print(f"[cost-watch] {now_iso()} THRESHOLD EXCEEDED total=${total:.4f} > ${threshold_usd:.2f}", flush=True)
            if target_pid is not None:
                try:
                    os.kill(target_pid, signal.SIGTERM)
                    print(f"[cost-watch] {now_iso()} sent SIGTERM to PID {target_pid}", flush=True)
                except ProcessLookupError:
                    print(f"[cost-watch] {now_iso()} target PID {target_pid} not found", flush=True)
            return 1

        # idle exit
        if idle_exit_seconds > 0 and time.time() - last_growth > idle_exit_seconds:
            print(f"[cost-watch] {now_iso()} log idle >{idle_exit_seconds}s; assuming run ended cleanly", flush=True)
            return 0

        time.sleep(poll_seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path, help="Path to synthesis run log")
    p.add_argument("--threshold-usd", type=float, default=225.0)
    p.add_argument("--target-pid", type=int, default=None, help="PID to SIGTERM if threshold exceeded")
    p.add_argument("--ledger", type=Path, default=Path("~/projects/mews/experiments/gate-0/cost-ledger.jsonl").expanduser())
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--idle-exit-seconds", type=int, default=600, help="Exit cleanly if log has not grown for this many seconds (0 disables)")
    args = p.parse_args()
    rc = watch(args.log, args.threshold_usd, args.target_pid, args.ledger, args.poll_seconds, args.idle_exit_seconds)
    sys.exit(rc)


if __name__ == "__main__":
    main()
