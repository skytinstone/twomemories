"""Run a single cell from the command line.

  python scripts/run_cell.py --model Qwen/Qwen3.5-0.8B --task docvqa --kv zero --state full --limit 5
  python scripts/run_cell.py --model Qwen/Qwen3.5-0.8B --task docvqa --blind --limit 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twomem.quant import Condition
from twomem.runner import RESULTS, run_blind, run_cell


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--kv", default="full")
    ap.add_argument("--state", default="full")
    ap.add_argument("--token-keep", type=float, default=1.0)
    ap.add_argument("--token-method", default="none")
    ap.add_argument("--scope", default="image")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=1310720)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cond = None if args.blind else Condition(args.kv, args.state, args.token_keep, args.token_method, args.scope)
    print("cell:", "blind" if cond is None else cond.label(), args.model, args.task, f"limit={args.limit}")
    if args.dry_run:
        return 0
    if cond is None:
        res = run_blind(args.model, args.task, args.limit, Path(args.out), max_pixels=args.max_pixels)
    else:
        res = run_cell(args.model, args.task, cond, None, args.limit, Path(args.out), max_pixels=args.max_pixels)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
