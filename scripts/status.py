"""Progress of running or finished phases, read from the result files only (safe while a run is writing).

  python scripts/status.py --out "$TWOMEM_RESULTS"                       # phases 1-4
  python scripts/status.py --phases preflight --set full --model Qwen/Qwen3.5-4B --limit 5
  python scripts/status.py --out "$TWOMEM_RESULTS" --log /content/drive/MyDrive/twomemories/logs/run_all.log

Per phase: finished cells, samples done / total, units charged to the phase (compute_log, the same number the
cap uses), and a remaining-time estimate. Below that: the cell written most recently, seconds since its last row
(a stalled run shows a growing number), the GPU and the last lines of the run log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase import _compute_rows, load_conditions, sec_per_sample  # noqa: E402

from twomem import data  # noqa: E402
from twomem.runner import RESULTS, cell_hash, cell_path, gpu_name, subset_id_of, units_per_hour  # noqa: E402

PHASES = ["phase1_probe", "phase2_bits", "phase3_tokens", "phase4_scale"]


def _rows(path: Path, h: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("cell_hash") == h:
                out.append(r)
    return out


def phase_status(phase: str, out: Path, cond_set=None, model=None, limit=None) -> dict:
    cfg, conds = load_conditions(phase, cond_set)
    models = [model] if model else cfg["models"]
    mp = cfg.get("max_pixels")
    cells = done_cells = n_done = n_total = 0
    secs = []
    for m in models:
        for task in cfg["tasks"]:
            n = limit or data.SUBSET_SIZES[task]
            sub = subset_id_of(task)
            for cond in [None] + conds:
                rows = _rows(cell_path(out, m, task, cond), cell_hash(m, task, cond, sub, mp))
                cells += 1
                done_cells += len(rows) >= n
                n_done += min(len(rows), n)
                n_total += n
                secs += [r["seconds"] for r in rows]
    log = out / "compute_log.jsonl"
    units = sum(r.get("units_est", 0.0) for r in _compute_rows(log) if r.get("phase") == phase)
    # before a phase has rows, fall back to this model's mean over earlier cells (the same estimate --dry-run uses)
    sps = sum(secs) / len(secs) if secs else sum(sec_per_sample(m, gpu_name(), log) for m in models) / len(models)
    return {"phase": phase, "models": models, "cells": cells, "done_cells": done_cells, "n_done": n_done,
            "n_total": n_total, "sec_per_sample": sps, "measured": bool(secs), "units": units}


def latest_cell(out: Path) -> str:
    files = [p for p in out.glob("*/*/*.jsonl")]
    if not files:
        return "no result rows yet"
    p = max(files, key=lambda f: f.stat().st_mtime)
    last = json.loads(p.read_text().splitlines()[-1])
    ts = dt.datetime.fromisoformat(last["timestamp"])
    age = (dt.datetime.now(dt.timezone.utc) - ts).total_seconds()
    return (f"{last['model_id']} {last['task']} {p.stem}: {sum(1 for l in p.open() if l.strip())} rows, "
            f"last row {age:.0f}s ago ({last['seconds']:.1f}s/sample, commit {last['commit']})")


def gpu_line() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else gpu_name()
    except (OSError, subprocess.TimeoutExpired):
        return gpu_name()


def main(argv=None):
    import yaml

    from run_phase import CONFIGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--phases", nargs="*", default=PHASES)
    ap.add_argument("--set", dest="cond_set", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log", default=None, help="run log to tail")
    ap.add_argument("--tail", type=int, default=5)
    args = ap.parse_args(argv)

    out = Path(args.out)
    comp = yaml.safe_load((CONFIGS / "compute.yaml").read_text())
    uph = units_per_hour(gpu_name())
    print(f"{dt.datetime.now().strftime('%H:%M:%S')}  results={out}  units/h={uph}")
    for ph in args.phases:
        s = phase_status(ph, out, args.cond_set, args.model, args.limit)
        cap = float(comp["phase_caps"].get(ph, comp["total_units"]))
        left = s["n_total"] - s["n_done"]
        h = left * s["sec_per_sample"] / 3600
        eta = f"eta {h * 60:.0f} min, ~{h * uph:.1f} more units" + ("" if s["measured"] else " (prior estimate)")
        mark = "done" if left == 0 else ("running/partial" if s["n_done"] else "not started")
        print(f"  {ph:14s} {mark:15s} cells {s['done_cells']:3d}/{s['cells']:<3d} samples {s['n_done']:5d}/"
              f"{s['n_total']:<5d} units {s['units']:6.2f}/{cap:<5.0f} {eta}")
    print("latest:", latest_cell(out))
    print("gpu:   ", gpu_line())
    if args.log and Path(args.log).exists():
        lines = Path(args.log).read_text(errors="replace").splitlines()
        print(f"--- last {args.tail} lines of {args.log}")
        print("\n".join(lines[-args.tail:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
