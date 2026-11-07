"""Run every cell of a phase config (blind cell per task first, then conditions). Resumable.

  python scripts/run_phase.py --phase preflight --limit 5
  python scripts/run_phase.py --phase preflight --set full --model Qwen/Qwen3.5-4B --limit 5
  python scripts/run_phase.py --phase phase1_probe --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from twomem.quant import Condition
from twomem.runner import COMPUTE_LOG, REPO, RESULTS, cell_done, gpu_name, read_jsonl, run_blind, run_cell

CONFIGS = REPO / "configs"


def load_conditions(phase: str, cond_set: str | None = None) -> tuple[dict, list[Condition]]:
    cfg = yaml.safe_load((CONFIGS / f"{phase}.yaml").read_text())
    if "condition_sets" in cfg:
        raw = cfg["condition_sets"][cond_set or cfg["default_set"]]
    else:
        raw = cfg["conditions"]
    conds: list[Condition] = []
    for item in raw:
        if isinstance(item, str):   # a phase name: take that phase's conditions
            conds += load_conditions(item)[1]
        else:
            conds.append(Condition.from_dict(item))
    seen, uniq = set(), []
    for c in conds:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return cfg, uniq


def _compute_rows(path: Path = COMPUTE_LOG) -> list[dict]:
    return read_jsonl(path)


def sec_per_sample(model_id: str, gpu: str, log: Path = COMPUTE_LOG) -> float:
    rows = [r for r in _compute_rows(log) if r.get("model_id") == model_id and r.get("gpu") == gpu and r.get("n_samples")]
    if rows:
        return sum(r["seconds"] for r in rows) / sum(r["n_samples"] for r in rows)
    comp = yaml.safe_load((CONFIGS / "compute.yaml").read_text())
    return float(comp["sec_per_sample_guess"]["default"])


def phase_units(phase: str, log: Path = COMPUTE_LOG) -> float:
    return sum(r.get("units_est", 0.0) for r in _compute_rows(log) if r.get("phase") == phase)


def main(argv=None):
    from twomem import data
    from twomem.runner import units_per_hour

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--set", dest="cond_set", default=None, help="condition set for configs with condition_sets")
    ap.add_argument("--model", default=None, help="override the config's model list with one model id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg, conds = load_conditions(args.phase, args.cond_set)
    models = [args.model] if args.model else cfg["models"]
    tasks = args.tasks or cfg["tasks"]
    comp = yaml.safe_load((CONFIGS / "compute.yaml").read_text())
    cap = float(comp["phase_caps"].get(args.phase, comp["total_units"]))
    gpu = gpu_name()
    comp_log = Path(args.out) / "compute_log.jsonl"

    n_samples = {t: args.limit or data.SUBSET_SIZES[t] for t in tasks}
    n_cells = len(models) * len(tasks) * (len(conds) + 1)
    total = sum(n_samples.values()) * (len(conds) + 1) * len(models)
    est_units = sum(total / len(models) * sec_per_sample(m, gpu, comp_log) / 3600 * units_per_hour(gpu) for m in models)
    print(f"phase={args.phase} models={models} tasks={tasks} conditions={len(conds)} (+1 blind per task)")
    print(f"cells={n_cells} samples={total} gpu={gpu} est_units={est_units:.2f} cap={cap} "
          f"used_so_far={phase_units(args.phase, comp_log):.2f}")
    if args.dry_run:
        for c in conds:
            print("  ", c.label())
        return 0

    from twomem.model import load

    out, mp = Path(args.out), cfg.get("max_pixels")
    for model_id in models:
        # cells still missing samples; a phase finished in an earlier session is skipped without loading the model
        todo = [(t, c) for t in tasks for c in [None] + conds if cell_done(out, model_id, t, c, mp) < n_samples[t]]
        print(f"{model_id}: {len(todo)} of {len(tasks) * (len(conds) + 1)} cells left", flush=True)
        if not todo:
            continue
        model, processor = load(model_id, max_pixels=mp)
        for task, cond in todo:
            if phase_units(args.phase, comp_log) >= cap:
                print(f"STOP: phase {args.phase} reached its cap of {cap} units")
                return 2
            kw = dict(model=model, processor=processor, max_pixels=mp, compute_log=comp_log, phase=args.phase)
            if cond is None:
                res = run_blind(model_id, task, args.limit, out, **kw)
            else:
                res = run_cell(model_id, task, cond, None, args.limit, out, **kw)
            print(f"{model_id} {task} {'blind' if cond is None else cond.label()}: new={res['n_new']} "
                  f"skipped={res['n_skipped']} sec={res['seconds']:.1f}", flush=True)
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
