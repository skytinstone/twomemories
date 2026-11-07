"""Progress of running or finished phases, read from the result files only (safe while a run is writing).

  python scripts/status.py --out "$TWOMEM_RESULTS"                     # phases 1-5, once
  python scripts/status.py --out "$TWOMEM_RESULTS" --watch 3           # refresh every 3 s
  python scripts/status.py --phases preflight --set full --model Qwen/Qwen3.5-4B --limit 5

The notebook monitor cell imports ``Monitor`` and calls ``render()`` every 3 s. Files are read incrementally
(only bytes appended since the last refresh), so a refresh costs a few stat calls.

Shown: overall and per-phase percent of samples, the current phase, finish times (wall-clock rate of the last
rows of the same model; a model with no rows yet uses the compute_log / prior estimate and is marked "prior"),
units charged per phase (compute_log, the number the cap uses), the latest cell and seconds since its last row,
the GPU, whether the run process is alive, and the last log lines.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase import CONFIGS, load_conditions, sec_per_sample  # noqa: E402

from twomem import data  # noqa: E402
from twomem.runner import RESULTS, cell_hash, cell_path, gpu_name, subset_id_of, units_per_hour  # noqa: E402

PHASES = ["phase1_probe", "phase2_bits", "phase3_tokens", "phase4_scale", "phase5_channels"]
TZ = ZoneInfo("Asia/Seoul")
RATE_WINDOW = 900   # seconds of recent rows used for the wall-clock rate


class _Tail:
    """Incremental jsonl reader: parses only complete lines appended since the last call."""

    def __init__(self, path: Path):
        self.path, self.offset, self.buf = Path(path), 0, b""

    def new_rows(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:   # file was replaced: start over
            self.offset, self.buf = 0, b""
        if size == self.offset:
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read(size - self.offset)
        self.offset = size
        *lines, self.buf = (self.buf + chunk).split(b"\n")
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows


def _ts(r: dict) -> float:
    return dt.datetime.fromisoformat(r["timestamp"]).timestamp()


def alive(pid_file: str | Path | None) -> bool | None:
    """True/False for the run process in ``pid_file``; None when there is no pid file. Zombies count as dead."""
    if not pid_file or not Path(pid_file).exists():
        return None
    try:
        pid = int(Path(pid_file).read_text())
    except ValueError:
        return None
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        if Path("/proc").is_dir():
            return False
    try:   # no /proc (macOS)
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_line() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else gpu_name()
    except (OSError, subprocess.TimeoutExpired):
        return gpu_name()


def _bar(frac: float, width: int = 24) -> str:
    n = round(frac * width)
    return "█" * n + "░" * (width - n)


def _dur(sec: float) -> str:
    m = int(sec // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def _clock(t: float) -> str:
    return dt.datetime.fromtimestamp(t, TZ).strftime("%m-%d %H:%M")


class Monitor:
    def __init__(self, out: str | Path = RESULTS, phases=PHASES, cond_set=None, model=None, limit=None,
                 log: str | Path | None = None, pid_file: str | Path | None = None):
        import yaml

        self.out, self.phases, self.log, self.pid_file = Path(out), list(phases), log, pid_file
        comp = yaml.safe_load((CONFIGS / "compute.yaml").read_text())
        self.caps = {p: float(comp["phase_caps"].get(p, comp["total_units"])) for p in self.phases}
        self.gpu = gpu_name()
        self.uph = units_per_hour(self.gpu)
        # plan: phase -> [(cell key, path, n, model)]; a cell shared by two phases has one key
        self.plan, self.models = {}, {}
        subs = {}
        for ph in self.phases:
            cfg, conds = load_conditions(ph, cond_set)
            models = [model] if model else cfg["models"]
            self.models[ph] = models
            cells = []
            for m in models:
                for task in cfg["tasks"]:
                    sub = subs.setdefault(task, subset_id_of(task))
                    for c in [None] + conds:
                        h = cell_hash(m, task, c, sub, cfg.get("max_pixels"))
                        cells.append((h, cell_path(self.out, m, task, c), limit or data.SUBSET_SIZES[task], m))
            self.plan[ph] = cells
        self.tails = {p: _Tail(p) for p in {c[1] for cells in self.plan.values() for c in cells}}
        self.comp_tail = _Tail(self.out / "compute_log.jsonl")
        self.done = defaultdict(set)        # cell hash -> sample ids
        self.times = defaultdict(list)      # model -> row timestamps
        self.secs = defaultdict(list)       # model -> per-row seconds
        self.units = defaultdict(float)     # phase -> units from compute_log
        self.latest = None                  # (timestamp, path, row)

    def refresh(self) -> None:
        for path, tail in self.tails.items():
            for r in tail.new_rows():
                if "sample_id" not in r or r["sample_id"] in self.done[r.get("cell_hash")]:
                    continue
                self.done[r["cell_hash"]].add(r["sample_id"])
                t = _ts(r)
                self.times[r["model_id"]].append(t)
                self.secs[r["model_id"]].append(r["seconds"])
                if self.latest is None or t >= self.latest[0]:
                    self.latest = (t, path, r)
        for r in self.comp_tail.new_rows():
            if r.get("phase"):
                self.units[r["phase"]] += r.get("units_est", 0.0)

    def sec_per_sample(self, model: str) -> tuple[float, str]:
        """(seconds per sample, basis). Wall clock over recent rows includes model loads and downloads."""
        ts = sorted(self.times.get(model, []))
        recent = [t for t in ts if t >= (ts[-1] if ts else 0) - RATE_WINDOW]
        if len(recent) >= 10 and recent[-1] > recent[0]:
            return (recent[-1] - recent[0]) / (len(recent) - 1), "rate"
        if self.secs.get(model):
            return sum(self.secs[model]) / len(self.secs[model]), "mean"
        return sec_per_sample(model, self.gpu, self.out / "compute_log.jsonl"), "prior"

    def snapshot(self) -> dict:
        self.refresh()
        now = time.time()
        phases, seen, t_left_all, units_left_all = [], set(), 0.0, 0.0
        uniq_done = uniq_total = 0
        current = None
        for ph in self.phases:
            n_done = n_total = cells_done = 0
            left = 0.0
            basis = set()
            for h, _, n, m in self.plan[ph]:
                d = min(len(self.done.get(h, ())), n)
                n_done += d
                n_total += n
                cells_done += d >= n
                if h not in seen:
                    seen.add(h)
                    uniq_done += d
                    uniq_total += n
                    sps, b = self.sec_per_sample(m)
                    left += (n - d) * sps
                    basis.add(b)
            t_left_all += left
            units_left_all += left / 3600 * self.uph
            if current is None and n_done < n_total:
                current = ph
            phases.append({"phase": ph, "n_done": n_done, "n_total": n_total, "cells_done": cells_done,
                           "cells": len(self.plan[ph]), "left_s": left, "finish": now + t_left_all,
                           "prior": "prior" in basis and left > 0,
                           "units": self.units.get(ph, 0.0), "cap": self.caps[ph]})
        return {"now": now, "phases": phases, "current": current, "done": uniq_done, "total": uniq_total,
                "left_s": t_left_all, "units_left": units_left_all, "latest": self.latest}

    def render(self) -> str:
        s = self.snapshot()
        frac = s["done"] / s["total"] if s["total"] else 0.0
        lines = [f"twomemories  {_clock(s['now'])} KST  results={self.out}  units/h={self.uph}", ""]
        if s["current"] is None:
            lines.append(f"전체 [{_bar(1.0)}] 100.0%  {s['done']:,}/{s['total']:,} samples  완료")
        else:
            lines.append(f"전체 [{_bar(frac)}] {100 * frac:5.1f}%  {s['done']:,}/{s['total']:,} samples  "
                         f"완료 예정 {_clock(s['now'] + s['left_s'])} KST ({_dur(s['left_s'])} 남음, "
                         f"~{s['units_left']:.1f} units 더)")
            i = self.phases.index(s["current"]) + 1
            lines.append(f"현재 단계: {i}/{len(self.phases)} {s['current']}")
        lines.append("")
        for p in s["phases"]:
            f = p["n_done"] / p["n_total"] if p["n_total"] else 0.0
            if p["n_done"] >= p["n_total"]:
                tail = "완료"
            else:
                tail = f"끝 {_clock(p['finish'])} KST ({_dur(p['left_s'])})" + (" prior" if p["prior"] else "")
            lines.append(f"  {p['phase']:14s} [{_bar(f, 16)}] {100 * f:5.1f}%  cells {p['cells_done']:3d}/"
                         f"{p['cells']:<3d} units {p['units']:6.2f}/{p['cap']:<4.0f} {tail}")
        lines.append("")
        if s["latest"]:
            t, path, r = s["latest"]
            n = len(self.done[r["cell_hash"]])
            lines.append(f"최근 셀: {r['model_id']} {r['task']} {path.stem}  {n} rows, 마지막 줄 {s['now'] - t:.0f}s 전 "
                         f"({r['seconds']:.1f}s/sample, commit {r['commit']})")
        else:
            lines.append("최근 셀: 아직 결과 없음")
        lines.append(f"GPU: {gpu_line()}")
        a = alive(self.pid_file)
        if a is not None:
            lines.append("프로세스: 실행 중" if a else "프로세스: 멈춤 (끝났거나 중지됨, 로그 확인)")
        if self.log and Path(self.log).exists():
            with Path(self.log).open("rb") as f:
                f.seek(max(0, Path(self.log).stat().st_size - 4096))
                tail = f.read().decode(errors="replace").splitlines()[-4:]
            lines += ["", f"--- {self.log}", *tail]
        return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--phases", nargs="*", default=PHASES)
    ap.add_argument("--set", dest="cond_set", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log", default=None, help="run log to tail")
    ap.add_argument("--pid-file", default=None)
    ap.add_argument("--watch", type=float, default=0, help="refresh interval in seconds (0 = print once)")
    args = ap.parse_args(argv)

    mon = Monitor(args.out, args.phases, args.cond_set, args.model, args.limit, args.log, args.pid_file)
    while True:
        text = mon.render()
        if not args.watch:
            print(text)
            return 0
        print("\033[2J\033[H" + text, flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
