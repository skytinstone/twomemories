"""Phase report in Korean and English: cells run, samples, compute units, and the retained tables.

  python analysis/report.py --phase phase1_probe --model Qwen/Qwen3.5-4B > docs/report_phase1.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analysis.make_tables import REPO, Index, load_rows, table2, table5
from twomem.runner import read_jsonl


def compute_summary(log: Path, phase: str | None) -> dict:
    rows = read_jsonl(log)
    if phase:
        rows = [r for r in rows if r.get("phase") == phase]
    by_gpu = defaultdict(lambda: {"cells": 0, "samples": 0, "seconds": 0.0, "units": 0.0, "vram": 0.0})
    for r in rows:
        g = by_gpu[r["gpu"]]
        g["cells"] += 1
        g["samples"] += r.get("n_samples", 0)
        g["seconds"] += r["seconds"]
        g["units"] += r["units_est"]
        g["vram"] = max(g["vram"], r.get("peak_vram_gb") or 0.0)
    return dict(by_gpu)


def render(phase: str | None, model: str, allow_local: bool, results: Path = REPO / "results") -> str:
    idx = Index(load_rows(results, allow_local))
    comp = compute_summary(results / "compute_log.jsonl", phase)
    out = [f"# 단계 보고 / Phase report: {phase or 'all'} ({model})", ""]
    if allow_local:
        out += ["> 로컬 검증 수치 포함, 논문 미사용. / Includes local check numbers; not for the paper.", ""]
    out += ["## 컴퓨팅 / Compute", "", "| GPU | cells | samples | sec | sec/sample | units (est.) | peak VRAM GB |",
            "|---|---|---|---|---|---|---|"]
    for gpu, g in comp.items():
        sps = g["seconds"] / g["samples"] if g["samples"] else float("nan")
        out.append(f"| {gpu} | {g['cells']} | {g['samples']} | {g['seconds']:.0f} | {sps:.2f} | {g['units']:.2f} | "
                   f"{g['vram']:.1f} |")
    md2, _ = table2(idx, model)
    md5, _ = table5(idx, model)
    out += ["", "## Table 2: 충실도 × 태스크, retained [95% CI] / fidelity x task", "", md2,
            "", "## Table 5: 토큰 × KV 비트 / tokens x KV bits (같은 MB 묶음 / iso-MB groups)", "", md5, "",
            "n/e = 평가 안 함 / not evaluated. retained = (NLL_blind − NLL_cond)/(NLL_blind − NLL_full), "
            "데이터셋 수준 / dataset level, paired bootstrap n=2000."]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--allow-local", action="store_true")
    args = ap.parse_args(argv)
    print(render(args.phase, args.model, args.allow_local, Path(args.results)))


if __name__ == "__main__":
    main()
