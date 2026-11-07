"""Paper tables from results/**/*.jsonl only.

Table 2: fidelity condition x task -> retained [95% paired-bootstrap CI].
Table 5: visual tokens x KV bits -> total MB and retained, grouped by matching MB.
Cells with no data print "n/e" (not evaluated). Rows from mps/cpu runs are local
checks and are excluded unless --allow-local (then the header says so).

  python analysis/make_tables.py --model Qwen/Qwen3.5-4B
  python analysis/make_tables.py --model Qwen/Qwen3.5-0.8B --allow-local --out /tmp/tables
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from twomem.metrics import paired_bootstrap
from twomem.quant import Condition
from twomem.runner import read_jsonl

REPO = Path(__file__).resolve().parents[1]
TASKS = ("docvqa", "chartqa", "textvqa", "pope")
TASK_NAMES = {"docvqa": "DocVQA", "chartqa": "ChartQA", "textvqa": "TextVQA", "pope": "POPE"}
LOCAL_GPUS = ("mps", "cpu")
FULL = Condition()

TABLE2_ROWS = [
    Condition(kv="4"), Condition(kv="noise"), Condition(kv="zero"),
    Condition(state="8"), Condition(state="8ef"), Condition(state="6"), Condition(state="6ef"),
    Condition(state="noise"), Condition(state="zero"), Condition(kv="zero", state="zero"),
    Condition(kv="4", state="8"), Condition(kv="4", state="8ef"),
    Condition(kv="4", state="8", scope="whole"), Condition(kv="4", state="8ef", scope="whole"),
]
TABLE5_TOKENS = [(1.0, "none"), (0.5, "uniform"), (0.25, "uniform")]
TABLE5_KV = ["full", "4"]


def load_rows(results: Path, allow_local: bool) -> list[dict]:
    rows = []
    for p in sorted(results.rglob("*.jsonl")):
        if p.name == "compute_log.jsonl":
            continue
        for r in read_jsonl(p):
            if allow_local or r.get("gpu") not in LOCAL_GPUS:
                rows.append(r)
    return rows


class Index:
    """Per-sample token-mean gold NLL, keyed by (model, task, condition or 'blind')."""

    def __init__(self, rows: list[dict]):
        self.nll = defaultdict(dict)
        self.mb = defaultdict(list)
        self.score = defaultdict(list)
        for r in rows:
            key = (r["model_id"], r["task"], "blind" if r.get("blind") else Condition.from_dict(r["cond"]))
            if r["sample_id"] in self.nll[key]:   # a sample written twice across sessions counts once
                continue
            self.nll[key][r["sample_id"]] = r["gold_nll"] / max(1, r["gold_ntok"])
            self.mb[key].append(r["total_mb"])
            self.score[key].append(r["score"])

    def retained(self, model: str, task: str, cond: Condition) -> dict | None:
        b, f, c = (self.nll.get((model, task, k)) for k in ("blind", FULL, cond))
        if not b or not f or not c:
            return None
        ids = sorted(set(b) & set(f) & set(c))
        if not ids:
            return None
        return paired_bootstrap([b[i] for i in ids], [f[i] for i in ids], [c[i] for i in ids])

    def mean_mb(self, model, task, cond) -> float | None:
        v = self.mb.get((model, task, cond))
        return float(np.mean(v)) if v else None


def _fmt(ci: dict | None, tex: bool = False) -> str:
    if ci is None:
        return "n/e"
    pm = (f"{ci['retained']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]")
    return pm.replace("[", "{\\scriptsize[").replace("]", "]}") if tex else pm


def _cond_name(c: Condition) -> str:
    s = f"KV {c.kv} / State {c.state}"
    if c.token_keep != 1.0:
        s += f" / tokens {int(c.token_keep * 100)}% {c.token_method}"
    if c.scope == "whole":
        s += " (whole)"
    return s


def table2(idx: Index, model: str) -> tuple[str, str]:
    head = ["condition"] + [TASK_NAMES[t] for t in TASKS]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{tabular}{l" + "c" * len(TASKS) + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for c in TABLE2_ROWS:
        cells = [idx.retained(model, t, c) for t in TASKS]
        md.append("| " + " | ".join([_cond_name(c)] + [_fmt(x) for x in cells]) + " |")
        tex.append(" & ".join([_cond_name(c)] + [_fmt(x, True) for x in cells]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def table5(idx: Index, model: str) -> tuple[str, str]:
    head = ["task", "tokens", "KV bits", "total MB", "retained [95% CI]", "MB group"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{tabular}{llccl c}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for t in TASKS:
        entries = []
        for keep, method in TABLE5_TOKENS:
            for kv in TABLE5_KV:
                c = Condition(kv=kv, token_keep=keep, token_method=method)
                entries.append((keep, kv, idx.mean_mb(model, t, c), idx.retained(model, t, c)))
        groups = _mb_groups([e[2] for e in entries])
        for (keep, kv, mbv, ci), g in zip(entries, groups):
            mbs = "n/e" if mbv is None else f"{mbv:.2f}"
            row = [TASK_NAMES[t], f"{int(keep * 100)}%", kv, mbs, _fmt(ci), g]
            md.append("| " + " | ".join(row) + " |")
            tex.append(" & ".join(row[:4] + [_fmt(ci, True), g]) + " \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    return "\n".join(md), "\n".join(tex)


def _mb_groups(mbs: list[float | None], rel: float = 0.10) -> list[str]:
    """Label entries whose total MB agree within ``rel`` with the same letter."""
    labels, reps = [], []
    for v in mbs:
        if v is None:
            labels.append("-")
            continue
        for i, r in enumerate(reps):
            if abs(v - r) / max(r, 1e-9) <= rel:
                labels.append(chr(ord("A") + i))
                break
        else:
            reps.append(v)
            labels.append(chr(ord("A") + len(reps) - 1))
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "paper" / "tables"))
    ap.add_argument("--allow-local", action="store_true")
    args = ap.parse_args(argv)

    idx = Index(load_rows(Path(args.results), args.allow_local))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    note = ("> LOCAL CHECK (mps/cpu rows included). Not for the paper.\n\n" if args.allow_local else "")
    for name, fn in (("table2", table2), ("table5", table5)):
        md, tex = fn(idx, args.model)
        (out / f"{name}.md").write_text(f"{note}**{name}** model={args.model}\n\n{md}\n")
        (out / f"{name}.tex").write_text(("% LOCAL CHECK, not for the paper\n" if args.allow_local else "") + tex + "\n")
        print(f"== {name} ({args.model})\n{md}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
