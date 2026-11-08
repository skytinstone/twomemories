"""Paper tables from results/**/*.jsonl only.

Table 2: fidelity condition x task -> accuracy retained [95% paired-bootstrap CI] (primary, D-31).
Table 2nll: the same with loss retained; POPE and DocVQA marked (letter case of the reference, D-31).
Table 3: raw task scores per condition (ANLS / relaxed acc / VQA acc / POPE acc and F1).
Table 5: visual tokens x KV bits -> total MB and both retained ratios, grouped by matching MB.
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

from twomem.metrics import paired_bootstrap, pope_f1
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
    Condition(kv="4", state="8"), Condition(kv="4", state="8ef"), Condition(kv="4", state="6"),
    Condition(kv="4", state="6ef"),
    Condition(kv="4", state="8", scope="whole"), Condition(kv="4", state="8ef", scope="whole"),
    Condition(kv="drop"), Condition(state="zeroconv"), Condition(kv="drop", state="zeroconv"),
]
TABLE5_TOKENS = [(1.0, "none"), (0.5, "uniform"), (0.25, "uniform")]
TABLE5_KV = ["full", "4"]
METRICS = ("acc", "nll")
NLL_CASE_MIXED = ("docvqa", "pope")   # references differ from the model's casing often (D-31)


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
    """Per-sample token-mean gold NLL, task score and prediction, keyed by (model, task, condition or 'blind')."""

    def __init__(self, rows: list[dict]):
        self.nll = defaultdict(dict)
        self.mb = defaultdict(list)
        self.score = defaultdict(dict)
        self.pred = defaultdict(dict)
        self.gold = {}
        for r in rows:
            key = (r["model_id"], r["task"], "blind" if r.get("blind") else Condition.from_dict(r["cond"]))
            if r["sample_id"] in self.nll[key]:   # a sample written twice across sessions counts once
                continue
            self.nll[key][r["sample_id"]] = r["gold_nll"] / max(1, r["gold_ntok"])
            self.mb[key].append(r["total_mb"])
            self.score[key][r["sample_id"]] = r["score"]
            self.pred[key][r["sample_id"]] = r["pred"]
            self.gold[(r["task"], r["sample_id"])] = r["meta"]["gold"]

    def retained(self, model: str, task: str, cond: Condition, metric: str = "acc") -> dict | None:
        """Dataset-level retained ratio with paired bootstrap CI. acc: (s_c - s_blind) / (s_full - s_blind),
        computed as the loss form on negated scores; nll: (N_blind - N_c) / (N_blind - N_full)."""
        src = self.score if metric == "acc" else self.nll
        b, f, c = (src.get((model, task, k)) for k in ("blind", FULL, cond))
        if not b or not f or not c:
            return None
        ids = sorted(set(b) & set(f) & set(c))
        if not ids:
            return None
        sign = -1.0 if metric == "acc" else 1.0
        return paired_bootstrap(*([sign * x[i] for i in ids] for x in (b, f, c)))

    def mean_score(self, model: str, task: str, cond) -> float | None:
        v = self.score.get((model, task, cond))
        return float(np.mean(list(v.values()))) if v else None

    def pope_f1(self, model: str, cond) -> float | None:
        p = self.pred.get((model, "pope", cond))
        if not p:
            return None
        ids = sorted(p)
        return pope_f1([p[i] for i in ids], [self.gold[("pope", i)] for i in ids])["f1"]

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


def _task_head(metric: str) -> list[str]:
    mark = "\u2020" if metric == "nll" else ""
    return [TASK_NAMES[t] + (mark if t in NLL_CASE_MIXED else "") for t in TASKS]


def table2(idx: Index, model: str, metric: str = "acc") -> tuple[str, str]:
    head = ["condition"] + _task_head(metric)
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{tabular}{l" + "c" * len(TASKS) + "}", "\\toprule",
           " & ".join(h.replace("\u2020", "$^\\dagger$") for h in head) + " \\\\", "\\midrule"]
    for c in TABLE2_ROWS:
        cells = [idx.retained(model, t, c, metric) for t in TASKS]
        md.append("| " + " | ".join([_cond_name(c)] + [_fmt(x) for x in cells]) + " |")
        tex.append(" & ".join([_cond_name(c)] + [_fmt(x, True) for x in cells]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def table3(idx: Index, model: str) -> tuple[str, str]:
    """Raw task scores: unaltered, blind and every Table 2 condition; POPE as accuracy / F1."""
    head = ["condition"] + [TASK_NAMES[t] for t in TASKS]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{tabular}{l" + "c" * len(TASKS) + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for c in [FULL, "blind"] + TABLE2_ROWS:
        cells = []
        for t in TASKS:
            m = idx.mean_score(model, t, c)
            s = "n/e" if m is None else f"{m:.3f}"
            if t == "pope" and m is not None:
                s += f" / {idx.pope_f1(model, c):.3f}"
            cells.append(s)
        name = "blind (no image)" if c == "blind" else ("unaltered" if c == FULL else _cond_name(c))
        md.append("| " + " | ".join([name] + cells) + " |")
        tex.append(" & ".join([name] + cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def table5(idx: Index, model: str) -> tuple[str, str]:
    head = ["task", "tokens", "KV bits", "total MB", "acc retained [95% CI]", "loss retained [95% CI]", "MB group"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{tabular}{llcccc c}", "\\toprule", " & ".join(head).replace("%", "\\%") + " \\\\", "\\midrule"]
    for t in TASKS:
        entries = []
        for keep, method in TABLE5_TOKENS:
            for kv in TABLE5_KV:
                c = Condition(kv=kv, token_keep=keep, token_method=method)
                entries.append((keep, kv, idx.mean_mb(model, t, c), idx.retained(model, t, c, "acc"),
                                idx.retained(model, t, c, "nll")))
        groups = _mb_groups([e[2] for e in entries])
        name = TASK_NAMES[t] + ("\u2020" if t in NLL_CASE_MIXED else "")
        for (keep, kv, mbv, ca, cn), g in zip(entries, groups):
            mbs = "n/e" if mbv is None else f"{mbv:.2f}"
            row = [name, f"{int(keep * 100)}%", kv, mbs, _fmt(ca), _fmt(cn), g]
            md.append("| " + " | ".join(row) + " |")
            tex.append(" & ".join([name.replace("\u2020", "$^\\dagger$"), row[1].replace("%", "\\%")] + row[2:4]
                                  + [_fmt(ca, True), _fmt(cn, True), g]) + " \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    return "\n".join(md), "\n".join(tex)


NLL_NOTE = ("\u2020 loss retained on DocVQA and POPE mixes in the letter case of the reference (lower-case gold, "
            "model writes e.g. 'No'); accuracy retained is the primary measure (D-31).")


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
    note = (f"> mps/cpu rows included (results={args.results}); paper numbers only from results/mac (D-30).\n\n" if args.allow_local else "")
    tables = (("table2", lambda i, m: table2(i, m, "acc")), ("table2nll", lambda i, m: table2(i, m, "nll")),
              ("table3", table3), ("table5", table5))
    for name, fn in tables:
        md, tex = fn(idx, args.model)
        foot = f"\n\n{NLL_NOTE}" if name in ("table2nll", "table5") else ""
        (out / f"{name}.md").write_text(f"{note}**{name}** model={args.model}\n\n{md}{foot}\n")
        (out / f"{name}.tex").write_text((f"% mps/cpu rows included (results={args.results}); paper numbers only from results/mac (D-30)\n" if args.allow_local else "") + tex + "\n")
        print(f"== {name} ({args.model})\n{md}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
