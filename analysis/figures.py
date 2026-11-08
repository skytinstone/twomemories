"""Figure: accuracy retained (filled) and loss retained (open) per condition and task with 95% CI
(results/*.jsonl only). Loss retained on DocVQA and POPE mixes in the reference casing (D-31).

  python analysis/figures.py --model Qwen/Qwen3.5-4B --out paper/figures/retained.pdf   # also writes the .png
Needs matplotlib (pip install matplotlib); not a runtime dependency of twomem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.make_tables import NLL_CASE_MIXED, REPO, TABLE2_ROWS, TASK_NAMES, TASKS, Index, _cond_name, load_rows


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default=str(REPO / "paper" / "figures" / "retained.pdf"))
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--allow-local", action="store_true")
    args = ap.parse_args(argv)

    idx = Index(load_rows(Path(args.results), args.allow_local))
    if not any(idx.retained(args.model, t, c, m) for t in TASKS for c in TABLE2_ROWS for m in ("acc", "nll")):
        print(f"no retained values for {args.model} yet; no figure")
        return
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 5), sharey=True)
    handles = {}
    for ax, t in zip(axes, TASKS):
        ys, labels = [], []
        for i, c in enumerate(TABLE2_ROWS):
            labels.append(_cond_name(c))
            for metric, dy, style in (("acc", -0.17, dict(fmt="o", color="#222")),
                                      ("nll", 0.17, dict(fmt="o", color="#888", mfc="white"))):
                ci = idx.retained(args.model, t, c, metric)
                if ci is None:
                    continue
                h = ax.errorbar(ci["retained"], i + dy, xerr=[[ci["retained"] - ci["lo"]], [ci["hi"] - ci["retained"]]],
                                capsize=2, ms=4, **style)
                handles.setdefault(metric, h)
            ys.append(i)
        ax.axvline(1.0, color="#999", lw=0.8)
        ax.axvline(0.0, color="#999", lw=0.8, ls="--")
        ax.set_title(TASK_NAMES[t] + (" (loss: case-mixed)" if t in NLL_CASE_MIXED else ""))
        ax.set_xlabel("retained")
    # the y axis is shared: set the labels and the top-down order once
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    names = {"acc": "accuracy retained", "nll": "loss retained"}
    fig.legend([handles[m] for m in handles], [names[m] for m in handles], loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
