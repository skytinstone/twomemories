"""Figure: retained ratio per condition and task with 95% CI (results/*.jsonl only).

  python analysis/figures.py --model Qwen/Qwen3.5-4B --out paper/figures/retained.pdf
Needs matplotlib (pip install matplotlib); not a runtime dependency of twomem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.make_tables import REPO, TABLE2_ROWS, TASK_NAMES, TASKS, Index, _cond_name, load_rows


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default=str(REPO / "paper" / "figures" / "retained.pdf"))
    ap.add_argument("--allow-local", action="store_true")
    args = ap.parse_args(argv)

    idx = Index(load_rows(REPO / "results", args.allow_local))
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 5), sharey=True)
    for ax, t in zip(axes, TASKS):
        ys, labels = [], []
        for i, c in enumerate(TABLE2_ROWS):
            ci = idx.retained(args.model, t, c)
            labels.append(_cond_name(c))
            if ci is None:
                continue
            ax.errorbar(ci["retained"], i, xerr=[[ci["retained"] - ci["lo"]], [ci["hi"] - ci["retained"]]],
                        fmt="o", color="#333", capsize=2)
            ys.append(i)
        ax.axvline(1.0, color="#999", lw=0.8)
        ax.axvline(0.0, color="#999", lw=0.8, ls="--")
        ax.set_title(TASK_NAMES[t])
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels))
        ax.set_xlabel("retained")
    if args.allow_local:
        fig.suptitle("LOCAL CHECK, not for the paper")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
