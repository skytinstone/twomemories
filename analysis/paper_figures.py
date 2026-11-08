"""Paper figures beyond the retained overview (analysis/figures.py), from results/**/*.jsonl only.

  fig_tokens_bits.pdf  accuracy retained against the bits the image's KV rows cost (phase 3)
  fig_channels.pdf     accuracy retained when each carrier is removed (phases 1 and 5)
  fig_case.pdf         loss retained against accuracy retained, every condition and task (D-31)

  python analysis/paper_figures.py --model Qwen/Qwen3.5-0.8B --results results/mac --allow-local --out paper/figures
Needs matplotlib; not a runtime dependency of twomem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.make_tables import NLL_CASE_MIXED, TABLE2_ROWS, TASK_NAMES, TASKS, Index, load_rows
from twomem.quant import Condition

# categorical slots 1-4 of the dataviz reference palette, in fixed order (validated light-mode palette)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
TASK_STYLE = {"docvqa": (BLUE, "o"), "chartqa": (ORANGE, "s"), "textvqa": (AQUA, "^"), "pope": (YELLOW, "D")}

TOKENS = [(1.0, "none"), (0.5, "uniform"), (0.25, "uniform")]
CHANNELS = [
    ("KV zero", Condition(kv="zero")),
    ("KV drop", Condition(kv="drop")),
    ("state zero", Condition(state="zero")),
    ("state + conv ($0_C$)", Condition(state="zeroconv")),
    ("KV + state zero", Condition(kv="zero", state="zero")),
    ("KV drop + $0_C$", Condition(kv="drop", state="zeroconv")),
]


def _style(plt):
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})


def _err(ci):
    return [[ci["retained"] - ci["lo"]], [ci["hi"] - ci["retained"]]]


def _save(fig, out: Path):
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(out)


def tokens_bits(idx: Index, model: str, out: Path, plt):
    """x: bits per element per original visual token (kept fraction x KV bits); one line per KV bit-width."""
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, t in zip(axes, TASKS):
        for kv, bits, color, marker, name in (("full", 16, BLUE, "o", "16-bit KV, fewer tokens"),
                                             ("4", 4, ORANGE, "s", "4-bit KV, fewer tokens")):
            xs, ys, cis = [], [], []
            for keep, method in TOKENS:
                ci = idx.retained(model, t, Condition(kv=kv, token_keep=keep, token_method=method), "acc")
                if ci is None:
                    continue
                xs.append(keep * bits)
                ys.append(ci["retained"])
                cis.append(ci)
            ax.plot(xs, ys, color=color, lw=1.2, label=name)
            for x, ci in zip(xs, cis):
                ax.errorbar(x, ci["retained"], yerr=_err(ci), fmt=marker, color=color, ms=3.5, capsize=1.5, lw=0.8,
                            mec="white", mew=0.8)
        ax.axvline(4, color=MUTED, lw=0.6, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_title(TASK_NAMES[t])
        ax.axhline(1.0, color=MUTED, lw=0.5)
        ax.axhline(0.0, color=MUTED, lw=0.5, ls="--")
    for ax in axes[2:]:
        ax.set_xlabel("bits / element / visual token")
    for ax in axes[::2]:
        ax.set_ylabel("accuracy retained")
    axes[0].set_ylim(-0.05, 1.12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.3)
    _save(fig, out / "fig_tokens_bits.pdf")


def channels(idx: Index, model: str, out: Path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, t in zip(axes, TASKS):
        for i, (_, c) in enumerate(CHANNELS):
            ci = idx.retained(model, t, c, "acc")
            if ci is None:
                continue
            ax.errorbar(ci["retained"], i, xerr=_err(ci), fmt="o", color=BLUE, ms=3.5, capsize=1.5, lw=0.8)
        ax.axvline(1.0, color=MUTED, lw=0.5)
        ax.axvline(0.0, color=MUTED, lw=0.5, ls="--")
        ax.set_xlim(-0.15, 1.1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(TASK_NAMES[t])
    for ax in axes[2:]:
        ax.set_xlabel("accuracy retained")
    axes[0].set_yticks(range(len(CHANNELS)), [n for n, _ in CHANNELS])
    axes[0].invert_yaxis()
    fig.tight_layout(pad=0.3)
    _save(fig, out / "fig_channels.pdf")


def case(idx: Index, model: str, out: Path, plt):
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    for t in TASKS:
        color, marker = TASK_STYLE[t]
        pts = [(idx.retained(model, t, c, "acc"), idx.retained(model, t, c, "nll")) for c in TABLE2_ROWS]
        pts = [(a["retained"], n["retained"]) for a, n in pts if a and n]
        name = TASK_NAMES[t] + (" (case-mixed)" if t in NLL_CASE_MIXED else "")
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, s=16, label=name,
                   edgecolors="white", linewidths=0.6, zorder=3)
    ax.plot([-0.8, 1.2], [-0.8, 1.2], color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("accuracy retained")
    ax.set_ylabel("loss retained")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    _save(fig, out / "fig_case.pdf")


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--results", default="results/mac")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--allow-local", action="store_true")
    args = ap.parse_args(argv)
    _style(plt)
    idx = Index(load_rows(Path(args.results), args.allow_local))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (tokens_bits, channels, case):
        fn(idx, args.model, out, plt)
        plt.close("all")


if __name__ == "__main__":
    main()
