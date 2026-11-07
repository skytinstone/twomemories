"""Figure 1 (overview): the two-stage prefill, the intervention point and the two memories.

  python paper/make_overview.py      # writes paper/figures/overview.{pdf,png}
Numbers in the figure are cache sizes of Qwen3.5-4B/9B (paper Table 1), not results.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
KV, ST, IMG, TXT, RED, GREY = "#e07b39", "#3d6fb6", "#f3c26b", "#d9dde3", "#c0392b", "#555555"


def box(ax, x, y, w, h, fc, ec="none", lw=1.0, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, color=GREY, ls="-", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(-0.6, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # token timeline
    y0, h = 3.55, 0.42
    segs = [("prefix $P$", 0.2, 1.0, TXT), ("image span $\\mathcal{V}$  ($n_v\\leq1280$ tokens)", 1.2, 4.4, IMG),
            ("question $Q$", 5.6, 1.9, TXT), ("answer:\nNLL, greedy", 7.5, 1.6, "#eef0f3")]
    for label, x, w, c in segs:
        box(ax, x, y0, w, h, c)
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=8)
    for i in range(22):   # visual token ticks
        ax.add_patch(Rectangle((1.3 + i * 0.19, y0 + 0.05), 0.12, 0.06, fc="#d99b2b", ec="none"))
    ax.annotate("", xy=(0.2, 4.15), xytext=(5.6, 4.15), arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(2.9, 4.24, "stage 1: unaltered computation", ha="center", fontsize=7.5, color=GREY)
    ax.annotate("", xy=(5.6, 4.15), xytext=(7.5, 4.15), arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(6.55, 4.24, "stage 2 reads $\\phi(\\mathcal{M}_e)$", ha="center", fontsize=7.5, color=GREY)

    # intervention point
    ax.plot([5.6, 5.6], [0.35, 3.5], color=RED, lw=1.4, ls=(0, (4, 2)))
    ax.text(5.65, 3.32, "$e$: intervene once, $\\phi$", color=RED, fontsize=8, ha="left", va="top")

    # KV cache memory
    box(ax, 0.2, 1.75, 5.2, 1.3, "#fdf1e8", ec=KV, lw=1.2)
    ax.text(0.32, 2.88, "KV cache  ($L_a=8$ attention layers)", fontsize=8.5, color=KV, weight="bold", va="center")
    for i in range(26):
        c = KV if 1.2 <= 0.35 + i * 0.19 < 5.5 else "#f2c7a8"
        ax.add_patch(Rectangle((0.35 + i * 0.19, 1.95), 0.15, 0.55, fc=c, ec="none", alpha=0.85))
    ax.text(0.32, 1.84, "one row per token; visual rows $K_{\\mathcal{V}},V_{\\mathcal{V}}$: 32.8 kB/token (bf16)",
            fontsize=6.8, color=GREY, va="center")
    ax.text(5.75, 2.55, "$f\\in\\{\\mathrm{id},Q_4,N_4,0\\}$\non visual rows only", fontsize=7.5, color=KV, va="center")

    # recurrent state memory
    box(ax, 0.2, 0.25, 5.2, 1.3, "#e9f0fa", ec=ST, lw=1.2)
    ax.text(0.32, 1.38, "Recurrent state  ($L_g=24$ GDN layers)", fontsize=8.5, color=ST, weight="bold", va="center")
    for r_ in range(3):
        for c_ in range(3):
            ax.add_patch(Rectangle((0.4 + c_ * 0.2, 0.42 + r_ * 0.2), 0.16, 0.16, fc=ST, ec="none", alpha=0.8))
    ax.text(1.15, 0.72, "$S^{\\ell}_e\\in\\mathbb{R}^{32\\times128\\times128}$ per layer, written by every token\n"
                        "fixed 50.3 MB (fp32), not indexed by token", fontsize=6.8, color=GREY, va="center")
    ax.text(5.75, 1.0, "$g\\in\\{\\mathrm{id},Q_8,Q_6,Q^{\\mathrm{EF}}_8,Q^{\\mathrm{EF}}_6,N_8,0\\}$\n"
                       "conv cache (last 3 tokens) kept", fontsize=7.5, color=ST, va="center")

    # writes from the image span into the two memories
    arrow(ax, (3.4, 3.53), (3.4, 3.06), color=KV)
    arrow(ax, (1.3, 3.53), (0.24, 1.2), color=ST, rad=0.55)   # around the KV box: the state is written directly
    # reads from stage 2
    arrow(ax, (6.5, 3.53), (5.42, 2.4), color=KV, ls="--", rad=0.15)
    arrow(ax, (6.7, 3.53), (5.42, 0.9), color=ST, ls="--", rad=0.25)
    ax.text(8.6, 1.75, "retained\n$\\mathcal{R}=\\dfrac{\\bar N_{\\mathrm{blind}}-\\bar N_c}"
                       "{\\bar N_{\\mathrm{blind}}-\\bar N_{\\mathrm{full}}}$", fontsize=8, ha="center", va="center")

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "overview.png", dpi=220, bbox_inches="tight")
    print(OUT / "overview.pdf")


if __name__ == "__main__":
    main()
