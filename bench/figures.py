"""Pareto and trust figures for the paper (M1 and M4).

    python3 bench/figures.py

log/bench-pareto-<task>.png   accuracy vs GPU latency and accuracy vs trainable
                              parameters, coloured by tier
log/bench-reliability-<task>.png, log/bench-coverage-<task>.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import pandas as pd                                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench.report import collect                                   # noqa: E402

COLORS = {"tier1-classical": "#8c8c8c", "tier2-classic-cnn": "#1f77b4",
          "tier3-efficient-cnn": "#2ca02c", "tier4-transformer": "#d62728",
          "tier5-foundation": "#9467bd", "phaseD-longtail": "#ff7f0e",
          "unlabelled": "#000000"}


def pareto_front(xs, ys):
    """lower x is better, higher y is better"""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i]))
    front, best = [], -1e18
    for i in order:
        if ys[i] > best:
            front.append(i)
            best = ys[i]
    return front


def panel(ax, df, xcol, xlabel, logx=True):
    d = df[df[xcol].notna() & df.test_acc.notna()]
    for tier, g in d.groupby("tier"):
        ax.scatter(g[xcol], g.test_acc, s=42, label=tier,
                   color=COLORS.get(tier, "#000000"), edgecolor="white", zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r.model[:22], (r[xcol], r.test_acc), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points", alpha=0.75)
    if len(d):
        f = pareto_front(d[xcol].tolist(), d.test_acc.tolist())
        s = d.iloc[f].sort_values(xcol)
        ax.plot(s[xcol], s.test_acc, "--", color="black", lw=1, alpha=0.6, zorder=2,
                label="Pareto front")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy")
    ax.grid(alpha=0.25, zorder=0)


def main():
    df = collect()
    for task in ("5class", "binary"):
        sub = df[df.task == task]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
        panel(axes[0], sub, "gpu_ms_b1", "GPU latency @ batch 1 (ms, log)")
        panel(axes[1], sub, "trainable_M", "trainable parameters (M, log)")
        panel(axes[2], sub, "gflops", "GFLOPs @ 224 (log)")
        axes[0].set_title(f"{task}: accuracy vs runtime")
        axes[1].set_title(f"{task}: accuracy vs trainable parameters")
        axes[2].set_title(f"{task}: accuracy vs compute")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"KAUHC V2 benchmark - accuracy / runtime / complexity trade-off "
                     f"({task})", fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        p = os.path.join(C.LOG_ROOT, f"bench-pareto-{task}.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[written] {p}")

    rel = os.path.join(C.LOG_ROOT, "bench-reliability.csv")
    cov = os.path.join(C.LOG_ROOT, "bench-coverage.csv")
    if not (os.path.exists(rel) and os.path.exists(cov)):
        return
    R, V = pd.read_csv(rel), pd.read_csv(cov)
    for task in ("5class", "binary"):
        sub = df[(df.task == task)].sort_values("test_acc", ascending=False)
        winners = (sub.groupby("tier").head(1).model.tolist())
        if not winners:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect")
        for m in winners:
            r = R[(R.task == task) & (R.model == m) & (R.calibration == "after")].dropna()
            if len(r):
                axes[0].plot(r.mean_confidence, r.accuracy, "o-", ms=4, label=m[:24])
            v = V[(V.task == task) & (V.model == m) & (V.calibration == "after")]
            if len(v):
                axes[1].plot(v.coverage, v.accuracy, "o-", ms=4, label=m[:24])
        axes[0].set_xlabel("confidence (after temperature scaling)")
        axes[0].set_ylabel("accuracy")
        axes[0].set_title(f"Reliability, tier winners ({task})")
        axes[1].set_xlabel("coverage (fraction of test frames kept)")
        axes[1].set_ylabel("accuracy on kept frames")
        axes[1].set_title(f"Selective prediction ({task})")
        axes[1].invert_xaxis()
        for a in axes:
            a.grid(alpha=0.25)
            a.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(C.LOG_ROOT, f"bench-trust-{task}.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[written] {p}")


if __name__ == "__main__":
    main()
