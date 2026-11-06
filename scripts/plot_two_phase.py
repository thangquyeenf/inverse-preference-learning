"""
Summarizes the two-phase (one-shot) selection sweep produced by scripts/run_two_phase.py.
Writes summary_runs.csv, summary.csv and paired.csv (difference to uniform, paired by seed) plus figures to --out.

Example:
    python scripts/plot_two_phase.py --sweep output/metaworld_two_phase --out output/metaworld_two_phase/plots
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Validated palette slots; each figure shows at most 6 colored methods, and density_bald / mix_bald reuse the
# std / badge hues (with distinct markers) because they never share a figure. Uniform is gray and dashed.
METHOD_STYLE = {
    "uniform": ("Uniform (paper baseline)", "#52514e", "--", "o"),
    "std": ("Std disagreement", "#2a78d6", "-", "o"),
    "bald": ("BALD (A1)", "#e34948", "-", "o"),
    "bald_kcenter": ("BALD + k-center", "#eda100", "-", "s"),
    "badge": ("BADGE", "#1baf7a", "-", "^"),
    "policy_bald_kcenter": ("Policy BALD + k-center (A2)", "#e87ba4", "-", "D"),
    "proxy_bald": ("Proxy-RM BALD (ours)", "#6250d6", "-", "o"),
    "proxy_density_bald": ("Proxy-RM BALD × density (ours)", "#008300", "-", "s"),
    "typiclust": ("TypiClust", "#eb6834", "-", "^"),
    "density_bald": ("BALD × density", "#2a78d6", "-", "D"),
    "mix_bald": ("½ uniform + ½ BALD", "#1baf7a", "-", "v"),
    "proxy_top": ("Proxy-RM top return", "#eda100", "-", "P"),
    "mix_proxy_top": ("½ uniform + ½ proxy top", "#e87ba4", "-", "X"),
}
TASK_NAMES = {"drawer-open": "Drawer Open", "button-press": "Button Press"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def load(sweep, total_steps):
    runs, curves = [], []
    for f in glob.glob(os.path.join(sweep, "*", "b*", "seed*", "phase2", "*", "log.csv")):
        task, budget, seed, _, method = f.split(os.sep)[-6:-1]
        e = pd.read_csv(f)[["step", "eval/success"]].dropna()
        if len(e) == 0 or e["step"].iloc[-1] < 0.975 * total_steps:
            continue  # Unfinished run
        info = dict(task=task, budget=int(budget[1:]), seed=int(seed[4:]), method=method)
        runs.append(dict(info, last10=e["eval/success"].iloc[-10:].mean(), peak=e["eval/success"].max(),
                         auc=e["eval/success"].mean()))
        curves.append(e.assign(**info))
    return pd.DataFrame(runs), pd.concat(curves, ignore_index=True)


def paired_vs_uniform(runs, metric):
    """Mean difference to uniform over matched (task, budget, seed) cells, with a 95% t-interval."""
    wide = runs.pivot_table(index=["task", "budget", "seed"], columns="method", values=metric)
    rows = []
    groups = list(wide.groupby(level=["task", "budget"]))
    groups += [(("all", b), sub) for b, sub in wide.groupby(level="budget")] + [(("all", "all"), wide)]
    for group, sub in groups:
        for method in [m for m in METHOD_STYLE if m in wide.columns and m != "uniform"]:
            diff = (sub[method] - sub["uniform"]).dropna()
            if len(diff) < 2:
                continue
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std() / np.sqrt(len(diff))
            p = stats.wilcoxon(diff).pvalue if len(diff) >= 6 and (diff != 0).any() else np.nan
            rows.append(dict(task=group[0], budget=group[1], method=method, n=len(diff), delta=diff.mean(),
                             lo=diff.mean() - half, hi=diff.mean() + half, win_rate=(diff > 0).mean(), wilcoxon_p=p))
    return pd.DataFrame(rows)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_budget(runs, out, methods, metric="last10", name="success_vs_budget.png"):
    """Success vs. budget, one panel per task, mean ± standard error over seeds."""
    tasks = [t for t in TASK_NAMES if t in runs["task"].unique()]
    methods = [m for m in methods if m in runs["method"].unique()]
    budgets = sorted(runs["budget"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 3.6), sharey=True, squeeze=False)
    offsets = np.linspace(-0.12, 0.12, len(methods))
    for ax, task in zip(axes[0], tasks):
        for off, method in zip(offsets, methods):
            label, color, ls, marker = METHOD_STYLE[method]
            g = runs[(runs["task"] == task) & (runs["method"] == method)].groupby("budget")[metric]
            mean, se = g.mean().reindex(budgets), g.sem().reindex(budgets)
            x = np.arange(len(budgets)) + off
            ax.errorbar(x, mean, yerr=se, color=color, ls=ls, lw=2, marker=marker, ms=8, capsize=3, label=label,
                        mec="#fcfcfb", mew=1.5)
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("Label budget (pairs)", color=MUTED)
        ax.set_title(TASK_NAMES[task], fontsize=11, loc="left", color=INK)
        _style(ax)
    axes[0, 0].set_ylabel("Success (last 10 evals)" if metric == "last10" else metric, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(handles), 4)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06 + 0.05 * ((len(handles) - 1) // ncol), 1, 1))
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def plot_paired(paired, out, methods, name="paired_vs_uniform.png"):
    """Difference to uniform (paired by seed, pooled over tasks) with 95% CI: one panel per budget, plus all budgets."""
    sub = paired[paired["task"] == "all"]
    budgets = sorted(b for b in sub["budget"].unique() if b != "all") + ["all"]
    methods = [m for m in methods if m in sub["method"].unique() and m != "uniform"]
    lo, hi = sub[sub["method"].isin(methods)]["lo"].min(), sub[sub["method"].isin(methods)]["hi"].max()
    fig, axes = plt.subplots(1, len(budgets), figsize=(2.9 * len(budgets) + 1.8, 0.5 * len(methods) + 1.5),
                             sharey=True, squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        for i, method in enumerate(methods):
            _, color, _, marker = METHOD_STYLE[method]
            row = sub[(sub["method"] == method) & (sub["budget"] == budget)].iloc[0]
            ax.errorbar(row["delta"], i, xerr=[[row["delta"] - row["lo"]], [row["hi"] - row["delta"]]], color=color,
                        marker=marker, ms=8, capsize=3, lw=2, mec="#fcfcfb", mew=1.5)
            ax.text(hi + 0.02 * (hi - lo), i, f"{row['delta']:+.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.22 * (hi - lo))
        title = "All budgets" if budget == "all" else f"{budget} labels"
        n = int(sub[sub["budget"] == budget]["n"].iloc[0])
        ax.set_title(f"{title} (n={n})", fontsize=10.5, loc="left", color=INK)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(range(len(methods)), [METHOD_STYLE[m][0] for m in methods])
    axes[0, 0].invert_yaxis()
    fig.supxlabel("Success difference to uniform, paired by seed, both tasks (mean, 95% CI)", color=MUTED, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="output/metaworld_two_phase")
    parser.add_argument("--out", type=str, default="output/metaworld_two_phase/plots")
    parser.add_argument("--total-steps", type=int, default=200000)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, curves = load(args.sweep, args.total_steps)
    runs.to_csv(os.path.join(args.out, "summary_runs.csv"), index=False)
    summary = (
        runs.groupby(["task", "budget", "method"])
        .agg(seeds=("seed", "count"), last10_mean=("last10", "mean"), last10_std=("last10", "std"),
             auc_mean=("auc", "mean"), auc_std=("auc", "std"))
        .round(3)
    )
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    paired = pd.concat([paired_vs_uniform(runs, "last10").assign(metric="last10"),
                        paired_vs_uniform(runs, "auc").assign(metric="auc")])
    paired.round(4).to_csv(os.path.join(args.out, "paired.csv"), index=False)
    print(summary.to_string())
    print(paired[(paired["task"] == "all") & (paired["metric"] == "last10")].round(3).to_string(index=False))

    round1 = ["uniform", "std", "bald", "bald_kcenter", "badge", "policy_bald_kcenter"]
    round2 = ["uniform", "bald", "proxy_bald", "proxy_density_bald", "density_bald", "typiclust", "mix_bald"]
    print("Saved", plot_budget(runs, args.out, round1, name="round1_success_vs_budget.png"))
    print("Saved", plot_paired(paired[paired["metric"] == "last10"], args.out, round1, name="round1_paired.png"))
    if any(m in runs["method"].unique() for m in round2[2:]):
        print("Saved", plot_budget(runs, args.out, round2, name="round2_success_vs_budget.png"))
        print("Saved", plot_paired(paired[paired["metric"] == "last10"], args.out, round2, name="round2_paired.png"))
