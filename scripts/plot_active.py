"""
Summarizes and plots the pool-based active query sweep produced by scripts/run_active_sweep.py.
Writes summary_runs.csv, summary.csv (mean/std over seeds) and learning-curve PNGs to --out.

Example:
    python scripts/plot_active.py --sweep output/metaworld_active --out output/metaworld_active/plots
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order (validated palette); baselines are dashed so identity is not color-alone.
METHOD_STYLE = {
    "static_e2": ("Random, all labels upfront (paper), E=2", "#2a78d6", "--"),
    "static_e5": ("Random, all labels upfront, E=5", "#6250d6", "--"),
    "uniform_e2": ("Random, incremental, E=2", "#008300", ":"),
    "uniform_e5": ("Random, incremental, E=5", "#eb6834", ":"),
    "std_e2": ("Std disagreement, E=2", "#1baf7a", "-"),
    "std_e5": ("Std disagreement, E=5", "#eda100", "-"),
    "bald_e5": ("BALD (A1), E=5", "#e34948", "-"),
    "policy_bald_e5": ("Policy-weighted BALD (A2), E=5", "#e87ba4", "-"),
}
TASK_NAMES = {"drawer-open": "Drawer Open", "assembly": "Assembly", "button-press": "Button Press"}


def load(sweep):
    runs, curves = [], []
    for f in glob.glob(os.path.join(sweep, "*", "*", "seed*", "log.csv")):
        task, cfg, seed = f.split(os.sep)[-4:-1]
        method, budget = cfg.rsplit("_b", 1)
        d = pd.read_csv(f)
        e = d[["step", "eval/success"]].dropna()
        if len(e) == 0:
            continue
        runs.append(
            dict(
                task=task,
                method=method,
                budget=int(budget),
                seed=seed,
                final_steps=int(e["step"].iloc[-1]),
                last10=e["eval/success"].iloc[-10:].mean(),
                peak=e["eval/success"].max(),
                auc=e["eval/success"].mean(),
            )
        )
        curves.append(e.assign(task=task, method=method, budget=int(budget), seed=seed))
    return pd.DataFrame(runs), pd.concat(curves, ignore_index=True)


def plot_curves(curves, out, smooth=5):
    tasks = [t for t in TASK_NAMES if t in curves["task"].unique()]
    budgets = sorted(curves["budget"].unique())
    fig, axes = plt.subplots(
        len(tasks), len(budgets), figsize=(5.2 * len(budgets), 3.4 * len(tasks)), sharex=True, sharey=True, squeeze=False
    )
    for i, task in enumerate(tasks):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            sub = curves[(curves["task"] == task) & (curves["budget"] == budget)]
            for method, (label, color, ls) in METHOD_STYLE.items():
                m = sub[sub["method"] == method]
                if len(m) == 0:
                    continue
                per_seed = m.pivot_table(index="step", columns="seed", values="eval/success")
                per_seed = per_seed.rolling(smooth, min_periods=1).mean()
                mean, std = per_seed.mean(axis=1), per_seed.std(axis=1).fillna(0)
                ax.plot(mean.index / 1000, mean.values, color=color, ls=ls, lw=2, label=label)
                ax.fill_between(mean.index / 1000, mean - std, mean + std, color=color, alpha=0.12, lw=0)
            ax.set_title(f"{TASK_NAMES[task]} · {budget} labels", fontsize=11, loc="left", color="#0b0b0b")
            ax.grid(axis="y", color="#e5e4e0", lw=0.8)
            ax.set_ylim(-0.02, 1.02)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#b5b4ae")
            ax.tick_params(colors="#52514e", labelsize=9)
            if i == len(tasks) - 1:
                ax.set_xlabel("Training steps (k)", color="#52514e")
            if j == 0:
                ax.set_ylabel("Success rate", color="#52514e")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(out, "learning_curves.png")
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="output/metaworld_active")
    parser.add_argument("--out", type=str, default="output/metaworld_active/plots")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, curves = load(args.sweep)
    runs = runs[runs["final_steps"] >= runs["final_steps"].max()]  # Only finished runs
    curves = curves.merge(runs[["task", "method", "budget", "seed"]], on=["task", "method", "budget", "seed"])
    runs.to_csv(os.path.join(args.out, "summary_runs.csv"), index=False)
    summary = (
        runs.groupby(["task", "budget", "method"])
        .agg(
            seeds=("seed", "count"),
            last10_mean=("last10", "mean"),
            last10_std=("last10", "std"),
            peak_mean=("peak", "mean"),
            auc_mean=("auc", "mean"),
        )
        .round(3)
    )
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    print(summary.to_string())
    print("Saved", plot_curves(curves, args.out))
