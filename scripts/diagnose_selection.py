"""
Diagnoses which pairs each selection method picked in the two-phase sweep.

An "oracle" reward model is trained on 90% of the hidden pool labels (held-out accuracy is printed) and used to
estimate segment returns. For every selection it reports, relative to the pool:
  gap            mean |R2 - R1| of the selected pairs divided by the pool mean (1 = like uniform)
  max_pct        mean return percentile of the better segment of each pair
  small_gap_frac fraction of near-tied pairs (gap below the pool median)
and correlates the paired difference to uniform of these statistics with the paired success difference.
The oracle is used for analysis only; no selection method sees it.

Example:
    python scripts/diagnose_selection.py --sweep output/metaworld_two_phase
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from research.algs import proxy_reward  # noqa: E402

KEYS = ("obs_1", "obs_2", "action_1", "action_2")


def oracle_returns(task, seed=0):
    data = np.load(f"datasets/metaworld/{task}_10000.npz")
    perm = np.random.default_rng(seed).permutation(len(data["label"]))
    train, test = perm[: int(0.9 * len(perm))], perm[int(0.9 * len(perm)) :]
    labeled = {k: data[k][train] for k in KEYS + ("label",)}
    member = proxy_reward.train_proxy(labeled, ensemble_size=1, steps=6000, seed=seed)
    r1 = proxy_reward.segment_returns(member, data["obs_1"], data["action_1"])[0]
    r2 = proxy_reward.segment_returns(member, data["obs_2"], data["action_2"])[0]
    acc = ((r2 > r1) == (data["label"] > 0.5))[test].mean()
    print(f"{task}: oracle reward model held-out accuracy {acc:.3f}", flush=True)
    return r1, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="output/metaworld_two_phase")
    parser.add_argument("--tasks", nargs="+", default=["drawer-open", "button-press"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = os.path.join(args.sweep, "plots")
    os.makedirs(out, exist_ok=True)

    rows = []
    for task in args.tasks:
        r1, r2 = oracle_returns(task)
        all_returns = np.sort(np.concatenate([r1, r2]))
        gap, best = np.abs(r2 - r1), np.maximum(r1, r2)
        for f in glob.glob(os.path.join(args.sweep, task, "b*", "seed*", "selection", "*.npy")):
            method = os.path.basename(f)[:-4]
            if method == "warmup_idx":
                continue
            sel_dir = os.path.dirname(f)
            idx = np.setdiff1d(np.load(f), np.load(os.path.join(sel_dir, "warmup_idx.npy")))  # Selected part only
            budget, seed = sel_dir.split(os.sep)[-3:-1]
            rows.append(dict(task=task, budget=int(budget[1:]), seed=int(seed[4:]), method=method,
                             gap=gap[idx].mean() / gap.mean(),
                             max_pct=(np.searchsorted(all_returns, best[idx]) / len(all_returns)).mean(),
                             small_gap_frac=(gap[idx] < np.median(gap)).mean()))
    diag = pd.DataFrame(rows)
    diag.to_csv(os.path.join(out, "selection_diag.csv"), index=False)
    print(diag.groupby("method")[["gap", "max_pct", "small_gap_frac"]].mean().round(3).to_string())

    runs_path = os.path.join(out, "summary_runs.csv")  # Written by scripts/plot_two_phase.py
    if os.path.exists(runs_path):
        cell = ["task", "budget", "seed"]
        joined = diag.merge(pd.read_csv(runs_path), on=cell + ["method"])
        base = joined[joined["method"] == "uniform"][cell + ["last10", "gap", "max_pct"]]
        joined = joined.merge(base, on=cell, suffixes=("", "_uniform"))
        joined = joined[joined["method"] != "uniform"]
        d_success = joined["last10"] - joined["last10_uniform"]
        for col in ("gap", "max_pct"):
            rho, p = stats.spearmanr(joined[col] - joined[col + "_uniform"], d_success)
            print(f"Spearman(Δ{col}, Δsuccess) = {rho:.3f} (p = {p:.3f}, n = {len(joined)})")
