"""
Two-phase (one-shot) active query selection for IPL on MetaWorld.

For each (task, budget, seed):
  1. Phase 1: train IPL (ensemble 5) on a uniform warm-start set of warmup_frac * budget pairs from the pool.
  2. Selection: score the rest of the pool once with the phase 1 model and pick the remaining labels in one batch,
     once per selection method (see research/algs/batch_selection.py).
  3. Phase 2: train IPL from scratch on warm-start + selected pairs with the paper's static setup (ensemble 2).
Phase 2 with the "uniform" method is the paper's random-query baseline, so methods differ only in which pairs they pick.

Example:
    python scripts/run_two_phase.py --out output/metaworld_two_phase --jobs 9
    python scripts/run_two_phase.py --out /tmp/smoke --tasks drawer-open --budgets 250 --seeds 1 \
        --phase1-steps 500 --phase2-steps 500
    python scripts/run_two_phase.py --domain robomimic --pool-tag _human --tasks lift can --budgets 100 250 \
        --out output/robomimic_two_phase --replay-pool
On Robomimic the pools come from scripts/make_robomimic_pool.py and the replay data is always every demo of the hdf5.
"""
import argparse
import copy
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).parent.parent
sys.path.append(str(REPO))
# Importing research sets MUJOCO_GL in this process; children must start from the original environment.
BASE_ENV = dict(os.environ)

STATIC_CONFIGS = dict(metaworld=REPO / "configs/metaworld/ipl_iql.yaml", robomimic=REPO / "configs/robomimic/ipl_iql.yaml")
POOL_KEYS = ("obs_1", "obs_2", "action_1", "action_2", "label")
DOMAIN = "metaworld"  # Set by --domain
POOL_TAG = ""  # Set by --pool-tag, e.g. "_bt20" for a pool written by scripts/make_noisy_pool.py
ROBOMIMIC_DATASET_TYPE = "mh"
ROBOMIMIC_HORIZON = 400  # robomimic's evaluation horizon for Lift and Can


def pool_path(task):
    if DOMAIN == "robomimic":  # POOL_TAG is "_human" or "_<n>" (dense-oracle labels)
        return REPO / f"datasets/robomimic_pool/{task}_{ROBOMIMIC_DATASET_TYPE}{POOL_TAG}.npz"
    return REPO / f"datasets/metaworld/{task}_10000{POOL_TAG}.npz"


def robomimic_hdf5(task):
    return str(REPO / f"datasets/robomimic/{task}/{ROBOMIMIC_DATASET_TYPE}/low_dim_v15.hdf5")


def pool_size(task):
    with np.load(pool_path(task)) as data:
        return len(data["label"])


def save_subset(task, idxs, path):
    with np.load(pool_path(task)) as data:
        np.savez(path, **{k: data[k][idxs] for k in POOL_KEYS})


def train_config(task, data_path, seed, ensemble_size, total_steps, eval_freq, replay_pool=False):
    with open(STATIC_CONFIGS[DOMAIN]) as f:
        config = copy.deepcopy(yaml.safe_load(f))
    config["seed"] = seed
    config["network_kwargs"]["critic_kwargs"]["ensemble_size"] = ensemble_size
    fb = config["dataset_kwargs"]["feedback_kwargs"]
    fb["path"] = str(data_path)
    fb["capacity"] = None
    config["trainer_kwargs"]["total_steps"] = total_steps
    config["trainer_kwargs"]["eval_freq"] = eval_freq
    if DOMAIN == "robomimic":
        # The paper's setup for the 500-query mh datasets. The replay data is every demo, labeled or not,
        # so robomimic runs always match the --replay-pool variant of MetaWorld.
        config["eval_env_kwargs"] = dict(path=robomimic_hdf5(task), horizon=ROBOMIMIC_HORIZON)
        config["dataset_kwargs"]["replay_kwargs"]["path"] = robomimic_hdf5(task)
        fb["segment_size"] = 100
        fb["subsample_size"] = 64
        return config
    config["eval_env"] = f"mw_{task}-v2"
    if replay_pool:  # V, the policy and the chi^2 term also train on the unlabeled pool.
        config["dataset_kwargs"]["replay_class"] = "SegmentPoolReplay"
        config["dataset_kwargs"]["replay_kwargs"]["path"] = str(pool_path(task))
    return config


def train(path, config, device):
    path = Path(path)
    if (path / "final_model.pt").exists():
        return f"skip {path}"
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "launch_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    env = dict(BASE_ENV, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", D4RL_SUPPRESS_IMPORT_ERROR="1")
    with open(path / "stdout.txt", "w") as log:
        ret = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", str(config_path), "--path", str(path), "--device", device],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return f"{'ok' if ret.returncode == 0 else 'FAIL(' + str(ret.returncode) + ')'} {path}"


def select_multiround(method, task, pool, warmup, candidates, k, seed, rng, eta):
    """Chooses k pairs in equal chunks. Before each chunk the proxy is retrained on the warm-start labels plus the labels
    of every pair chosen so far (read from the pool, as a teacher would answer them), and the chunk is chosen from the
    remaining candidates. Round 0 uses the same proxy seed as the one-shot methods. Returns pool indices."""
    from research.algs import batch_selection, proxy_reward

    base, rounds = batch_selection.MULTIROUND[method]
    with np.load(pool_path(task)) as data:
        labels = data["label"]
    chosen = np.zeros(0, dtype=int)
    for r in range(rounds):
        revealed = np.concatenate([warmup, chosen])
        labeled = {key: v[revealed] for key, v in pool.items()}
        labeled["label"] = labels[revealed]
        members = proxy_reward.train_proxy(labeled, seed=seed + 1000 * r)
        rest = np.setdiff1d(candidates, chosen)
        stats = {"probs": np.zeros((1, len(rest)))}  # select() only reads the number of candidates from it
        for i in (1, 2):
            stats[f"proxy_return_{i}"] = proxy_reward.segment_returns(members, pool[f"obs_{i}"][rest], pool[f"action_{i}"][rest])
        size = k * (r + 1) // rounds - len(chosen)
        chosen = np.concatenate([chosen, rest[batch_selection.select(base, stats, size, rng, eta=eta)]])
    return chosen


def select(run_dir, task, budget, seed, methods, eta):
    """Runs every selection method with the phase 1 model. Writes selection/<method>.npz and .npy (pool indices)."""
    import torch

    from research.algs import batch_selection
    from research.utils.config import Config

    run_dir = Path(run_dir)
    sel_dir = run_dir / "selection"
    todo = [m for m in methods if not (sel_dir / f"{m}.npz").exists()]
    if len(todo) == 0:
        return f"skip selection {run_dir}"
    torch.set_num_threads(1)

    config = Config.load(str(run_dir / "phase1")).parse()
    config["checkpoint"] = str(run_dir / "phase1" / "final_model.pt")
    model = config.get_model(device="cpu")
    model.setup_train_dataset()  # Provides the discount used to compute the implicit reward.
    model.eval()

    warmup = np.load(sel_dir / "warmup_idx.npy")
    with np.load(pool_path(task)) as data:
        pool = {k: data[k] for k in POOL_KEYS if k != "label"}
    lim = 1 - 1e-5  # Match the clipping applied by PairwiseComparisonDataset.
    pool["action_1"] = np.clip(pool["action_1"], -lim, lim)
    pool["action_2"] = np.clip(pool["action_2"], -lim, lim)
    candidates = np.setdiff1d(np.arange(len(pool["obs_1"])), warmup)
    stats = model.pair_statistics(
        {k: v[candidates] for k, v in pool.items()},
        with_policy=any(m in batch_selection.NEEDS_POLICY for m in todo),
        with_features=any(m in batch_selection.NEEDS_FEATURES for m in todo),
    )
    if any(m in batch_selection.NEEDS_RAW for m in todo):
        raw = batch_selection.pair_embedding(pool["obs_1"], pool["obs_2"], pool["action_1"], pool["action_2"])
        stats["raw"], stats["raw_labeled"] = raw[candidates], raw[warmup]
    if any(m in batch_selection.NEEDS_PROXY for m in todo):
        from research.algs import proxy_reward

        with np.load(sel_dir / "warmup.npz") as data:
            labeled = {k: data[k] for k in POOL_KEYS}  # Only the warm-start labels are revealed.
        members = proxy_reward.train_proxy(labeled, seed=seed)
        stats["proxy_probs"] = proxy_reward.proxy_probs(members, {k: v[candidates] for k, v in pool.items()})
        for i in (1, 2):
            stats[f"proxy_return_{i}"] = proxy_reward.segment_returns(
                members, pool[f"obs_{i}"][candidates], pool[f"action_{i}"][candidates]
            )
    k = budget - len(warmup)
    for m in todo:
        rng = np.random.default_rng(seed)
        if m in batch_selection.MULTIROUND:
            chosen = select_multiround(m, task, pool, warmup, candidates, k, seed, rng, eta)
        else:
            chosen = candidates[batch_selection.select(m, stats, k, rng, eta=eta)]
        assert len(np.intersect1d(chosen, warmup)) == 0 and len(np.unique(chosen)) == k
        idxs = np.concatenate([warmup, chosen])
        np.save(sel_dir / f"{m}.npy", idxs)
        save_subset(task, idxs, sel_dir / f"{m}.npz")
    return f"ok selection {run_dir} ({', '.join(todo)})"


if __name__ == "__main__":
    from research.algs.batch_selection import METHODS, ROUND2_METHODS, ROUND3_METHODS, ROUND9_METHODS

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="output/metaworld_two_phase")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="Also accepts " + " ".join(ROUND2_METHODS + ROUND3_METHODS + ROUND9_METHODS))
    parser.add_argument("--tasks", nargs="+", default=["drawer-open", "button-press"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[250, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--warmup-frac", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--phase1-steps", type=int, default=30000)
    parser.add_argument("--phase2-steps", type=int, default=200000)
    parser.add_argument("--eval-freq", type=int, default=2500, help="Phase 2 evaluation period")
    parser.add_argument("--domain", type=str, default="metaworld", choices=list(STATIC_CONFIGS))
    parser.add_argument("--replay-pool", action="store_true",
                        help="Phase 2 also trains V, the policy and chi^2 on the unlabeled pool (writes phase2_replay/)")
    parser.add_argument("--pool-tag", type=str, default="",
                        help="Use datasets/metaworld/<task>_10000<tag>.npz, e.g. noisy labels from make_noisy_pool.py")
    args = parser.parse_args()
    POOL_TAG = args.pool_tag
    DOMAIN = args.domain
    if DOMAIN == "robomimic":
        assert args.replay_pool, "robomimic always trains on every demo; pass --replay-pool so outputs go to phase2_replay/"
    for t in args.tasks:
        assert pool_path(t).exists(), f"missing pool {pool_path(t)}"

    # Order by seed first so partial sweeps still cover every configuration.
    cells = [(t, b, s) for s, t, b in itertools.product(args.seeds, args.tasks, args.budgets)]
    run_dir = lambda t, b, s: Path(args.out) / t / f"b{b}" / f"seed{s}"  # noqa: E731

    def phase1(cell):
        t, b, s = cell
        sel_dir = run_dir(t, b, s) / "selection"
        sel_dir.mkdir(parents=True, exist_ok=True)
        if not (sel_dir / "warmup.npz").exists():
            rng = np.random.default_rng(1000 * s + b)
            warmup = rng.choice(pool_size(t), size=int(args.warmup_frac * b), replace=False)
            np.save(sel_dir / "warmup_idx.npy", warmup)
            save_subset(t, warmup, sel_dir / "warmup.npz")
        config = train_config(t, sel_dir / "warmup.npz", s, 5, args.phase1_steps, max(args.phase1_steps // 3, 1))
        return train(run_dir(t, b, s) / "phase1", config, args.device)

    def selection(cell):
        t, b, s = cell
        return select(run_dir(t, b, s), t, b, s, args.methods, args.eta)

    def phase2(job):
        (t, b, s), m = job
        data = run_dir(t, b, s) / "selection" / f"{m}.npz"
        config = train_config(t, data, s, 2, args.phase2_steps, args.eval_freq, replay_pool=args.replay_pool)
        return train(run_dir(t, b, s) / ("phase2_replay" if args.replay_pool else "phase2") / m, config, args.device)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        print(f"Phase 1: {len(cells)} runs", flush=True)
        for msg in pool.map(phase1, cells):
            print(msg, flush=True)
        print("Selection", flush=True)
        for msg in pool.map(selection, cells):
            print(msg, flush=True)
        jobs = [(c, m) for c in cells for m in args.methods]
        print(f"Phase 2: {len(jobs)} runs", flush=True)
        for msg in pool.map(phase2, jobs):
            print(msg, flush=True)
