"""
Trivial-reward controls for the two-phase sweep: how much of each task can be solved from the pool data alone?

Every condition trains on the same unlabeled pool transitions as phase 2 with --replay-pool:
  iql_zero           IQL with reward 0 (advantage-weighted behavior cloning of the pool)
  iql_random         IQL with a fixed U(0, 1) reward per transition
  ipl_random_labels  IPL with the replay pool and `budget` uniform pairs whose labels are replaced by coin flips
  ipl_uniform        the same pairs with their true labels (the uniform baseline without phase 1, used to screen tasks)
A task where these reach the success of IPL with real labels cannot tell selection methods apart.
Results go to <out>/<task>/seed<s>/<condition>/ and compare with phase2_replay/uniform of run_two_phase.py.

Example:
    python scripts/run_trivial_reward.py --out output/metaworld_trivial --jobs 9
    python scripts/run_trivial_reward.py --out /tmp/smoke --tasks drawer-open --seeds 1 --steps 500
"""
import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import run_two_phase  # noqa: E402
from run_two_phase import POOL_KEYS, pool_path, train, train_config  # noqa: E402

CONDITIONS = ("iql_zero", "iql_random", "ipl_random_labels", "ipl_uniform")


def iql_config(task, seed, total_steps, eval_freq, reward):
    config = train_config(task, None, seed, 2, total_steps, eval_freq)
    alg_kwargs = config["alg_kwargs"]
    config["alg"] = "IQL"
    config["alg_kwargs"] = {k: alg_kwargs[k] for k in ("tau", "beta", "expectile", "offline_steps", "random_steps")}
    replay_kwargs = config["dataset_kwargs"]["replay_kwargs"]
    if run_two_phase.DOMAIN == "robomimic":  # Every demo of the hdf5, as in the IPL runs
        assert reward == "zero", "robomimic supports only the zero-reward control"
        config["dataset"] = "RobomimicDataset"
        config["dataset_kwargs"] = dict(path=replay_kwargs["path"], batch_size=replay_kwargs["batch_size"],
                                        discount=config["dataset_kwargs"]["discount"], distributed=False, reward=reward)
        return config
    config["dataset"] = "SegmentPoolReplay"
    config["dataset_kwargs"] = dict(path=str(pool_path(task)), batch_size=replay_kwargs["batch_size"],
                                    discount=config["dataset_kwargs"]["discount"], reward=reward, reward_seed=seed)
    return config


def uniform_subset(task, budget, seed, path, random_labels):
    rng = np.random.default_rng(seed)
    with np.load(pool_path(task)) as data:
        idxs = rng.choice(len(data["label"]), size=budget, replace=False)
        subset = {k: data[k][idxs] for k in POOL_KEYS}
    if random_labels:
        subset["label"] = rng.integers(0, 2, size=budget).astype(subset["label"].dtype)
    np.savez(path, **subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="output/metaworld_trivial")
    parser.add_argument("--tasks", nargs="+",
                        default=["drawer-open", "button-press", "door-close", "plate-slide", "sweep-into"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS[:3]), choices=CONDITIONS)
    parser.add_argument("--budget", type=int, default=1000, help="Pairs for the ipl_ conditions")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--domain", type=str, default="metaworld", choices=list(run_two_phase.STATIC_CONFIGS))
    parser.add_argument("--pool-tag", type=str, default="", help="See run_two_phase.py, e.g. _human on robomimic")
    args = parser.parse_args()
    run_two_phase.DOMAIN, run_two_phase.POOL_TAG = args.domain, args.pool_tag
    eval_freq = min(args.eval_freq, args.steps)

    def run(job):
        task, seed, condition = job
        path = Path(args.out) / task / f"seed{seed}" / condition
        if condition.startswith("ipl_"):
            path.mkdir(parents=True, exist_ok=True)
            data = path / "labels.npz"
            uniform_subset(task, args.budget, seed, data, random_labels=condition == "ipl_random_labels")
            config = train_config(task, data, seed, 2, args.steps, eval_freq, replay_pool=True)
        else:
            config = iql_config(task, seed, args.steps, eval_freq, reward=condition[len("iql_"):])
        return train(path, config, args.device)

    jobs = list(itertools.product(args.tasks, args.seeds, args.conditions))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for msg in pool.map(run, jobs):
            print(msg, flush=True)
