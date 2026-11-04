"""
Launches the pool-based active query selection sweep for IPL on MetaWorld locally.
Each run is a separate scripts/train.py process limited to one thread; finished runs are skipped.

Example:
    python scripts/run_active_sweep.py --out output/metaworld_active --jobs 8
    python scripts/run_active_sweep.py --out /tmp/smoke --total-steps 3000 --seeds 1 --tasks drawer-open --budgets 500
"""
import argparse
import copy
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
POOL_CONFIG = REPO / "configs/metaworld_active/ipl_iql_pool.yaml"
STATIC_CONFIG = REPO / "configs/metaworld/ipl_iql.yaml"

# name -> overrides on top of the pool config. "static" uses the original paper setup instead.
METHODS = {
    "static_e2": dict(),  # Paper baseline: first N pairs of the file, all labeled at step 0, ensemble 2.
    "uniform_e5": dict(query_strategy="uniform"),
    "std_e2": dict(query_strategy="std", ensemble_size=2),
    "std_e5": dict(query_strategy="std"),
    "bald_e5": dict(query_strategy="bald"),
    "policy_bald_e5": dict(query_strategy="policy_bald", query_kwargs=dict(eta=1.0)),
}


def build_config(method, task, budget, seed, total_steps):
    is_static = method.startswith("static")
    with open(STATIC_CONFIG if is_static else POOL_CONFIG) as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config["eval_env"] = f"mw_{task}-v2"
    config["seed"] = seed
    fb = config["dataset_kwargs"]["feedback_kwargs"]
    fb["path"] = f"datasets/metaworld/{task}_10000.npz"
    fb["capacity"] = budget
    if not is_static:
        overrides = dict(METHODS[method])
        if "ensemble_size" in overrides:
            config["network_kwargs"]["critic_kwargs"]["ensemble_size"] = overrides.pop("ensemble_size")
        config["alg_kwargs"].update(overrides)
        config["alg_kwargs"]["max_feedback"] = budget
        config["alg_kwargs"]["num_uniform_feedback"] = int(0.2 * budget)
    if total_steps is not None:
        config["trainer_kwargs"]["total_steps"] = total_steps
        config["trainer_kwargs"]["eval_freq"] = max(total_steps // 4, 1)
        if not is_static:
            # Spend the whole budget within the first half of training, as in the full config.
            num_rounds = budget // config["alg_kwargs"]["init_feedback_size"]
            config["alg_kwargs"]["reward_freq"] = max(total_steps // (2 * num_rounds), 1)
    return config


def run(job, out, device):
    method, task, budget, seed, total_steps = job
    path = Path(out) / task / f"{method}_b{budget}" / f"seed{seed}"
    if (path / "final_model.pt").exists():
        return f"skip {path}"
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "launch_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(build_config(method, task, budget, seed, total_steps), f, sort_keys=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", D4RL_SUPPRESS_IMPORT_ERROR="1")
    with open(path / "stdout.txt", "w") as log:
        ret = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", str(config_path), "--path", str(path), "--device", device],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return f"{'ok' if ret.returncode == 0 else 'FAIL(' + str(ret.returncode) + ')'} {path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="output/metaworld_active")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    parser.add_argument("--tasks", nargs="+", default=["drawer-open", "assembly"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--total-steps", type=int, default=None, help="Override for smoke tests.")
    args = parser.parse_args()

    for m in args.methods:
        assert m in METHODS, f"Unknown method {m}"
    # Order by seed first so partial sweeps still cover every configuration.
    jobs = [
        (m, t, b, s, args.total_steps)
        for s, t, b, m in itertools.product(args.seeds, args.tasks, args.budgets, args.methods)
    ]
    print(f"Launching {len(jobs)} runs with {args.jobs} parallel jobs", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for msg in pool.map(lambda j: run(j, args.out, args.device), jobs):
            print(msg, flush=True)
