"""
Creates a MetaWorld v3 comparison pool in the format of datasets/metaworld/<task>_10000.npz.

Port of create_metaworld_offline_dataset.py + create_comparison_dataset.py to MetaWorld v3 (gymnasium API) that
keeps the original data recipe:
  expert       scripted policy acting on a goal-observable copy of the same task, plus Gaussian noise (epsilon)
  within_env   the same, but the policy copy is reset independently (wrong goal / object positions)
  random       uniform random actions
  cross_env    the scripted policy of a random other task, plus Gaussian noise
Episodes end after 25 successful steps or at the time limit. Pairs of 25-step segments are sampled uniformly from
all windows inside episodes and labeled by the discounted dense-reward return (label 1 if segment 2 is better).

Example:
    python scripts/create_metaworld_v3_pool.py --env door-open --out datasets/metaworld/door-open_10000.npz
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from metaworld import policies  # noqa: E402
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as OBSERVABLE  # noqa: E402

from research.envs.metaworld import SawyerEnv  # noqa: E402


def observable_env(task):
    return OBSERVABLE[task + "-v3-goal-observable"]()


def scripted_policy(task):
    name = "".join(s.capitalize() for s in task.split("-")).replace("PegInsert", "PegInsertion")
    return vars(policies)["Sawyer" + name + "V3Policy"]()


def collect_episode(env, policy_env, policy, epsilon, noise_type, sync_goal):
    obs = env.reset()
    if sync_goal:  # Put the policy copy in the same task instance as the data env.
        policy_env._last_rand_vec = env.unwrapped._env._last_rand_vec.copy()
    policy_env._freeze_rand_vec = sync_goal
    policy_obs, _ = policy_env.reset()
    observations, actions, rewards, successes = [], [], [], 0
    success_steps = 25
    for _ in range(env._max_episode_steps):
        if noise_type == "random":
            action = env.action_space.sample()
        else:
            action = policy.get_action(policy_obs) + epsilon * np.random.randn(env.action_space.shape[0])
        action = np.clip(action, -1 + 1e-5, 1 - 1e-5)
        next_obs, reward, done, info = env.step(action)
        policy_obs, _, _, _, _ = policy_env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs
        if info["success"]:
            successes += 1
            success_steps -= 1
        if done or success_steps == 0 or policy_env.curr_path_length >= policy_env.max_path_length:
            break
    return np.array(observations, np.float32), np.array(actions, np.float32), np.array(rewards, np.float32), successes > 0


def make_pairs(episodes, num_pairs, segment_size, discount, rng):
    windows = [(i, s) for i, (o, _, _, _) in enumerate(episodes) for s in range(len(o) - segment_size + 1)]
    picks = rng.choice(len(windows), size=2 * num_pairs, replace=True)
    weights = discount ** np.arange(segment_size)
    segs = {"obs": [], "action": [], "return": []}
    for w in picks:
        i, s = windows[w]
        o, a, r, _ = episodes[i]
        segs["obs"].append(o[s : s + segment_size])
        segs["action"].append(a[s : s + segment_size])
        segs["return"].append((r[s : s + segment_size] * weights).sum())
    obs, action, ret = np.array(segs["obs"]), np.array(segs["action"]), np.array(segs["return"])
    return dict(obs_1=obs[:num_pairs], obs_2=obs[num_pairs:], action_1=action[:num_pairs], action_2=action[num_pairs:],
                label=(ret[:num_pairs] < ret[num_pairs:]).astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="MetaWorld task without version, like door-open")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--expert-ep", type=int, default=100)
    parser.add_argument("--within-env-ep", type=int, default=100)
    parser.add_argument("--random-ep", type=int, default=100)
    parser.add_argument("--cross-env-ep", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1.0, help="Std of the Gaussian action noise")
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--segment-size", type=int, default=25)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert not os.path.exists(args.out), "Refusing to overwrite " + args.out
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = SawyerEnv(args.env + "-v2", seed=False)  # Same wrapper and observation space as evaluation.
    own_env, own_policy = observable_env(args.env), scripted_policy(args.env)
    tasks = [k[: -len("-v3-goal-observable")] for k in OBSERVABLE if not k.startswith(args.env + "-v3")]
    episodes, stats = [], {}
    plan = [("expert", args.expert_ep), ("within_env", args.within_env_ep), ("random", args.random_ep),
            ("cross_env", args.cross_env_ep)]
    for kind, count in plan:
        for _ in range(count):
            if kind == "cross_env":
                other = random.choice(tasks)
                policy_env, policy = observable_env(other), scripted_policy(other)
            else:
                policy_env, policy = own_env, own_policy
            ep = collect_episode(env, policy_env, policy, args.epsilon, "random" if kind == "random" else "gaussian",
                                 sync_goal=kind == "expert")
            episodes.append(ep)
            stats.setdefault(kind, []).append((len(ep[0]), ep[3], ep[2].sum()))
        lengths, succ, ret = map(np.array, zip(*stats[kind]))
        print(f"{kind}: {count} ep, mean length {lengths.mean():.0f}, success {succ.mean():.2f}, "
              f"return {ret.mean():.0f}", flush=True)

    pool = make_pairs(episodes, args.num_pairs, args.segment_size, args.discount, np.random.default_rng(args.seed))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **pool)
    print(f"Saved {args.out}: {len(pool['label'])} pairs from {sum(len(e[0]) for e in episodes)} transitions, "
          f"label mean {pool['label'].mean():.2f}")
