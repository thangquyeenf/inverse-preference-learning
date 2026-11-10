"""
Builds segment-pair pools for Robomimic in the format of the MetaWorld two-phase sweep (obs_1, obs_2, action_1, action_2, label).

Every pool is taken from one robomimic low_dim hdf5 (v1.5 by default, i.e. robosuite 1.5 observations) flattened over demos:
  <task>_human.npz   The Preference Transformer human pairs, located in the hdf5 by their action sequences, with the
                     observations re-read from the hdf5 (robosuite 1.5 changed the sign / layout of the object-state
                     features, so PT's stored observations do not match the installed environment). label = human label
                     (0, 0.5 or 1). Also stores label_dense, the label of the dense oracle below, for comparison.
  <task>_<n>.npz     n uniformly sampled pairs labeled by the dense oracle: robosuite's staged shaped reward
                     (reward_shaping=True) recomputed from the stored simulator states. PT's own script labels use the
                     sparse success reward and are ties for 86-98% of the pairs, so they cannot serve as an oracle.
Both store start_1 / start_2 (flat transition index) so a pair can be traced back to the hdf5.

Example:
    python scripts/make_robomimic_pool.py --task lift --dataset-type mh --pt datasets/preference_transformer/Lift/num500_q100_human_train.npz
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

OBS_KEYS = [
    "object",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
]


def load_flat(path):
    """Concatenates every demo in hdf5 order. Returns obs, action, state and the demo id of every transition."""
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
        obs = np.concatenate([np.concatenate([f["data"][d]["obs"][k][:] for k in OBS_KEYS], axis=1) for d in demos])
        action = np.concatenate([f["data"][d]["actions"][:] for d in demos])
        state = np.concatenate([f["data"][d]["states"][:] for d in demos])
        demo_id = np.concatenate([np.full(len(f["data"][d]["actions"]), i) for i, d in enumerate(demos)])
        env_meta = json.loads(f["data"].attrs["env_args"])
    return obs.astype(np.float32), action.astype(np.float32), state, demo_id, env_meta


def dense_rewards(env_meta, state, action):
    """Shaped reward r(s_t, a_t) of robosuite, recomputed from the stored simulator state of each transition."""
    from robomimic.utils import env_utils

    env_meta = json.loads(json.dumps(env_meta))
    env_meta["env_kwargs"]["reward_shaping"] = True
    env = env_utils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
    env = env.env
    env.reset()
    rewards = np.zeros(len(state), dtype=np.float32)
    for t in range(len(state)):
        env.sim.set_state_from_flattened(state[t])
        env.sim.forward()
        rewards[t] = env.reward(action[t])
    return rewards


def locate(segments, action, tol=2e-3):
    """Flat start index of each action segment, -1 if it is not found."""
    starts = np.full(len(segments), -1)
    for i, seg in enumerate(segments):
        cand = np.where(np.abs(action - seg[0]).max(axis=1) < tol)[0]
        cand = cand[cand + len(seg) <= len(action)]
        for c in cand:
            if np.abs(action[c : c + len(seg)] - seg).max() < tol:
                starts[i] = c
                break
    return starts


def oracle_label(returns_1, returns_2):
    return np.where(returns_1 > returns_2, 0.0, np.where(returns_1 < returns_2, 1.0, 0.5)).astype(np.float32)


def gather(x, starts, length):
    return x[starts[:, None] + np.arange(length)[None]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="lift or can")
    parser.add_argument("--dataset-type", type=str, default="mh")
    parser.add_argument("--hdf5", type=str, default=None, help="Defaults to datasets/robomimic/<task>/<type>/low_dim_v15.hdf5")
    parser.add_argument("--pt", type=str, default=None, help="PT human pairs to relocate (skipped if omitted)")
    parser.add_argument("--num-scripted", type=int, default=10000)
    parser.add_argument("--segment-size", type=int, default=100)
    parser.add_argument("--out", type=str, default="datasets/robomimic_pool")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hdf5 = args.hdf5 or f"datasets/robomimic/{args.task}/{args.dataset_type}/low_dim_v15.hdf5"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}_{args.dataset_type}"

    obs, action, state, demo_id, env_meta = load_flat(hdf5)
    reward = dense_rewards(env_meta, state, action)
    np.save(out / f"{name}_dense_reward.npy", reward)
    print(f"{name}: {len(obs)} transitions, {demo_id.max() + 1} demos, dense reward mean {reward.mean():.3f}")
    L = args.segment_size
    cum = np.concatenate([[0.0], np.cumsum(reward)])
    seg_return = lambda s: cum[s + L] - cum[s]  # noqa: E731

    if args.pt is not None:
        pt = np.load(args.pt)
        assert pt["action_1"].shape[1] == L
        s1, s2 = locate(pt["action_1"], action), locate(pt["action_2"], action)
        found = (s1 >= 0) & (s2 >= 0)
        print(f"located {found.sum()}/{len(found)} PT pairs")
        s1, s2, human = s1[found], s2[found], pt["label"][found].astype(np.float32)
        dense = oracle_label(seg_return(s1), seg_return(s2))
        decided = (human != 0.5) & (dense != 0.5)
        print(
            f"human ties {np.mean(human == 0.5):.2f}, oracle ties {np.mean(dense == 0.5):.2f}, "
            f"agreement on decided pairs {np.mean(human[decided] == dense[decided]):.3f} (n={decided.sum()})"
        )
        np.savez(
            out / f"{name}_human.npz",
            obs_1=gather(obs, s1, L), obs_2=gather(obs, s2, L),
            action_1=gather(action, s1, L), action_2=gather(action, s2, L),
            label=human, label_dense=dense, start_1=s1, start_2=s2,
        )

    if args.num_scripted > 0:
        rng = np.random.default_rng(args.seed)
        s1 = rng.integers(0, len(obs) - L + 1, size=args.num_scripted)
        s2 = rng.integers(0, len(obs) - L + 1, size=args.num_scripted)
        label = oracle_label(seg_return(s1), seg_return(s2))
        print(f"scripted pool: {args.num_scripted} pairs, oracle ties {np.mean(label == 0.5):.3f}")
        np.savez(
            out / f"{name}_{args.num_scripted}.npz",
            obs_1=gather(obs, s1, L), obs_2=gather(obs, s2, L),
            action_1=gather(action, s1, L), action_2=gather(action, s2, L),
            label=label, start_1=s1, start_2=s2,
        )
