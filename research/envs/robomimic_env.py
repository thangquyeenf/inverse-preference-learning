from typing import Optional

import gym
import h5py
import numpy as np
from robomimic.config import config_factory
from robomimic.utils import env_utils, file_utils, obs_utils
from robosuite.wrappers import GymWrapper


class RoboMimicEnv(gym.Env):
    def __init__(self, path, horizon: Optional[int] = None):
        env_meta = file_utils.get_env_metadata_from_dataset(dataset_path=path)
        if horizon is not None:
            # robosuite defaults to 1000 steps; robomimic evaluates Lift / Can with 400.
            env_meta["env_kwargs"]["horizon"] = horizon
        env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        ).env
        env.ignore_done = False
        env._max_episode_steps = env.horizon
        keys = [
            "object-state",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]
        self.env = GymWrapper(env, keys=keys)
        self._max_episode_steps = self.env.horizon
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=self.env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=self.env.action_space.dtype,
        )

    def step(self, action: np.ndarray):
        out = self.env.step(action)
        if len(out) == 5:  # robosuite >= 1.5 follows the gymnasium API
            observation, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            observation, reward, done, info = out

        success = bool(self.env._check_success())
        info["success"] = success
        if success:
            done = True

        return observation, reward, done, info

    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        if isinstance(out, tuple):  # gymnasium returns (obs, info)
            out = out[0]
        return out
