"""
Simple wrapper for registering metaworld enviornments
properly with gym.
Supports both the original MetaWorld v2 (mujoco_py, gym API) and MetaWorld v3 (mujoco, gymnasium API).
For v3, environment names ending in -v2 are mapped to their -v3 counterparts.
"""
import gym
import numpy as np

try:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS as _MW_ENVS

    MW_VERSION = 2
except ImportError:
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS as _MW_ENVS

    MW_VERSION = 3


def _resolve_env_name(env_name):
    if MW_VERSION == 3 and env_name.endswith("-v2"):
        env_name = env_name[: -len("-v2")] + "-v3"
    return env_name


class SawyerEnv(gym.Env):
    def __init__(self, env_name, seed=True):
        self._env = _MW_ENVS[_resolve_env_name(env_name)]()
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        self._seed = seed
        self._needs_seed = False
        if self._seed:
            self._seed_env()  # Seed it at zero for now.

        self.observation_space = gym.spaces.Box(
            low=self._env.observation_space.low, high=self._env.observation_space.high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self._env.action_space.low, high=self._env.action_space.high, dtype=np.float32
        )
        self._max_episode_steps = self._env.max_path_length

    def _seed_env(self):
        if MW_VERSION == 2:
            self._env.seed(0)
        else:
            self._needs_seed = True  # gymnasium seeds through the next reset call.

    def seed(self, seed=None):
        if self._seed:
            self._seed_env()

    def evaluate_state(self, state, action):
        return self._env.evaluate_state(state, action)

    def step(self, action):
        self._episode_steps += 1
        if MW_VERSION == 2:
            obs, reward, done, info = self._env.step(action)
            state = self._env.sim.get_state()
            qpos, qvel = state.qpos, state.qvel
        else:
            obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            qpos, qvel = self._env.data.qpos, self._env.data.qvel
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Add the underlying state to the info
        info["state"] = np.concatenate((qpos, qvel), axis=0)
        return obs.astype(np.float32), float(reward), done, info

    def set_state(self, state):
        qpos, qvel = state[: self._env.model.nq], state[self._env.model.nq :]
        self._env.set_state(qpos, qvel)

    def reset(self, **kwargs):
        self._episode_steps = 0
        if MW_VERSION == 2:
            return self._env.reset(**kwargs).astype(np.float32)
        if self._needs_seed:
            kwargs["seed"] = 0
            self._needs_seed = False
        obs, _ = self._env.reset(**kwargs)
        return obs.astype(np.float32)

    def render(self, mode="rgb_array", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        assert MW_VERSION == 2, "Rendering is only implemented for MetaWorld v2"
        # stack multiple views
        view_1 = self._env.render(offscreen=True, camera_name="corner", resolution=(width, height))
        view_2 = self._env.render(offscreen=True, camera_name="topview", resolution=(width, height))
        return np.concatenate((view_1, view_2), axis=0)

    def __getattr__(self, name):
        return getattr(self._env, name)
