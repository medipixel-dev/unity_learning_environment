# -*- coding: utf-8 -*-

import sys

from gym import spaces
from mlagents.envs import UnityEnvironment
import numpy as np

env_cfg = dict(Sokoban=dict(gridSize=5, numGoals=1, numBoxes=1), Drone=dict(numGoals=5), Drone_discrete=dict(numGoals=5))


class Sokoban:
    spec = None
    name = None
    action_space = None
    observation_space = None

    def __init__(
        self,
        env_path: str,
        env_name: str,
        cfg: dict,
        train_mode=True,
        worker_id: int = 1,
    ):
        self.env = UnityEnvironment(file_name=env_path, worker_id=worker_id)
        self.default_brain = self.env.brain_names[0]
        self.cfg = cfg
        self.name = env_name
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )
        self.train_mode = train_mode

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode, config=self.cfg)[
            self.default_brain
        ]
        return env_info.visual_observations[0][0].reshape(3, 84, 84)

    def step(self, action):
        env_info = self.env.step(action.tolist())[self.default_brain]
        observation = env_info.visual_observations[0][0].reshape(3, 84, 84)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        info = None
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def seed(self, seed):
        pass


class Drone:
    spec = None
    name = None
    action_space = None
    observation_space = None

    def __init__(
        self,
        env_path: str,
        env_name: str,
        cfg: dict,
        train_mode: bool = True,
        worker_id: int = 1,
    ):
        self.env = UnityEnvironment(file_name=env_path, worker_id=worker_id)
        self.default_brain = self.env.brain_names[0]
        self.cfg = cfg
        self.name = env_name
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.train_mode = train_mode

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode, config=self.cfg)[
            self.default_brain
        ]
        return env_info.vector_observations[0]

    def step(self, action):
        env_info = self.env.step(action.tolist())[self.default_brain]
        observation = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        info = None
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def seed(self, seed):
        pass

class Drone_discrete:
    spec = None
    name = None
    action_space = None
    observation_space = None

    def __init__(
        self,
        env_path: str,
        env_name: str,
        cfg: dict,
        train_mode: bool = True,
        worker_id: int = 1,
    ):
        self.env = UnityEnvironment(file_name=env_path, worker_id=worker_id)
        self.default_brain = self.env.brain_names[0]
        self.cfg = cfg
        self.name = env_name
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        self.train_mode = train_mode

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode, config=self.cfg)[
            self.default_brain
        ]
        return env_info.visual_observations[0][0].reshape(1, 84, 84)

    def step(self, action):
        env_info = self.env.step(action.tolist())[self.default_brain]
        observation = env_info.visual_observations[0][0].reshape(1, 84, 84)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        info = None
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def seed(self, seed):
        pass

def unity_env_generator(env_name: str, train_mode: bool = True, worker_id: int = 1):
    env_path = f"UnityEnv/{env_name}/{env_name}"
    env = getattr(sys.modules[__name__], env_name)(
        env_path, env_name, env_cfg[env_name], train_mode, worker_id
    )
    return env