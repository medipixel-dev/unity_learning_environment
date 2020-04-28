# -*- coding: utf-8 -*-

import sys

from gym import spaces
from mlagents.envs import UnityEnvironment
import numpy as np



Sokoban_env_cfg = {"gridSize": 5, "numGoals": 1, "numBoxes": 1}
Drone_env_cfg = {"numGoals": 5}

class gym_():
    def make(self, env_name):
        env_path = env_name + "/" + env_name
        return getattr(sys.modules[__name__], env_name)(env_path, env_name)


class Sokoban():
    spec = None
    name = None
    action_space = None
    observation_space = None

    def __init__(self, env_path, env_name, env_cfg=Sokoban_env_cfg):
        self.env = UnityEnvironment(file_name=env_path, worker_id=2)
        self.default_brain = self.env.brain_names[0]
        self.env_cfg = env_cfg
        self.name = env_name
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )

    def reset(self):
        env_info = self.env.reset(train_mode=True, config=self.env_cfg)[
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

class Drone():
    spec = None
    name = None
    action_space = None
    observation_space = None

    def __init__(self, env_path, env_name, env_cfg=Drone_env_cfg):
        self.env = UnityEnvironment(file_name=env_path, worker_id=1)
        self.default_brain = self.env.brain_names[0]
        self.env_cfg = env_cfg
        self.name = env_name
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self):
        env_info = self.env.reset(train_mode=True, config=self.env_cfg)[
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

def unity_env_generator(env_id):
    gym = gym_()
    env = gym.make(env_id)
    return env
