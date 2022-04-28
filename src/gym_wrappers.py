from gym import ActionWrapper, Env, ObservationWrapper, RewardWrapper, Wrapper
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
import numpy as np


class AgentIDWrapper(ObservationWrapper):
    def observation(self, obs):
        n_agents = len(obs)
        obs_size = len(obs[0])
        obs_agent_ids = np.zeros((n_agents, obs_size + n_agents))
        obs_agent_ids[:, :obs_size] = obs
        obs_agent_ids[:, obs_size:] = np.eye(n_agents)

        return obs_agent_ids


class LastActionWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_sample = env.reset()
        self.n_agents = len(obs_sample)
        self.base_obs_size = len(obs_sample[0])
        self.n_actions = env.action_space[0].n
        self.a_last = tuple([0] * self.n_agents)

    def step(self, a):
        self.a_last = tuple(a)
        obs, r, done, info = self.env.step(a)
        obs = self.observation(obs)

        return obs, r, done, info

    def observation(self, obs):
        a_last_onehots = np.zeros((self.n_agents, self.n_actions))
        last_a_obs = np.zeros((self.n_agents,
                               self.base_obs_size + self.n_actions))
        last_a_obs[:, :self.base_obs_size] = obs
        last_a_obs[:, self.base_obs_size:][np.arange(self.n_agents), self.a_last] = 1

        return last_a_obs


class MeanRewardWrapper(RewardWrapper):
    def reward(self, reward):
        return np.mean(reward)


class SumRewardWrapper(RewardWrapper):
    def reward(self, reward):
        return np.sum(reward)


class AllDoneWrapper(Wrapper):
    def step(self, a):
        obs, r, done, info = self.env.step(a)

        return obs, r, np.all(done), info
