from ma_gym.envs.combat import Combat
import numpy as np


class FullyObservableCombat(Combat):
    def get_agent_obs(self):
        N = self.n_agents + self._n_opponents
        obs = np.zeros((N, 4))
        grid_shape = np.array(self._grid_shape)
        for i in range(self.n_agents):
            obs[i, :2] = self.agent_pos[i] / grid_shape
            obs[i, 2] = self._agent_cool[i]
            obs[i, 3] = self.agent_health[i] / self._init_health
        for i in range(self._n_opponents):
            obs[self.n_agents + i, :2] = self.opp_pos[i] / grid_shape
            obs[self.n_agents + i, 2] = self._opp_cool[i]
            obs[self.n_agents + i, 3] = self.opp_health[i] / self._init_health

        return np.broadcast_to(obs.ravel(), (self.n_agents, N * 4))
