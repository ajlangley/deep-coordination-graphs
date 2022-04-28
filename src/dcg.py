from itertools import product
import numpy as np
import torch
from torch import nn


class DCG(nn.Module):
    def __init__(self, nodes, edges, n_actions):
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.edges = nn.ModuleList(edges)
        self.N = len(nodes)
        self.E = len(edges)
        self.actions = torch.tensor(list(product(range(n_actions),
                                                 repeat=len(nodes))),
                                    dtype=torch.int64)

    def eval_action(self, obs, a):
        node_vals = sum([f.eval_action(obs, a) for f in self.nodes])
        edge_vals = sum([f.eval_action(obs, a) for f in self.edges])

        return node_vals / self.N + edge_vals / self.E

    def argmax(self, obs):
        q_vals = self._compute_q_vals(obs)

        return self.actions[torch.argmax(q_vals, dim=-1)]

    def max(self, obs):
        q_vals = self._compute_q_vals(obs)

        return torch.max(q_vals, dim=-1)[0]

    def _compute_q_vals(self, obs):
        node_vals = sum([f.eval_actions(obs, self.actions) for f in self.nodes])
        edge_vals = sum([f.eval_actions(obs, self.actions) for f in self.edges])
        q_vals = node_vals / self.N + edge_vals / self.E

        return q_vals

    def to(self, device):
        super().to(device)
        self.actions = self.actions.to(device)

    def cpu(self):
        super().cpu()
        self.actions = self.actions.cpu()

    def cuda(self):
        super().cpu()
        self.actions = self.actions.cuda()

class DCGNode(nn.Module):
    def __init__(self, network, agent_id):
        super().__init__()
        self.network = network
        self.agent_id = agent_id

    def eval_action(self, obs, a):
        if obs.dim() == 3:
            local_obs = torch.flatten(obs[:, self.agent_id], start_dim=1)
            net_output = self.network(local_obs)
            q_val = torch.gather(net_output,
                                 1,
                                 a[:, self.agent_id].unsqueeze(-1)).ravel()
        else:
            local_obs = obs[self.agent_id].ravel()
            net_output = self.network(local_obs)
            q_val = net_output[a[self.agent_id]]

        return q_val

    def eval_actions(self, obs, actions):
        if obs.dim() == 3:
            local_obs = torch.flatten(obs[:, self.agent_id], start_dim=1)
            net_output = self.network(local_obs)
            q_vals = net_output[:, actions[:, self.agent_id]]
        else:
            local_obs = obs[self.agent_id].ravel()
            net_output = self.network(local_obs)
            q_vals = net_output[actions[:, self.agent_id]]

        return q_vals


class DCGEdge(nn.Module):
    def __init__(self, network, agent_id1, agent_id2, n_actions):
        super().__init__()
        self.network = network
        self.agent_id1 = agent_id1
        self.agent_id2 = agent_id2
        self.agent_ids = [agent_id1, agent_id2]
        self.n_actions = n_actions

    def forward(self, local_obs):
        return self.network(local_obs)

    def eval_action(self, obs, a):
        if obs.dim() == 3:
            local_obs = torch.flatten(obs[:, self.agent_ids], start_dim=1)
            net_output = self.forward(local_obs).view(obs.size(0),
                                                      self.n_actions,
                                                      self.n_actions)
            q_val = net_output[torch.arange(obs.size(0)),
                               a[:, self.agent_id1],
                               a[:, self.agent_id2]]
        else:
            local_obs = obs[self.agent_ids].ravel()
            net_output = self.forward(local_obs).view(self.n_actions,
                                                      self.n_actions)
            q_val = net_output[a[self.agent_id1], a[self.agent_id2]]

        return q_val


    def eval_actions(self, obs, actions):
        action_indices = tuple()
        if obs.dim() == 3:
            local_obs = torch.flatten(obs[:, self.agent_ids], start_dim=1)
            net_output = self.network(local_obs).view(obs.size(0),
                                                      self.n_actions,
                                                      self.n_actions)
            q_vals = net_output[:,
                                actions[:, self.agent_id1],
                                actions[:, self.agent_id2]]
        else:
            local_obs = obs[self.agent_ids].ravel()
            net_output = self.network(local_obs).view(self.n_actions,
                                                      self.n_actions)
            q_vals = net_output[actions[:, self.agent_id1], actions[:, self.agent_id2]]

        return q_vals


class FactorizedDCGEdge(DCGEdge):
    def __init__(self, network, agent_ids, n_actions):
        super().__init__()
