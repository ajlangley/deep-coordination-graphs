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

    def forward(self, obs):
        local_obs = self.get_local_obs(obs)

        return self.network.forward(local_obs)

    def eval_action(self, obs, a):
        net_output = self.forward(obs)
        if obs.dim() == 3:
            q_val = torch.gather(net_output,
                                 1,
                                 a[:, self.agent_id].unsqueeze(-1)).ravel()
        else:
            q_val = net_output[a[self.agent_id]]

        return q_val

    def eval_actions(self, obs, actions):
        net_output = self.forward(obs)
        if obs.dim() == 3:
            q_vals = net_output[:, actions[:, self.agent_id]]
        else:
            q_vals = net_output[actions[:, self.agent_id]]

        return q_vals

    def get_local_obs(self, obs):
        if obs.dim() == 3:
            local_obs = torch.flatten(obs[:, self.agent_id], start_dim=1)
        else:
            local_obs = obs[self.agent_id].ravel()

        return local_obs


class DCGEdge(nn.Module):
    def __init__(self, network, agent_id1, agent_id2, n_actions):
        super().__init__()
        self.network = network
        self.agent_id1 = agent_id1
        self.agent_id2 = agent_id2
        self.agent_ids = [agent_id1, agent_id2]
        self.n_actions = n_actions

    def forward(self, obs):
        local_obs = self.get_local_obs(obs)
        if obs.dim() == 3:
            net_output = self.network(local_obs).view(obs.size(0),
                                                      self.n_actions,
                                                      self.n_actions)
        else:
            net_output = self.network(local_obs).view(self.n_actions,
                                                      self.n_actions)

        return net_output

    def eval_action(self, obs, a):
        net_output = self.forward(obs)
        if obs.dim() == 3:
            q_val = net_output[torch.arange(obs.size(0)),
                               a[:, self.agent_id1],
                               a[:, self.agent_id2]]
        else:
            q_val = net_output[a[self.agent_id1], a[self.agent_id2]]

        return q_val


    def eval_actions(self, obs, actions):
        net_output = self.forward(obs)
        if obs.dim() == 3:
            q_vals = net_output[:,
                                actions[:, self.agent_id1],
                                actions[:, self.agent_id2]]
        else:
            q_vals = net_output[actions[:, self.agent_id1], actions[:, self.agent_id2]]

        return q_vals

    def get_local_obs(self, obs):
        if obs.dim() == 3:
            local_obs = torch.flatten(obs[:, self.agent_ids], start_dim=1)
        else:
            local_obs = obs[self.agent_ids].ravel()

        return local_obs


class FactoredDCGEdge(DCGEdge):
    def __init__(self, factors1, factors2, agent_id1, agent_id2, n_actions):
        super().__init__(None, agent_id1, agent_id2, n_actions)
        self.factors1 = nn.ModuleList(factors1)
        self.factors2 = nn.ModuleList(factors2)

    def forward(self, obs):
        local_obs = self.get_local_obs(obs)
        if local_obs.dim() == 2:
            f1_outputs = torch.stack([f(local_obs) for f in self.factors1])
            f2_outputs = torch.stack([f(local_obs) for f in self.factors2])
            f1_outputs = torch.permute(f1_outputs, (1, 0, 2))
            f2_outputs = torch.permute(f2_outputs, (1, 0, 2))
            net_output = torch.einsum('ijk,ijl->ikl', f1_outputs, f2_outputs)
        else:
            f1_outputs = torch.stack([f(local_obs) for f in self.factors1])
            f2_outputs = torch.stack([f(local_obs) for f in self.factors2])
            net_output = torch.einsum('ij,ik->jk', f1_outputs, f2_outputs)

        return net_output
