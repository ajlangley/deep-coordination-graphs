from itertools import product
import numpy as np
import torch
from torch import nn


class DCG(nn.Module):
    def __init__(self, nodes, edges, n_actions, device=torch.device('cpu'),
                 msg_passing_iters=8):
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.edges = nn.ModuleList(edges)
        self.N = len(nodes)
        self.E = len(edges)
        self.n_actions = n_actions
        self.actions = torch.tensor(list(product(range(n_actions),
                                                 repeat=len(nodes))),
                                    dtype=torch.int64)
        self.msg_passing_iters = msg_passing_iters
        self.device = device

        self.edges_from = torch.tensor([e.agent_id1 for e in edges],
                                        device=device,
                                        dtype=torch.long)
        self.edges_to = torch.tensor([e.agent_id2 for e in edges],
                                     device=device,
                                     dtype=torch.long)

    def eval_action(self, obs, a):
        node_vals = sum([f.eval_action(obs, a) for f in self.nodes])
        edge_vals = sum([f.eval_action(obs, a) for f in self.edges])

        return node_vals / self.N + edge_vals / self.E

    def argmax(self, obs):
        max_val, a_max = self.message_passing(obs)

        return a_max

    def max(self, obs):
        max_val, a_max = self.message_passing(obs)

        return max_val

    def message_passing(self, obs):
        batch_size = 1 if obs.dim() == 2 else obs.size(0)
        if batch_size == 1:
            obs = torch.unsqueeze(obs, 0)

        batch_size = obs.size(0)
        node_batch_indices = np.repeat(np.arange(batch_size), self.N)
        node_indices = np.tile(np.arange(self.N), batch_size)
        edge_batch_indices = np.repeat(np.arange(batch_size), self.E)
        edge_indices = np.tile(np.arange(self.E), batch_size)

        node_vals = torch.stack([f.forward(obs) for f in self.nodes])
        edge_vals = torch.stack([f.forward(obs) for f in self.edges])
        node_vals = torch.transpose(node_vals, 0, 1)
        edge_vals = torch.transpose(edge_vals, 0, 1)
        a_max = torch.argmax(node_vals, dim=-1)
        q_max = self._eval_action_from_outputs(a_max, node_vals, edge_vals,
                                               node_batch_indices, node_indices,
                                               edge_batch_indices, edge_indices)

        msg_forw = torch.zeros((obs.size(0), self.E, self.n_actions),
                               dtype=torch.float32,
                               device=self.device)
        msg_back = torch.zeros((obs.size(0), self.E, self.n_actions),
                               dtype=torch.float32,
                               device=self.device)

        q = node_vals / self.N
        for t in range(1, self.msg_passing_iters + 1):
            forw_vals = torch.unsqueeze(q[:, self.edges_from] - msg_back, -1) \
                            + edge_vals / self.E
            back_vals = torch.unsqueeze(q[:, self.edges_to] - msg_forw, -1) \
                            + torch.transpose(edge_vals, -2, -1) / self.E
            msg_forw = torch.max(forw_vals, dim=-2)[0]
            msg_back = torch.max(back_vals, dim=-2)[0]
            # Message normalization
            msg_forw -= torch.sum(msg_forw, dim=-1).unsqueeze(-1) / self.n_actions
            msg_back -= torch.sum(msg_back, dim=-1).unsqueeze(-1) / self.n_actions
            # Update q. Implement as a loop for now
            q = node_vals / self.N
            for i, (j, k) in enumerate(zip(self.edges_from, self.edges_to)):
                q[:, k] += msg_forw[:, i]
                q[:, j] += msg_back[:, i]

            a = torch.argmax(q, dim=-1)
            q_val = self._eval_action_from_outputs(a, node_vals, edge_vals, node_batch_indices,
                                                   node_indices, edge_batch_indices, edge_indices)
            update_indices = torch.argwhere(q_val > q_max).ravel()
            a_max[update_indices] = a[update_indices]
            q_max[update_indices] = q_val[update_indices]

        if batch_size == 1:
            q_max, a_max = q_max.ravel(), a_max.ravel()

        return q_max, a_max

    def _eval_action_from_outputs(self, a, node_outputs, edge_outputs, node_batch_indices,
                                  node_indices, edge_batch_indices, edge_indices):
        batch_size = len(a)
        node_a_indices = a[:, np.arange(self.N)].ravel()
        edge_a_indices_1 = a[:, self.edges_from].ravel()
        edge_a_indices_2 = a[:, self.edges_to].ravel()
        node_vals = node_outputs[node_batch_indices, node_indices,
                                 node_a_indices].view(batch_size, -1)
        node_val = torch.sum(node_vals, dim=1)
        edge_vals = edge_outputs[edge_batch_indices, edge_indices,
                                 edge_a_indices_1, edge_a_indices_2].view(batch_size, -1)
        edge_val = torch.sum(edge_vals, dim=1)

        return node_val / self.N + edge_val / self.E

    def _compute_q_vals(self, obs):
        node_vals = sum([f.eval_actions(obs, self.actions) for f in self.nodes])
        edge_vals = sum([f.eval_actions(obs, self.actions) for f in self.edges])
        q_vals = node_vals / self.N + edge_vals / self.E

        return q_vals

    def to(self, device):
        super().to(device)
        self.actions = self.actions.to(device)
        self.edges_from = self.edges_from.to(device)
        self.edges_to = self.edges_to.to(device)
        self.device = device

    def cpu(self):
        super().cpu()
        self.actions = self.actions.cpu()
        self.edges_from = self.edges_from.cpu()
        self.edges_to = self.edges_to.cpu()
        self.device = torch.device('cpu')

    def cuda(self):
        super().cuda()
        self.actions = self.actions.cuda()
        self.edges_from = self.edges_from.cuda()
        self.edges_to = self.edges_to.cuda()
        self.device = torch.device('cuda')


class RDCG(nn.Module):
    def __init__(self, nodes, edges, encoder, rnn_cell, n_actions,
                 device=torch.device('cpu'), msg_passing_iters=8):
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.edges = nn.ModuleList(edges)
        self.encoder = encoder
        self.rnn_cell = rnn_cell
        self.N = len(nodes)
        self.E = len(edges)
        self.n_actions = n_actions
        self.actions = torch.tensor(list(product(range(n_actions),
                                                 repeat=len(nodes))),
                                    dtype=torch.int64)
        self.msg_passing_iters = msg_passing_iters
        self.device = device

        self.node_indices = torch.arange(self.N)
        self.edge_indices = torch.arange(self.E)
        self.edges_from = torch.tensor([e.agent_id1 for e in edges],
                                        device=device,
                                        dtype=torch.long)
        self.edges_to = torch.tensor([e.agent_id2 for e in edges],
                                     device=device,
                                     dtype=torch.long)

        self.h = None

    def eval_action(self, obs, a):
        B, T = obs.size()[:2]
        a = a.view(B * T, -1)
        encoding = self._encode_obs(obs)
        encoding = encoding.reshape(B * T, self.N, -1)
        node_vals = sum([f.eval_action(encoding, a) for f in self.nodes])
        edge_vals = sum([f.eval_action(encoding, a) for f in self.edges])
        q_val = node_vals / self.N + edge_vals / self.E
        q_val = q_val.view(B, T)

        return q_val

    def max(self, obs):
        B, T = obs.size()[:2]
        encoding = self._encode_obs(obs)
        encoding = encoding.reshape(B * T, self.N, -1)
#         q_max = self._message_passing(encoding)[0]
#         q_max = q_max.view(B, T)

        q_vals = self._compute_q_vals(encoding)
        q_max = torch.max(q_vals, dim=-1).value.view(B, T)

        return q_max

    def argmax(self, obs):
        B, T = obs.size()[:2]
        encoding = self._encode_obs(obs)
        encoding = encoding.reshape(B * T, self.N, -1)
#         a_max = self._message_passing(encoding)[1]
#         a_max = a_max.view(B, T, -1)

        q_vals = self._compute_q_vals(encoding)
        max_indices = torch.argmax(q_vals, dim=-1)
        a_max = self.actions[max_indices].view(B, T, -1)

        return a_max

    def _message_passing(self, obs):
        # Should create all these tensors using torch functions eventually...
        batch_size = obs.size(0)
        node_batch_indices = np.repeat(np.arange(batch_size), self.N)
        node_indices = np.tile(np.arange(self.N), batch_size)
        edge_batch_indices = np.repeat(np.arange(batch_size), self.E)
        edge_indices = np.tile(np.arange(self.E), batch_size)

        node_vals = torch.stack([f.forward(obs) for f in self.nodes])
        edge_vals = torch.stack([f.forward(obs) for f in self.edges])
        node_vals = torch.transpose(node_vals, 0, 1)
        edge_vals = torch.transpose(edge_vals, 0, 1)
        a_max = torch.argmax(node_vals, dim=-1)
        q_max = self._eval_action_from_outputs(a_max, node_vals, edge_vals,
                                               node_batch_indices, node_indices,
                                               edge_batch_indices, edge_indices)

        msg_forw = torch.zeros((obs.size(0), self.E, self.n_actions),
                               dtype=torch.float32,
                               device=self.device)
        msg_back = torch.zeros((obs.size(0), self.E, self.n_actions),
                               dtype=torch.float32,
                               device=self.device)

        q = node_vals / self.N
        for t in range(1, self.msg_passing_iters + 1):
            forw_vals = torch.unsqueeze(q[:, self.edges_from] - msg_back, -1) \
                            + edge_vals / self.E
            back_vals = torch.unsqueeze(q[:, self.edges_to] - msg_forw, -1) \
                            + torch.transpose(edge_vals, -2, -1) / self.E
            msg_forw = torch.max(forw_vals, dim=-2)[0]
            msg_back = torch.max(back_vals, dim=-2)[0]
            # Message normalization
            msg_forw -= torch.sum(msg_forw, dim=-1).unsqueeze(-1) / self.n_actions
            msg_back -= torch.sum(msg_back, dim=-1).unsqueeze(-1) / self.n_actions
            # Update q. Implement as a loop for now
            q = node_vals / self.N
            for i, (j, k) in enumerate(zip(self.edges_from, self.edges_to)):
                q[:, k] += msg_forw[:, i]
                q[:, j] += msg_back[:, i]

            a = torch.argmax(q, dim=-1)
            q_val = self._eval_action_from_outputs(a, node_vals, edge_vals, node_batch_indices,
                                                   node_indices, edge_batch_indices, edge_indices)
            update_indices = torch.argwhere(q_val > q_max).ravel()
            a_max[update_indices] = a[update_indices]
            q_max[update_indices] = q_val[update_indices]

        return q_max, a_max

    def _compute_q_vals(self, obs):
        node_vals = sum([f.eval_actions(obs, self.actions) for f in self.nodes])
        edge_vals = sum([f.eval_actions(obs, self.actions) for f in self.edges])
        q_vals = node_vals / self.N + edge_vals / self.E

        return q_vals

    def _eval_action_from_outputs(self, a, node_outputs, edge_outputs, node_batch_indices,
                                  node_indices, edge_batch_indices, edge_indices):
        batch_size = len(a)
        node_a_indices = a[:, np.arange(self.N)].ravel()
        edge_a_indices_1 = a[:, self.edges_from].ravel()
        edge_a_indices_2 = a[:, self.edges_to].ravel()
        node_vals = node_outputs[node_batch_indices, node_indices,
                                 node_a_indices].view(batch_size, -1)
        node_val = torch.sum(node_vals, dim=1)
        edge_vals = edge_outputs[edge_batch_indices, edge_indices,
                                 edge_a_indices_1, edge_a_indices_2].view(batch_size, -1)
        edge_val = torch.sum(edge_vals, dim=1)

        return node_val / self.N + edge_val / self.E

    def _encode_obs(self, obs):
        B, T = obs.size()[:2]
        encoding = self.encoder(obs)
        encoding = torch.permute(encoding, (0, 2, 1, 3)).reshape(B * self.N, T, -1)
        rnn_output, self.h = self.rnn_cell(encoding, hx=self.h)
        rnn_output = torch.permute(rnn_output.view(B, self.N, T, -1),
                                   (0, 2, 1, 3))

        return rnn_output

    def to(self, device):
        super().to(device)
        self.actions = self.actions.to(device)
        self.node_indices = self.node_indices.to(device)
        self.edge_indices = self.edge_indices.to(device)
        self.edges_from = self.edges_from.to(device)
        self.edges_to = self.edges_to.to(device)
        self.device = device

    def cpu(self):
        super().cpu()
        self.actions = self.actions.cpu()
        self.node_indices = self.node_indices.cpu()
        self.edge_indices = self.edge_indices.cpu()
        self.edges_from = self.edges_from.cpu()
        self.edges_to = self.edges_to.cpu()
        self.device = torch.device('cpu')

    def cuda(self):
        super().cuda()
        self.actions = self.actions.cuda()
        self.node_indices = self.node_indices.cuda()
        elf.edge_indices = self.edge_indices.cuda()
        self.edges_from = self.edges_from.cuda()
        self.edges_to = self.edges_to.cuda()
        self.device = torch.device('cuda')


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


class DCGSharedWeights(nn.Module):
    def __init__(self, f_node, f_edge, n_nodes, n_actions, edges=None, msg_passing_iters=8,
                 device=torch.device('cpu')):
        super().__init__()
        self.f_node = f_node
        self.f_edge = f_edge
        self.n_actions = n_actions
        if edges is None:
            self.edges = [(i, j) for i in range(n_nodes) \
                             for j in range(n_nodes) if i != j]
        else:
            self.edges = edges
        self.N = n_nodes
        self.E = len(self.edges)
        self.msg_passing_iters = msg_passing_iters
        self.actions = torch.tensor(list(product(range(n_actions),
                                                 repeat=n_nodes)),
                                    dtype=torch.int64)
        if type(self.edges) is not torch.Tensor:
            self.edges = torch.tensor(self.edges, dtype=torch.long)

        self.edges_from = torch.tensor([i for (i, j) in self.edges],
                                        device=device,
                                        dtype=torch.long)
        self.edges_to = torch.tensor([j for (i, j) in self.edges],
                                     device=device,
                                     dtype=torch.long)

    def eval_action(self, obs, a):
        batch_size = obs.size(0) if obs.dim() == 3 else 1
        if batch_size == 1:
            obs = obs.unsqueeze(0)

        node_outputs = self.compute_node_outputs(obs)
        edge_outputs = self.compute_edge_outputs(obs)
        edge_indices = torch.ravel(self.edges)
        if batch_size > 1:
            a_pairwise = a[:, edge_indices].view(-1, self.E, 2)
            a_pairwise_joint = a_pairwise[:, :, 0] * self.n_actions + a_pairwise[:, :, 1]
        else:
            a_pairwise = a[edge_indices]
        node_vals = torch.gather(node_outputs, -1, a.unsqueeze(-1))
        node_vals = node_vals.squeeze(-1)
        edge_vals = torch.gather(edge_outputs, -1, a_pairwise_joint.unsqueeze(-1))
        edge_vals = edge_vals.squeeze(-1)
        node_val = torch.mean(node_vals, dim=-1)
        edge_val = torch.mean(edge_vals, dim=-1)

        if batch_size == 1:
            node_val, edge_val = node_val.ravel(), edge_val.ravel()

        return node_val + edge_val

    def compute_node_outputs(self, obs):
        node_inputs = obs.view(obs.size(0) * self.N, -1)
        node_outputs = self.f_node(node_inputs).view(-1, self.N, self.n_actions)

        return node_outputs

    def compute_edge_outputs(self, obs):
        edge_indices = torch.ravel(self.edges)
        edge_inputs = obs[:, edge_indices].view(-1, obs.size(-1) * 2)
        edge_outputs = self.f_edge(edge_inputs).view(-1, self.E, self.n_actions ** 2)

        return edge_outputs

    def argmax(self, obs):
        max_val, a_max = self.message_passing(obs)

        return a_max

    def max(self, obs):
        max_val, a_max = self.message_passing(obs)

        return max_val

    def message_passing(self, obs):
        is_batch = obs.dim() == 3
        if is_batch == False:
            obs = torch.unsqueeze(obs, 0)

        batch_size = obs.size(0)
        node_batch_indices = np.repeat(np.arange(batch_size), self.N)
        node_indices = np.tile(np.arange(self.N), batch_size)
        edge_batch_indices = np.repeat(np.arange(batch_size), self.E)
        edge_indices = np.tile(np.arange(self.E), batch_size)

        node_vals = self.compute_node_outputs(obs)
        edge_vals = self.compute_edge_outputs(obs).view(-1, self.E,
                                                        self.n_actions,
                                                        self.n_actions)
        a_max = torch.argmax(node_vals, dim=-1)
        q_max = self._eval_action_from_outputs(a_max, node_vals, edge_vals,
                                               node_batch_indices, node_indices,
                                               edge_batch_indices, edge_indices)

        msg_forw = torch.zeros((obs.size(0), self.E, self.n_actions),
                               dtype=torch.float32,
                               device=self.device)
        msg_back = torch.zeros((obs.size(0), self.E, self.n_actions),
                               dtype=torch.float32,
                               device=self.device)

        q = node_vals / self.N
        for t in range(1, self.msg_passing_iters + 1):
            forw_vals = torch.unsqueeze(q[:, self.edges_from] - msg_back, -1) \
                            + edge_vals / self.E
            back_vals = torch.unsqueeze(q[:, self.edges_to] - msg_forw, -1) \
                            + torch.transpose(edge_vals, -2, -1) / self.E
            msg_forw = torch.max(forw_vals, dim=-2)[0]
            msg_back = torch.max(back_vals, dim=-2)[0]
            # Message normalization
            msg_forw -= torch.sum(msg_forw, dim=-1).unsqueeze(-1) / self.n_actions
            msg_back -= torch.sum(msg_back, dim=-1).unsqueeze(-1) / self.n_actions
            # Update q. Implement as a loop for now
            q = node_vals / self.N
            for i, (j, k) in enumerate(zip(self.edges_from, self.edges_to)):
                q[:, k] += msg_forw[:, i]
                q[:, j] += msg_back[:, i]

            a = torch.argmax(q, dim=-1)
            q_val = self._eval_action_from_outputs(a, node_vals, edge_vals, node_batch_indices,
                                                   node_indices, edge_batch_indices, edge_indices)
            update_indices = torch.argwhere(q_val > q_max).ravel()
            a_max[update_indices] = a[update_indices]
            q_max[update_indices] = q_val[update_indices]

        if is_batch == False:
            q_max, a_max = q_max.ravel(), a_max.ravel()

        return q_max, a_max

    def _eval_action_from_outputs(self, a, node_outputs, edge_outputs, node_batch_indices,
                                  node_indices, edge_batch_indices, edge_indices):
        batch_size = len(a)
        node_a_indices = a[:, np.arange(self.N)].ravel()
        edge_a_indices_1 = a[:, self.edges_from].ravel()
        edge_a_indices_2 = a[:, self.edges_to].ravel()
        node_vals = node_outputs[node_batch_indices, node_indices,
                                 node_a_indices].view(batch_size, -1)
        node_val = torch.sum(node_vals, dim=1)
        edge_vals = edge_outputs[edge_batch_indices, edge_indices,
                                 edge_a_indices_1, edge_a_indices_2].view(batch_size, -1)
        edge_val = torch.sum(edge_vals, dim=1)

        return node_val / self.N + edge_val / self.E

    def to(self, device):
        super().to(device)
        self.actions = self.actions.to(device)
        self.edges = self.edges.to(device)
        self.edges_from = self.edges_from.to(device)
        self.edges_to = self.edges_to.to(device)
        self.device = device

    def cpu(self):
        super().cpu()
        self.actions = self.actions.cpu()
        self.edges = self.edges.cpu()
        self.edges_from = self.edges_from.cpu()
        self.edges_to = self.edges_to.cpu()
        self.device = device

    def cuda(self):
        super().cpu()
        self.actions = self.actions.cuda()
        self.edges = self.edges.cuda()
        self.edges_from = self.edges_from.cuda()
        self.edges_to = self.edges_to.cuda()
        self.device = device
