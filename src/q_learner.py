from copy import deepcopy
import numpy as np
import torch
from torch.optim import RMSprop
from torch import nn


class QLearner(nn.Module):
    def __init__(self, qnet, env, lr=5e-4, discount=0.99, LossFun=nn.MSELoss, Optimizer=RMSprop,
                 use_double_q_learning=True, optimizer_args={}):
        super().__init__()
        self.qnet = qnet
        self.q_target = deepcopy(qnet)
        self.env = env
        self.lr = lr
        self.discount = discount
        self.loss_fun = LossFun()
        self.optimizer = Optimizer(qnet.parameters(), lr=lr, **optimizer_args)
        self.use_double_q_learning = use_double_q_learning
        self.device = torch.device('cpu')

    def eps_greedy_action(self, obs, eps):
        if np.random.uniform() < eps:
            return self.env.action_space.sample()
        else:
            return self.qnet.argmax(obs).detach().cpu().numpy()

    def train(self, obs, a, r, done, obs_new):
        self.qnet.train()

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        obs_new = torch.tensor(obs_new, dtype=torch.float32, device=self.device)

        q_pred = self.qnet.eval_action(obs, a)
        with torch.no_grad():
            if self.use_double_q_learning:
                a_max = self.qnet.argmax(obs_new)
                q_max = self.q_target.eval_action(obs_new, a_max)
            else:
                q_max = self.q_target.max(obs_new)
        target = r + (1 - done) * self.discount * q_max
        loss = self.loss_fun(q_pred, target.detach())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evaluate(self, n_eval):
        self.qnet.eval()
        cum_r = 0
        with torch.no_grad():
            for _ in range(n_eval):
                obs = self.env.reset()
                done = False
                while done == False:
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    a = self.qnet.argmax(obs).detach().cpu().numpy()
                    obs, r, done, _ = self.env.step(a)
                    cum_r += r

        return cum_r / n_eval

    def update_target_net(self):
        self.q_target = deepcopy(self.qnet)

    def save(self, fp):
        pass

    def to(self, device):
        super().to(device)
        self.qnet.to(device)
        self.qnet.actions = self.qnet.actions.to(device)
        self.device = device

    def cpu(self):
        super().cpu()
        self.qnet.cpu()
        self.device = torch.device('cpu')

    def cuda(self):
        super().cpu()
        self.qnet.cuda()
        self.device = torch.device('cuda')


class RecurrentQLearner(nn.Module):
    def __init__(self, qnet, env, lr=5e-4, discount=0.99, Loss=nn.SmoothL1Loss,
                 Optimizer=RMSprop, use_double_q_learning=True, grad_norm_clip=10,
                 optimizer_args={}):
        super().__init__()
        self.qnet = qnet
        self.q_target = deepcopy(qnet)
        self.env = env
        self.lr = lr
        self.discount = discount
        self.loss_fun = Loss(reduction='none')
        self.optimizer = Optimizer(qnet.parameters(), lr=lr, **optimizer_args)
        self.use_double_q_learning = use_double_q_learning
        self.grad_norm_clip = grad_norm_clip
        self.device = torch.device('cpu')

    def train(self, obs, a, r, done, fill_mask):
        self.qnet.train()

        obs = obs.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        done = done.to(self.device)
        fill_mask = fill_mask.to(self.device)

#         print(obs.size(), a.size(), r.size(), done.size(), fill_mask.size())

        self.reset_rnn_cell()
        q_pred = self.qnet.eval_action(obs[:, :-1], a)
        with torch.no_grad():
            if self.use_double_q_learning:
                self.reset_rnn_cell()
                # Inefficient..
                a_max = self.qnet.argmax(obs)
                q_max = self.q_target.eval_action(obs, a_max)[:, 1:]
            else:
                q_max = self.q_target.max(obs)[:, 1:]
        targets = r + (1 - done) * self.discount * q_max
        losses = self.loss_fun(q_pred.ravel(), targets.detach().ravel())
        loss = torch.sum(losses * fill_mask.ravel()) / torch.sum(fill_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evaluate(self, n_eval):
        self.qnet.eval()
        cum_r = 0
        with torch.no_grad():
            for _ in range(n_eval):
                obs = self.env.reset()
                self.reset_rnn_cell()
                done = False
                while done == False:
                    obs = torch.tensor(obs,
                                       dtype=torch.float32,
                                       device=self.device).view(1, 1,
                                                                len(obs),
                                                                len(obs[0]))
                    a = self.eps_greedy_action(obs, 0)
                    obs, r, done, _ = self.env.step(a)
                    cum_r += r

        return cum_r / n_eval

    def eps_greedy_action(self, obs, eps):
        if np.random.uniform() < eps:
            return self.env.action_space.sample()
        else:
            return self.qnet.argmax(obs).detach().cpu().numpy().ravel()

    def generate_eps_greedy_trajectory(self, eps):
        self.reset_rnn_cell()
        obs = self.env.reset()
        observations = [obs]
        actions = []
        rewards = []
        dones = []
        done = False
        while done == False:
            obs_tensor = torch.tensor(obs,
                                      dtype=torch.float32,
                                      device=self.device).view(1, 1,
                                                               len(obs),
                                                               len(obs[0]))
            a = self.eps_greedy_action(obs_tensor, eps)
            obs, r, done, _ = self.env.step(a)
            observations.append(obs)
            actions.append(a)
            rewards.append(r)
            dones.append(done)

        return observations, actions, rewards, dones

    def reset_rnn_cell(self):
        self.qnet.h = None
        self.q_target.h = None

    def update_target_net(self):
        self.q_target.load_state_dict(self.qnet.state_dict())

    def to(self, device):
        super().to(device)
        self.qnet.to(device)
        self.q_target.to(device)
        self.device = device

    def cpu(self):
        super().cpu()
        self.qnet.cpu()
        self.q_target.cpu()
        self.device = torch.device('cpu')

    def cuda(self):
        super().cuda()
        self.qnet.cuda()
        self.q_target.cuda()
        self.device = torch.device('cuda')
