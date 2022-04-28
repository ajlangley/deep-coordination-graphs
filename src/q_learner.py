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
        loss = self.loss_fun(q_pred, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evaluate(self, n_eval):
        self.qnet.eval()
        cum_r = 0
        with torch.no_grad():
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
        self.qnet.actions = self.qnet.actions.to(device)
        self.device = device

    def cpu(self):
        super().cpu()
        self.actions = self.actions.cpu()

    def cuda(self):
        super().cpu()
        self.actions = self.actions.cuda()
