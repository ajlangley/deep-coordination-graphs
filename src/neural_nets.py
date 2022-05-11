import torch
from torch import nn


class FactoredMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, K, n_actions):
        super().__init__()
        f1 = build_mlp(input_size, hidden_sizes, K * n_actions)
        f2 = build_mlp(input_size, hidden_sizes, K * n_actions)
        self.factors = nn.ModuleList([f1, f2])
        self.K = K
        self.n_actions = n_actions

    def __call__(self, x):
        f1_output = self.factors[0](x)
        f2_output = self.factors[1](x)
        f1_output = f1_output.view(-1, self.K, self.n_actions)
        f2_output = f2_output.view(-1, self.K, self.n_actions)

        if x.dim() == 2:
            net_output = torch.einsum('ijk,ijl->ikl', f1_output, f2_output)
        else:
            net_output = torch.einsum('ij,ik->jk', f1_output, f2_output)

        net_output = torch.flatten(net_output, start_dim=-2)

        return net_output
        

class GRUFeatExtractor(nn.Module):
    def __init__(self, input_size, h_size):
        super().__init__()
        self.rnn = nn.GRUCell(input_size, h_size)
        self.h = torch.zeros(h_size)
        self.device = torch.device('cpu')

    def __call__(self, x):
        self.h = self.rnn(x, self.h)

        return self.h

    def reset(self, batch_size=1, h0=None):
        if h0 is None:
            self.h = torch.ones((batch_size, self.h.size(-1)),
                                dtype=torch.float,
                                device=self.device)
            if batch_size == 1:
                self.h = self.h.ravel()
        else:
            self.h = h0

    def to(self, device):
        super().to(device)
        self.h = self.h.to(device)
        self.device = device


def build_mlp(input_size, hidden_sizes, output_size, Activation=nn.LeakyReLU):
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for l, (in_feat, out_feat) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), 2):
        layers.append(nn.Linear(in_features=in_feat,
                                out_features=out_feat,
                                dtype=torch.float))
        if l != len(layer_sizes):
            layers.append(Activation())

    return nn.Sequential(*layers)
