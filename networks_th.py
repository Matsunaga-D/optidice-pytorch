import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


def _mlp(sizes, activation, output_activation):
    # Ref: OpenAI Spinning Up
    layers = []
    for j in range(len(sizes)-1):
        activ = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activ()]
    return nn.Sequential(*layers)


@torch.no_grad()
def _init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        # m.bias.data._fill(0.01)


@torch.no_grad()
def _init_weights_last_layer(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-1e-3, b=1e-3)
        torch.nn.init.uniform_(m.bias, a=-1e-3, b=1e-3)


class TanhNormalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, name='TanhNormalPolicy',
                 mean_range=(-7., 7.), logstd_range=(-5., 2.), eps=1e-6):
        super(TanhNormalPolicy, self).__init__()

        self._state_dim = state_dim
        self._action_dim = action_dim

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

        self._fc_layers = _mlp([state_dim] + list(hidden_sizes), activation=nn.ReLU,
                            output_activation=nn.ReLU)
        self._fc_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self._fc_logstd = nn.Linear(hidden_sizes[-1], action_dim)

        self._fc_layers.apply(_init_weights)
        self._fc_mean.apply(_init_weights_last_layer)
        self._fc_logstd.apply(_init_weights_last_layer)

    def forward(self, state, deterministic=False, with_logprob=True):
        net_out = self._fc_layers(state)
        mu = self._fc_mean(net_out)
        logstd = self._fc_logstd(net_out)

        mu = torch.clamp(mu, self.mean_min, self.mean_max)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)

        pi_distr = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distr.rsample()

        if with_logprob:
            log_pi = pi_distr.log_prob(pi_action).sum(axis=-1)
            log_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1) # TODO: check with tf ver.
        else:
            log_pi = None

        pi_action = torch.tanh(pi_action)

        return pi_action, log_pi


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes, output_activation_fn=None, output_dim=None, name='ValueNetwork'):
        super(ValueNetwork, self).__init__()