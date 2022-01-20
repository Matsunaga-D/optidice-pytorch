import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _mlp(sizes, activation, output_activation):
    # Ref: OpenAI Spinning Up
    layers = []
    for j in range(len(sizes)-1):
        activ = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activ()]
    return nn.Sequential(*layers)


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
    def forward(self):
        pass

