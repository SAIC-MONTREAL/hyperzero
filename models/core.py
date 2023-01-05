import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils


def gaussian_logprob(noise, log_std):
    """
    Compute Gaussian log probability.
    """
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """
    Apply squashing function.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class DeterministicActor(nn.Module):
    """
    Original TD3 actor.
    """
    def __init__(self, feature_dim, action_dim, hidden_dim):
        super(DeterministicActor, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

        self.apply(utils.weight_init)

    def forward(self, state):
        a = self.policy(state)
        return torch.tanh(a)


class Critic(nn.Module):
    """
    Original TD3 critic.
    """
    def __init__(self, feature_dim, action_dim, hidden_dim):
        super().__init__()

        # Q1 architecture
        self.Q1_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.Q2_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(utils.weight_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.Q1_net(sa)
        q2 = self.Q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.Q1_net(sa)
        return q1
