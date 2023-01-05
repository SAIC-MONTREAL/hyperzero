import hydra
import torch
import torch.nn as nn

import utils.utils as utils
from models.hypenet_core import HyperNetwork, DoubleHeadedHyperNetwork, Meta_Embadding


class HyperPolicy(nn.Module):
    """
    Approximates the mapping R(\phi) -> \pi^* (a|s)
    """
    def __init__(self, input_param_dim, state_dim, action_dim, embed_dim, hidden_dim):
        super().__init__()

        self.hyper_policy = HyperNetwork(
            meta_v_dim=input_param_dim,
            z_dim=embed_dim,
            base_v_input_dim=state_dim,
            base_v_output_dim=action_dim,
            dynamic_layer_dim=hidden_dim,
            base_output_activation=torch.tanh
        )

    def forward(self, input_param, state):
        z, action = self.hyper_policy(input_param, state)
        return action


class HyperRLSolution(nn.Module):
    """
    Baseline. Approximates the mapping R(\phi) -> Q^*(s, a), \pi^*(s)
    """
    def __init__(self, input_param_dim, state_dim, action_dim, embed_dim, hidden_dim):
        super().__init__()

        self.hyper_rl_net = DoubleHeadedHyperNetwork(
            meta_v_dim=input_param_dim,
            z_dim=embed_dim,
            base_v_input_dim=[state_dim, state_dim + action_dim],
            base_v_output_dim=[action_dim, 1],
            dynamic_layer_dim=hidden_dim,
            base_output_activation=[torch.tanh, None]
        )

    def forward(self, input_param, state, action):
        state_action = torch.cat([state, action], dim=-1)
        z, pred_action, q_value = self.hyper_rl_net(input_param, state, state_action)
        return z, pred_action, q_value

    def embed_task(self, input_param):
        z = self.hyper_rl_net.embed(input_param)
        return z

    def predict_action(self, z, state):
        pred_action = self.hyper_rl_net.forward_net_1(z, state)
        return pred_action

    def predict_q_value(self, z, state, action):
        state_action = torch.cat([state, action], dim=-1)
        q_value = self.hyper_rl_net.forward_net_2(z, state_action)
        return q_value


class MLPActionPredictor(nn.Module):
    """
    Baseline. Approximates the mapping R(\phi) -> \pi^*(s)
    """
    def __init__(self, input_param_dim, state_dim, action_dim, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = Meta_Embadding(input_param_dim, embed_dim)

        self.policy_net = nn.Sequential(
            nn.Linear(embed_dim + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(utils.weight_init)

    def forward(self, input_param, state):
        emb = self.embedding(input_param)
        emb_states = torch.cat([emb, state], dim=-1)
        action = self.policy_net(emb_states)
        return torch.tanh(action)


class MLPRLSolution(nn.Module):
    """
    Baseline. Approximates the mapping R(\phi) -> Q^*(s, a), \pi^*(s)
    """
    def __init__(self, input_param_dim, state_dim, action_dim, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = Meta_Embadding(input_param_dim, embed_dim)

        self.q_net = nn.Sequential(
            nn.Linear(embed_dim + state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.policy_net = nn.Sequential(
            nn.Linear(embed_dim + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, input_param, state, action):
        task = self.embed_task(input_param)
        pred_action = self.predict_action(task, state)
        q_value = self.predict_q_value(task, state, action)
        return task, pred_action, q_value

    def embed_task(self, input_param):
        task = self.embedding(input_param)
        return task

    def predict_action(self, task, state):
        task_state = torch.cat([task, state], dim=-1)
        pred_action = self.policy_net(task_state)
        return torch.tanh(pred_action)

    def predict_q_value(self, task, state, action):
        task_state_action = torch.cat([task, state, action], dim=-1)
        q_value = self.q_net(task_state_action)
        return q_value


class MLPContextEncoder(nn.Module):
    """
    Context encoder of PEARL.
    """
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, embed_dim)
        self.log_var = nn.Linear(hidden_dim, embed_dim)

        self.apply(utils.weight_init)

    def encode(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        z = self.fc(state_action)
        return self.mu(z), self.log_var(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, state, action):
        mu, log_var = self.encode(state, action)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
