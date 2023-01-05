import copy

import numpy as np

import hydra
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import learn2learn as l2l

import utils.utils as utils
from models.rl_regressor import MLPActionPredictor, HyperPolicy, MLPContextEncoder
from utils.dataloader import FastTensorDataLoader


class PolicyApproximator:
    """
    Approximates a family of near-optimal policies.
    Uses either a conditional MLP or a hypernetwork.
    """
    def __init__(self, model, input_dim, state_dim,
                 action_dim, device, lr, embed_dim,
                 hidden_dim, noise_clip, use_clipped_noise):
        self.device = device
        self.lr = lr
        self.use_clipped_noise = use_clipped_noise
        self.noise_clip = noise_clip

        # model
        if model == 'mlp':
            model_fn = MLPActionPredictor
        elif model == 'hyper':
            model_fn = HyperPolicy
        else:
            raise NotImplementedError

        self.policy = model_fn(input_dim,
                               state_dim,
                               action_dim,
                               embed_dim,
                               hidden_dim).to(device)

        # optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.train()
        self.policy.train()

    def train(self, training=True):
        self.training = training
        self.policy.train(training)

    def act(self, input_param, obs):
        input_param = torch.as_tensor(input_param, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy(input_param, obs).cpu().numpy()[0]
        return action.astype(np.float32)

    def eval(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_batches = len(data_loader)
        self.train(True)

        for batch_idx, batch in enumerate(data_loader):
            input_param, state, action, next_state, reward, discount, value = batch

            predicted_action = self.policy(input_param, state)
            loss = F.mse_loss(predicted_action, action)

            metrics['valid/loss_action_pred'] += loss.item()
            metrics['valid/loss_total'] += loss.item()

        for k in metrics.keys():
            metrics[k] /= num_batches
        return metrics

    def update(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_batches = len(data_loader)
        self.train(True)

        for batch_idx, batch in enumerate(data_loader):
            input_param, state, action, next_state, reward, discount, value = batch

            if self.use_clipped_noise:
                # Add clipped noise to the input param
                input_param_noise = (torch.randn_like(input_param)).clamp(-self.noise_clip, self.noise_clip)
                input_param += input_param_noise

            predicted_action = self.policy(input_param, state)

            loss = F.mse_loss(predicted_action, action)

            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            metrics['train/loss_action_pred'] += loss.item()
            metrics['train/loss_total'] += loss.item()

        for k in metrics.keys():
            metrics[k] /= num_batches
        return metrics

    def save(self, model_dir, name):
        model_save_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.policy.state_dict(), f'{model_save_dir}/policy.pt')

    def load(self, model_dir, name):
        print(f"Loading the model from {model_dir}, name: {name}")
        model_load_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')

        self.policy.load_state_dict(
            torch.load(f'{model_load_dir}/policy.pt', map_location=self.device)
        )


class MetaPolicyApproximator(PolicyApproximator):
    """
    Approximates a family of near-optimal policies.
    Uses either MAML or PEARL.
    """
    def __init__(self, model, input_dim, state_dim,
                 action_dim, device, lr, fast_lr, embed_dim,
                 hidden_dim, noise_clip, use_clipped_noise,
                 adaptation_steps, use_pearl, kl_lambda):
        super().__init__(model, input_dim, state_dim,
                         action_dim, device, lr, embed_dim,
                         hidden_dim, noise_clip, use_clipped_noise)
        assert model == 'mlp', "MAML only works with MLP policy"
        assert not use_clipped_noise, "MAML/PEARL cannot use clipped noise."
        del self.policy
        del self.policy_optimizer

        self.adaptation_steps = adaptation_steps
        self.use_pearl = use_pearl
        self.kl_lambda = kl_lambda
        self.prev_action = None
        self.prev_state = None
        self.action_dim = action_dim
        self.state_dim = state_dim

        if self.use_pearl:
            # PEARL baseline
            context_dim = embed_dim
            self.context_encoder = MLPContextEncoder(state_dim,
                                                     action_dim,
                                                     embed_dim,
                                                     hidden_dim).to(device)
            self.context_encoder_optimizer = torch.optim.Adam(self.context_encoder.parameters(), lr=lr)
        else:
            # MAML baseline
            context_dim = input_dim

        policy = MLPActionPredictor(context_dim,
                                    state_dim,
                                    action_dim,
                                    embed_dim,
                                    hidden_dim).to(device)
        self.policy = l2l.algorithms.MAML(policy, lr=fast_lr, first_order=False)

        # optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.train()
        self.policy.train()

    def fast_adapt(self, learner, batch):
        eval_metrics = dict()
        batch_size = batch[0].shape[0]

        # separate data into adaptation/evaluation sets
        adaptation_indices = np.zeros(batch_size, dtype=bool)
        adaptation_indices[np.arange(batch_size // 2)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)

        adapt_input_param, adapt_state, adapt_action, adapt_next_state, adapt_reward, adapt_discount, adapt_value = utils.select_indices(
            batch, adaptation_indices)
        eval_input_param, eval_state, eval_action, eval_next_state, eval_reward, eval_discount, eval_value = utils.select_indices(
            batch, evaluation_indices)

        # adapt the model
        for step in range(self.adaptation_steps):
            if self.use_pearl:
                adapt_context, _, _ = self.context_encoder(adapt_state, adapt_action)
                adapt_predicted_action = learner(adapt_context, adapt_state)
            else:
                adapt_predicted_action = learner(adapt_input_param, adapt_state)
            adapt_loss = F.mse_loss(adapt_predicted_action, adapt_action)
            learner.adapt(adapt_loss)

        # evaluate the adapted model
        if self.use_pearl:
            eval_context, eval_mu, eval_log_var = self.context_encoder(eval_state, eval_action)
            eval_predicted_action = learner(eval_context, eval_state)
            eval_loss = F.mse_loss(eval_predicted_action, eval_action)

            # compute KL loss
            kl_div = self.compute_kl_div(eval_mu, eval_log_var)
            kl_loss = self.kl_lambda * kl_div
            eval_loss += kl_loss
            eval_metrics['loss_kl'] = kl_loss.item()

        else:
            eval_predicted_action = learner(eval_input_param, eval_state)
            eval_loss = F.mse_loss(eval_predicted_action, eval_action)

        eval_metrics['loss_action_pred'] = eval_loss.item()
        eval_metrics['loss_total'] = eval_loss.item()

        return eval_loss, eval_metrics

    def compute_kl_div(self, mu, log_var):
        assert self.use_pearl
        kl_div = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        return kl_div.mean()

    def act(self, input_param, obs):
        if self.use_pearl:
            # uses previous action and state to infer the context
            if self.prev_action is None:
                self.prev_action = torch.zeros(1, self.action_dim).to(self.device)
            if self.prev_state is None:
                self.prev_state = torch.zeros(1, self.state_dim).to(self.device)

            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            context, _, _ = self.context_encoder(self.prev_state, self.prev_action)
            action = self.policy(context, obs)

            self.prev_action = action.clone()
            self.prev_state = obs.clone()

            action = action.cpu().numpy()[0]
        else:
            input_param = torch.as_tensor(input_param, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.policy(input_param, obs).cpu().numpy()[0]

        return action.astype(np.float32)

    def eval(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_tasks = data_loader.n_tasks
        self.train(True)

        for task_idx in range(num_tasks):
            learner = self.policy.clone()
            batch = data_loader.sample(task_idx)

            _, meta_loss_logs = self.fast_adapt(learner, batch)

            for k, v in meta_loss_logs.items():
                metrics[f'valid/{k}'] += v

        for k in metrics.keys():
            metrics[k] /= num_tasks
        return metrics

    def finetune(self, input_param, state, action):
        input_param = torch.as_tensor(input_param, dtype=torch.float32, device=self.device)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        data_loader = FastTensorDataLoader(input_param, state, action, batch_size=512, shuffle=True, device=self.device)
        finetuner = torch.optim.Adam(self.policy.parameters(), lr=0.001)

        # Updates the actual policy!
        for step in range(self.adaptation_steps):
            for batch_idx, batch in enumerate(data_loader):
                batch_input_param, batch_state, batch_action = batch
                if self.use_pearl:
                    adapt_context, _, _ = self.context_encoder(batch_state, batch_action)
                    adapt_predicted_action = self.policy(adapt_context, batch_state)
                else:
                    adapt_predicted_action = self.policy(batch_input_param, batch_state)
                adapt_loss = F.mse_loss(adapt_predicted_action, batch_action)

                # self.policy.adapt(adapt_loss)
                finetuner.zero_grad()
                adapt_loss.backward()
                finetuner.step()

    def update(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_tasks = data_loader.n_tasks
        self.train(True)

        self.policy_optimizer.zero_grad()
        if self.use_pearl:
            self.context_encoder_optimizer.zero_grad()

        for task_idx in range(num_tasks):
            # compute meta-training loss
            learner = self.policy.clone()
            batch = data_loader.sample(task_idx)

            meta_loss, meta_loss_logs = self.fast_adapt(learner, batch)
            meta_loss.backward()

            for k, v in meta_loss_logs.items():
                metrics[f'train/{k}'] += v

        for p in self.policy.parameters():
            p.grad.data.mul_(1.0 / num_tasks)
        self.policy_optimizer.step()

        if self.use_pearl:
            for p in self.context_encoder.parameters():
                p.grad.data.mul_(1.0 / num_tasks)
            self.context_encoder_optimizer.step()

        for k in metrics.keys():
            metrics[k] /= num_tasks
        return metrics
