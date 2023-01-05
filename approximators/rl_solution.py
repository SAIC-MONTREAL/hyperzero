import hydra
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import learn2learn as l2l

from models.rl_regressor import MLPRLSolution, HyperRLSolution
import utils.utils as utils


class RLApproximator:
    """
    Approximates a family of near-optimal Rl solutions.
    Uses either a conditional MLP or a hypernetwork.
    """
    def __init__(self, model, input_dim, state_dim, action_dim,
                 device, lr, embed_dim, hidden_dim, noise_clip,
                 use_clipped_noise, use_td, td_weight, value_weight):
        self.device = device
        self.lr = lr
        self.model = model
        self.use_td_error = use_td
        self.use_clipped_noise = use_clipped_noise
        self.noise_clip = noise_clip
        self.td_weight = td_weight
        self.value_weight = value_weight

        # model
        if model == 'mlp':
            model_fn = MLPRLSolution
        elif model == 'hyper':
            model_fn = HyperRLSolution
        else:
            raise NotImplementedError

        self.rl_net = model_fn(input_dim,
                               state_dim,
                               action_dim,
                               embed_dim,
                               hidden_dim).to(device)

        # optimizer
        self.rl_net_optimizer = torch.optim.Adam(self.rl_net.parameters(), lr=lr)

        self.train()
        self.rl_net.train()

    def train(self, training=True):
        self.training = training
        self.rl_net.train(training)

    def act(self, input_param, obs):
        input_param = torch.as_tensor(input_param, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        task_emb = self.rl_net.embed_task(input_param)
        action = self.rl_net.predict_action(task_emb, obs).cpu().numpy()[0]
        return action.astype(np.float32)

    def q(self, input_param, obs, action):
        input_param = torch.as_tensor(input_param, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        task_emb = self.rl_net.embed_task(input_param)
        q = self.rl_net.predict_q_value(task_emb, obs, action).cpu().numpy()[0]
        return q.astype(np.float32)

    def eval(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_batches = len(data_loader)
        self.train(True)

        for batch_idx, batch in enumerate(data_loader):
            input_param, state, action, next_state, reward, discount, value = batch

            task_emb, predicted_action, predicted_value = self.rl_net(input_param, state, action)

            loss_action = F.mse_loss(predicted_action, action)
            loss_value = F.mse_loss(predicted_value, value)
            loss = loss_action + self.value_weight * loss_value
            # evaluate the TD error in any case
            loss_td = self.get_td_error(task_emb, next_state, reward, discount, value)

            if self.use_td_error:
                loss += self.td_weight * loss_td

            metrics['valid/loss_action_pred'] += loss_action.item()
            metrics['valid/loss_value_pred'] += self.value_weight * loss_value.item()
            metrics['valid/loss_td'] += self.td_weight * loss_td.item()
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
                # Add clipped noise to the reward param
                input_param_noise = (torch.randn_like(input_param)).clamp(-self.noise_clip, self.noise_clip)
                input_param += input_param_noise

            task_emb, predicted_action, predicted_value = self.rl_net(input_param, state, action)

            loss_action = F.mse_loss(predicted_action, action)
            loss_value = F.mse_loss(predicted_value, value)
            loss = loss_action + self.value_weight * loss_value

            if self.use_td_error:
                loss_td = self.get_td_error(task_emb, next_state, reward, discount, value)
                loss += self.td_weight * loss_td
                metrics['train/loss_td'] += self.td_weight * loss_td.item()

            self.rl_net_optimizer.zero_grad()
            loss.backward()
            self.rl_net_optimizer.step()

            metrics['train/loss_action_pred'] += loss_action.item()
            metrics['train/loss_value_pred'] += self.value_weight * loss_value.item()
            metrics['train/loss_total'] += loss.item()

        for k in metrics.keys():
            metrics[k] /= num_batches
        return metrics

    def get_td_error(self, task_emb, next_state, reward, discount, q):
        with torch.no_grad():
            next_action = self.rl_net.predict_action(task_emb, next_state)
        target_q = self.rl_net.predict_q_value(task_emb, next_state, next_action)
        target_q = reward + discount * target_q

        td_error = F.mse_loss(q, target_q)
        return td_error

    def save(self, model_dir, name):
        model_save_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.rl_net.state_dict(), f'{model_save_dir}/rl_net.pt')

    def load(self, model_dir, name):
        print(f"Loading the model from {model_dir}, name: {name}")
        model_load_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')

        self.rl_net.load_state_dict(
            torch.load(f'{model_load_dir}/rl_net.pt', map_location=self.device)
        )


class MetaRLApproximator(RLApproximator):
    """
    Approximates a family of near-optimal policies. Uses MAML.
    """
    def __init__(self, model, input_dim, state_dim, action_dim,
                 device, lr, fast_lr, embed_dim, hidden_dim, noise_clip,
                 use_clipped_noise, use_td, td_weight, value_weight,
                 adaptation_steps):
        super().__init__(model, input_dim, state_dim, action_dim,
                         device, lr, embed_dim, hidden_dim, noise_clip,
                         use_clipped_noise, use_td, td_weight, value_weight)
        assert model == 'mlp', "MAML only works with MLP RL approximator"
        assert not use_clipped_noise, "MAML cannot use clipped noise."
        del self.rl_net
        del self.rl_net_optimizer

        self.adaptation_steps = adaptation_steps

        # MAML model
        policy = MLPRLSolution(input_dim,
                               state_dim,
                               action_dim,
                               embed_dim,
                               hidden_dim).to(device)
        self.rl_net = l2l.algorithms.MAML(policy, lr=fast_lr, first_order=False)

        # optimizer
        self.rl_net_optimizer = torch.optim.Adam(self.rl_net.parameters(), lr=lr)

        self.train()
        self.rl_net.train()

    def fast_adapt(self, learner, batch):
        eval_metrics = dict()
        batch_size = batch[0].shape[0]

        # separate data into adaptation/evalutation sets
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
            adapt_task_emb, adapt_predicted_action, adapt_predicted_value = learner(adapt_input_param, adapt_state, adapt_action)

            adapt_action_loss = F.mse_loss(adapt_predicted_action, adapt_action)
            adapt_value_loss = F.mse_loss(adapt_predicted_value, adapt_value)
            adapt_loss = adapt_action_loss + self.value_weight * adapt_value_loss

            if self.use_td_error:
                adapt_loss_td = self.get_td_error(adapt_task_emb, adapt_next_state, adapt_reward, adapt_discount, adapt_value)
                adapt_loss += self.td_weight * adapt_loss_td

            learner.adapt(adapt_loss)

        # evaluate the adapted model
        eval_task_emb, eval_predicted_action, eval_predicted_value = learner(eval_input_param, eval_state, eval_action)

        eval_action_loss = F.mse_loss(eval_predicted_action, eval_action)
        eval_value_loss = F.mse_loss(eval_predicted_value, eval_value)
        eval_loss = eval_action_loss + self.value_weight * eval_value_loss

        if self.use_td_error:
            eval_loss_td = self.get_td_error(eval_task_emb, eval_next_state, eval_reward, eval_discount, eval_value)
            eval_loss += self.td_weight * eval_loss_td

        eval_metrics['loss_action_pred'] = eval_action_loss.item()
        eval_metrics['loss_value_pred'] = eval_value_loss.item()
        eval_metrics['loss_total'] = eval_loss.item()

        return eval_loss, eval_metrics

    def eval(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_tasks = data_loader.n_tasks
        num_batches = len(data_loader)
        self.train(True)

        for batch_idx in range(num_batches):
            for task_idx in range(num_tasks):
                learner = self.rl_net.clone()
                batch = data_loader.sample(task_idx)

                _, meta_loss_logs = self.fast_adapt(learner, batch)

                for k, v in meta_loss_logs.items():
                    metrics[f'valid/{k}'] += v

        for k in metrics.keys():
            metrics[k] /= (num_tasks * num_batches)
        return metrics

    def update(self, data_loader):
        metrics = defaultdict(lambda: 0)

        num_tasks = data_loader.n_tasks
        num_batches = len(data_loader)
        self.train(True)

        for batch_idx in range(num_batches):
            self.rl_net_optimizer.zero_grad()
            for task_idx in range(num_tasks):
                # compute meta-training loss
                learner = self.rl_net.clone()
                batch = data_loader.sample(task_idx)

                meta_loss, meta_loss_logs = self.fast_adapt(learner, batch)
                meta_loss.backward()

                for k, v in meta_loss_logs.items():
                    metrics[f'train/{k}'] += v

            for p in self.rl_net.parameters():
                p.grad.data.mul_(1.0 / num_tasks)
            self.rl_net_optimizer.step()

        for k in metrics.keys():
            metrics[k] /= (num_tasks * num_batches)
        return metrics