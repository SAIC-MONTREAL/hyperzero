import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from numpy.random import default_rng

import torch
from torch.utils.data import TensorDataset


def _get_reward_param(data_dir, domain_task, seed, test_fraction):
    """
    Helper function to get the reward parameter from an .npy rollout file.
    """
    datadir = Path(data_dir)
    paths_to_load = sorted(datadir.glob(f'**/{domain_task}*_seed_*.npy'))

    # get reward parameters in a super hacked up way
    all_reward_params = sorted(list(set([re.findall('\-(.*?)\_', str(p))[0] for p in paths_to_load])))
    # remove margin=0.0 from reward params
    reward_params = [x for x in all_reward_params if '-0.0' not in x]

    # Random picking based on the seed
    rng = default_rng(seed)
    rng.shuffle(reward_params)
    test_size = int(test_fraction * len(reward_params))
    train_reward_params = reward_params[test_size:]
    test_reward_params = reward_params[:test_size]

    return train_reward_params, test_reward_params


def _get_dynamic_param(data_dir, domain_task, seed, test_fraction):
    """
    Helper function to get the dynamics parameter from an .npy rollout file.
    """
    datadir = Path(data_dir)
    paths_to_load = sorted(datadir.glob(f'**/{domain_task}*_seed_*.npy'))

    # get dynamic parameters in a super hacked up way
    dynamic_params = sorted(list(set([re.findall('\_dyn_(.*?)\__', str(p))[0] for p in paths_to_load])))

    # Random picking based on the seed
    rng = default_rng(seed)
    rng.shuffle(dynamic_params)
    test_size = int(test_fraction * len(dynamic_params))
    train_dynamic_params = dynamic_params[test_size:]
    test_dynamic_params = dynamic_params[:test_size]

    return train_dynamic_params, test_dynamic_params


def _get_reward_dynamic_param(data_dir, domain_task, seed, test_fraction):
    """
    Helper function to get the reward and dynamics parameters from an .npy rollout file.
    """
    datadir = Path(data_dir)
    paths_to_load = sorted(datadir.glob(f'**/{domain_task}*_seed_*.npy'))

    # get reward-dynamic parameters in a super hacked up way
    reward_dynamic_params = sorted(list(set([re.findall('\-(.*?)\__', str(p))[0] for p in paths_to_load])))

    # Random picking based on the seed
    rng = default_rng(seed)
    rng.shuffle(reward_dynamic_params)
    test_size = int(test_fraction * len(reward_dynamic_params))
    train_reward_dynamic_params = reward_dynamic_params[test_size:]
    test_reward_dynamic_params = reward_dynamic_params[:test_size]

    return train_reward_dynamic_params, test_reward_dynamic_params


class RLSolutionDataset:
    """
    Dataset of near-optimal trajectories on a family of MDPs.
    Used for training HyperZero and MLP baselines.
    """
    def __init__(self, data_dir, domain_task, input_to_model, seed, device):
        assert input_to_model in ['rew', 'dyn', 'rew_dyn']
        self.data_dir = data_dir
        self.domain_task = domain_task
        self.input_to_model = input_to_model
        self.device = device
        self.test_fraction = 0.15
        self.seed = seed

        # set the data keys
        if input_to_model == 'rew':
            self.data_keys = ['reward_param', 'state', 'action', 'next_state', 'reward', 'discount', 'value']
            self.train_input_params, self.test_input_params = _get_reward_param(data_dir, domain_task, self.seed, self.test_fraction)
        elif input_to_model == 'dyn':
            self.data_keys = ['dynamics_param', 'state', 'action', 'next_state', 'reward', 'discount', 'value']
            self.train_input_params, self.test_input_params = _get_dynamic_param(data_dir, domain_task, self.seed, self.test_fraction)
        elif input_to_model == 'rew_dyn':
            self.data_keys = ['reward_dynamics_param', 'state', 'action', 'next_state', 'reward', 'discount', 'value']
            self.train_input_params, self.test_input_params = _get_reward_dynamic_param(data_dir, domain_task, self.seed, self.test_fraction)

        self.setup()

    def setup(self):
        train_tensors, test_tensors = [], []

        # load the dataset
        self.train_data_np, self.test_data_np = self._load_dataset(flatten=True)

        # concatenate reward and dynamic parameters
        self.train_data_np['reward_dynamics_param'] = np.concatenate((self.train_data_np['reward_param'],
                                                                      self.train_data_np['dynamics_param']),
                                                                     axis=-1)
        self.test_data_np['reward_dynamics_param'] = np.concatenate((self.test_data_np['reward_param'],
                                                                     self.test_data_np['dynamics_param']),
                                                                    axis=-1)

        for k in self.data_keys:
            train_tensors.append(
                torch.tensor(self.train_data_np[k], dtype=torch.float, device=self.device)
            )
            test_tensors.append(
                torch.tensor(self.test_data_np[k], dtype=torch.float, device=self.device)
            )

        self.train_dataset = TensorDataset(*train_tensors)
        self.test_dataset = TensorDataset(*test_tensors)

    def _load_dataset(self, flatten=False):
        train_data, test_data = self._generate_data(flatten)
        return train_data, test_data

    def _generate_data(self, flatten=False):
        datadir = Path(self.data_dir)
        train_data_np, test_data_np = defaultdict(list), defaultdict(list)
        data = {
            'train': train_data_np,
            'test': test_data_np
        }

        for stage, input_params in zip(['train', 'test'],
                                       [self.train_input_params, self.test_input_params]):
            for r in input_params:
                paths_to_load = sorted(datadir.glob(f'**/{self.domain_task}*_seed_*{r}_*.npy'))

                for p in paths_to_load:
                    print(f"Loading data from {str(p)}")
                    d = np.load(str(p), allow_pickle=True).item()
                    for k, v in d.items():
                        if flatten:
                            # save the data as (n_episodes * n_steps, ?)
                            n_episodes, n_steps = v.shape[0], v.shape[1]
                            data[stage][k].append(v.reshape(n_episodes * n_steps, -1))
                        else:
                            # save the data as (n_episodes, n_steps, ?)
                            data[stage][k].append(v)

            # concatenate the loaded data
            for k, v in data[stage].items():
                data[stage][k] = np.concatenate(v, axis=0)

            with open(os.path.join(self.data_dir, f'{stage}-{self.input_to_model}-params-seed-{self.seed}.txt'), 'w') as f:
                f.write(str(input_params))

        return data['train'], data['test']

    @property
    def reward_param_dim(self):
        return self.train_data_np['reward_param'].shape[-1]

    @property
    def dynamic_param_dim(self):
        return self.train_data_np['dynamics_param'].shape[-1]

    @property
    def reward_dynamic_param_dim(self):
        return self.train_data_np['reward_dynamics_param'].shape[-1]

    @property
    def state_dim(self):
        return self.train_data_np['state'].shape[-1]

    @property
    def action_dim(self):
        return self.train_data_np['action'].shape[-1]


class RLSolutionMetaDataset(RLSolutionDataset):
    """
    Dataset of near-optimal trajectories on a family of MDPs.
    Used for training meta learning (MAML and PEARL) baselines.
    """
    def __init__(self, data_dir, domain_task, input_to_model, seed, device):
        super().__init__(data_dir, domain_task, input_to_model, seed, device)

    def _load_dataset(self, flatten=False):
        meta_train_data, meta_test_data = self._generate_data(flatten)
        return meta_train_data, meta_test_data

    def _generate_data(self, flatten=False):
        datadir = Path(self.data_dir)
        train_data_np, test_data_np = defaultdict(list), defaultdict(list)
        data = {
            'train': train_data_np,
            'test': test_data_np
        }

        for stage, input_params in zip(['train', 'test'],
                                       [self.train_input_params, self.test_input_params]):
            for r in input_params:
                paths_to_load = sorted(datadir.glob(f'**/{self.domain_task}*_seed_*{r}_*.npy'))

                for p in paths_to_load:
                    print(f"Loading data from {str(p)}")
                    d = np.load(str(p), allow_pickle=True).item()
                    for k, v in d.items():
                        if flatten:
                            # save the data as (n_episodes * n_steps, ?)
                            n_episodes, n_steps = v.shape[0], v.shape[1]
                            data[stage][k].append(v.reshape(n_episodes * n_steps, -1))
                        else:
                            # save the data as (n_episodes, n_steps, ?)
                            data[stage][k].append(v)

            # concatenate the loaded data
            for k, v in data[stage].items():
                data[stage][k] = np.stack(v, axis=0)       # note the difference from the standard dataset

            with open(os.path.join(self.data_dir, f'meta-{stage}-{self.input_to_model}-params-seed-{self.seed}.txt'), 'w') as f:
                f.write(str(input_params))
        return data['train'], data['test']

    @property
    def n_tasks(self):
        return self.train_data_np['state'].shape[0]
