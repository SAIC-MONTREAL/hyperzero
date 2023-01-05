"""
RL approximator training loop.
Works on DMC with states and pixel observations.
"""

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import platform
import logging
import math

if platform.system() == 'Linux':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.dataset import RLSolutionDataset, RLSolutionMetaDataset
from utils.dataloader import FastTensorDataLoader, FastTensorMetaDataLoader

torch.backends.cudnn.benchmark = True

# If using multirun, set the GPUs here:
AVAILABLE_GPUS = [1, 2, 3, 4, 0]


def make_approximator(input_dim, state_dim, action_dim, cfg, device=None):
    cfg.input_dim = input_dim
    cfg.state_dim = state_dim
    cfg.action_dim = action_dim
    if device is not None:
        cfg.device = device
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # hacked up way to see if we are using MAML or not
        self.is_meta_learning = True if 'meta' in self.cfg.approximator_name else False

        self.setup()

        if cfg.input_to_model == 'rew':
            input_dim = self.dataset.reward_param_dim
        elif cfg.input_to_model == 'dyn':
            input_dim = self.dataset.dynamic_param_dim
        elif cfg.input_to_model == 'rew_dyn':
            input_dim = self.dataset.reward_dynamic_param_dim
        else:
            raise NotImplementedError

        self.approximator = make_approximator(input_dim,
                                              self.dataset.state_dim,
                                              self.dataset.action_dim,
                                              self.cfg.approximator)
        self.timer = utils.Timer()
        self._global_epoch = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = SummaryWriter(str(self.work_dir))

        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        self.rollout_dir = Path(self.cfg.rollout_dir).expanduser().joinpath(self.cfg.domain_task)

        # load dataset
        self.load_dataset()

        # save cfg and git sha
        utils.save_cfg(self.cfg, self.work_dir)
        utils.save_git_sha(self.work_dir)

    def load_dataset(self):
        dataset_fn = RLSolutionMetaDataset if self.is_meta_learning else RLSolutionDataset
        dataloader_fn = FastTensorMetaDataLoader if self.is_meta_learning else FastTensorDataLoader

        self.dataset = dataset_fn(
            self.rollout_dir,
            self.cfg.domain_task,
            self.cfg.input_to_model,
            self.cfg.seed,
            self.device,
        )

        if self.is_meta_learning:
            batch_size = int(self.dataset.n_tasks * self.cfg.k_shot * 2)
        else:
            batch_size = self.cfg.batch_size

        self.train_loader = dataloader_fn(*self.dataset.train_dataset[:], device=self.device,
                                          batch_size=batch_size, shuffle=True)
        self.test_loader = dataloader_fn(*self.dataset.test_dataset[:], device=self.device,
                                         batch_size=batch_size, shuffle=True)

    @property
    def global_epoch(self):
        return self._global_epoch

    def train(self):
        # predicates
        train_until_epoch = utils.Until(self.cfg.num_train_epochs)
        save_every_epoch = utils.Every(self.cfg.save_every_frames)

        metrics = dict()
        best_valid_total_loss = math.inf
        best_valid_value_loss = math.inf
        best_valid_action_loss = math.inf
        best_valid_td_loss = math.inf

        while train_until_epoch(self.global_epoch):
            metrics.update()

            if self.is_meta_learning:
                self.train_loader.shuffle_indices()
                self.test_loader.shuffle_indices()

            metrics.update(self.approximator.update(self.train_loader))
            metrics.update(self.approximator.eval(self.test_loader))

            # Log metrics
            print(f"Epoch {self.global_epoch + 1} "
                  f"\t Train loss {metrics['train/loss_total']:.3f} "
                  f"\t Valid loss {metrics['valid/loss_total']:.3f}")
            for k, v in metrics.items():
                self.logger.add_scalar(k, v, self.global_epoch + 1)
            utils.dump_dict(f"{self.work_dir}/train_valid.csv", metrics)

            # Save the model
            if metrics['valid/loss_total'] <= best_valid_total_loss:
                best_valid_total_loss = metrics['valid/loss_total']
                self.approximator.save(self.model_dir, 'best_total')
            if metrics['valid/loss_action_pred'] <= best_valid_action_loss:
                best_valid_action_loss = metrics['valid/loss_action_pred']
                self.approximator.save(self.model_dir, 'best_action')
            if 'valid/loss_value_pred' in metrics:
                if metrics['valid/loss_value_pred'] <= best_valid_value_loss:
                    best_valid_value_loss = metrics['valid/loss_value_pred']
                    self.approximator.save(self.model_dir, 'best_value')
            if 'valid/loss_td' in metrics:
                if metrics['valid/loss_td'] <= best_valid_td_loss:
                    best_valid_td_loss = metrics['valid/loss_td']
                    self.approximator.save(self.model_dir, 'best_td')

            if save_every_epoch(self.global_epoch + 1):
                self.approximator.save(self.model_dir, self.global_epoch + 1)

            self._global_epoch += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(version_base=None, config_path='cfgs', config_name='config_rl_approximator')
def main(cfg):
    log = logging.getLogger(__name__)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        cfg.device = f"{cfg.device}:{device_id}"
        log.info(f"Total number of GPUs is {AVAILABLE_GPUS}, running on {cfg.device}.")
    except omegaconf.errors.MissingMandatoryValue:
        pass

    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
