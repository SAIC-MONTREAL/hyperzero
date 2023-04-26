# Hypernetworks for Zero-shot Transfer in Reinforcement Learning
Author's PyTorch implementation of HyperZero. If you use our code, please cite our [AAAI 2023 paper](https://arxiv.org/abs/2211.15457):

```bib
@article{rezaei2022hypernetworks,
  title={Hypernetworks for Zero-shot Transfer in Reinforcement Learning},
  author={Rezaei-Shoshtari, Sahand and Morissette, Charlotte and Hogan, Francois Robert and Dudek, Gregory and Meger, David},
  journal={arXiv preprint arXiv:2211.15457},
  year={2022}
}
```

## Setup
* We recommend using a conda virtual environment to run the code.
Create the virtual environment:
```commandline
conda create -n contextual_env python=3.9
conda activate hyperzero_env
pip install --upgrade pip
```
* This package requires [Contextual Control Suite](https://github.com/SAIC-MONTREAL/contextual-control-suite) 
to run. First, install that package following its instructions.
* Clone this package and install the rest of its requirements:
```commandline
pip install -r requirements.txt
```

## Instructions
The training of HyperZero is done in two steps:
1. Obtaining the near-optimal rollout dataset 
   1. First option is to train the RL agents yourself and then collect the rollouts. 
   2. Second option is to use our published dataset.
2. Training HyperZero on the dataset.

### Step 1: Obtaining the Near-optimal Rollout Dataset 

#### Option A: Training RL Agents
* To train standard RL on a [Contextual Control Suite](https://github.com/SAIC-MONTREAL/contextual-control-suite) environment with default reward and dynamics parameters, use:
```commandline
python train.py agent@_global_=td3 task@_global_=cheetah_run reward@_global_=cheetah_default dynamics@_global_=default
```
* We use [Hydra](https://github.com/facebookresearch/hydra) to specify configs. To sweep over context parameters,
use the `--multirun` argument. For example, the following command sweeps over reward margins of ` range(0.5,10.1,0.5)` 
with linear reward function and default dynamics parameters:
```commandline
python train.py --multirun agent@_global_=td3 task@_global_=cheetah_run reward@_global_=overwrite_all reward_parameters.ALL.margin='range(0.5,10.1,.5)' reward_parameters.ALL.sigmoid='linear'  dynamics@_global_=default
```
* As another example, the following commands sweeps over a grid of reward margins of `range(1,5.1,1)` and dynamics 
parameters of `range(0.3,0.71,0.05)`:
```commandline
python train.py --multirun agent@_global_=td3 task@_global_=cheetah_run reward@_global_=overwrite_all reward_parameters.ALL.margin='range(1,5.1,1)' reward_parameters.ALL.sigmoid='linear' dynamics@_global_=cheetah dynamics_parameters.length='range(0.3,0.71,0.05)'
```
* **Note:** Be mindful! These commands launch a lot of training scripts! 

* To evaluate the RL agents and generate the dataset used for training hyperzero, you can use [eval.py](eval.py). 
A helper script is set up to load each trained RL agent and generates a set of `.npy` files to be later loaded 
by [RLSolutionDataset](utils/dataset.py):
```commandline
python eval_many_agents.py --rootdir <path-to-the-root-dir-of-RL-runs> --domain_task cheetah_run
```

#### Option B: Using the Published Dataset
* Instead of training RL agents, you can download our published dataset from 
[here](https://mcgill-my.sharepoint.com/:f:/g/personal/sahand_rezaei-shoshtari_mail_mcgill_ca/EhDgTXh3v-pIhTHZXM1xaz0BMWT-N8jNheVm2156mhbZdA?e=hMu4N1). 
This dataset was used to generate some of the results in our [AAAI 2023 paper](https://arxiv.org/abs/2211.15457).
* Simply extract the dataset in the desired location and proceed to Step 2.

### Step 2: Training HyperZero
* Finally, to train hyperzero (or the baselines), use the following command. It trains and saves the RL regressor:
```commandline
python train_rl_regressor.py rollout_dir=<path-to-npy-logs> domain_task=cheetah_run approximator@_global_=hyperzero input_to_model=rew
```
* The argument `input_to_model` specifies the MDP context that is used to generate the policies. It can take `rew`, 
`dyn` and `rew_dyn`.
* Or to train a bunch of RL regressors, use:
```commandline
 python train_rl_regressor.py --multirun rollout_dir=<path-to-npy-logs> domain_task=cheetah_run input_to_model=`rew`
```
* To visualize the RL regressor and roll-out the policy, you can use [eval.py](eval.py). A helper script is set up that
loads several RL regressors and evaluates them:
```commandline
python eval_many_approximators.py --rootdir <path-to-the-root-dir-of-RL-runs> --approximator_rootdir <path-to-the-trained-approximators> --rollout_dir <path-to-npy-logs> --domain_task cheetah_run
```
