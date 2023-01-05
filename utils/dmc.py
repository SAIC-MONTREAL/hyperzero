import copy

import gym
from collections import deque, OrderedDict
from typing import Any, NamedTuple

import dm_env
import numpy as np

from contextual_control_suite import suite

from dm_control import manipulation
from dm_control.suite.wrappers import action_scale, pixels
from dm_control.rl.control import FLAT_OBSERVATION_KEY
from dm_env import StepType, specs, TimeStep


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationSpecWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        wrapped_observation_spec = env.observation_spec()[FLAT_OBSERVATION_KEY]
        self._observation_spec = specs.Array(wrapped_observation_spec.shape,
                                             wrapped_observation_spec.dtype,
                                             'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation[FLAT_OBSERVATION_KEY]
        return time_step._replace(observation=obs)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class GymWrapper(dm_env.Environment):
    """Only works with Gym envs with continuous actions and states,
    also works with the fork of gym-miniworld with continuous actions
    https://github.com/sahandrez/gym-miniworld/tree/continuous_actions"""
    def __init__(self, env):
        self._env = env
        assert isinstance(env, gym.core.Env)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Box)

        if len(env.observation_space.shape) == 3:
            # Pixel observations
            self.pixel_obs = True
            pixel_spec = specs.Array(shape=env.observation_space.shape,
                                     dtype=np.uint8,
                                     name='pixels')
            self._obs_spec = OrderedDict(pixels=pixel_spec)
        else:
            # State observations
            self.pixel_obs = False
            state_spec = specs.Array(shape=env.observation_space.shape,
                                     dtype=np.float64,
                                     name='observations')
            self._obs_spec = OrderedDict(observations=state_spec)

        self._action_spec = specs.BoundedArray(shape=env.action_space.shape,
                                               dtype=np.float64,
                                               minimum=env.action_space.low,
                                               maximum=env.action_space.high)

    def step(self, action):
        first = False
        obs, reward, done, _ = self._env.step(self._convert_action(action))
        return self._to_time_step(obs, reward, done, first)

    def reset(self):
        obs, reward, done, first = self._env.reset(), 0.0, False, True
        return self._to_time_step(obs, reward, done, first)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def _convert_action(self, action):
        return np.array(action, dtype=np.float64)

    def _convert_obs(self, obs):
        if self.pixel_obs:
            return OrderedDict(pixels=np.array(obs, dtype=np.uint8))
        return OrderedDict(observations=np.array(obs, dtype=np.float64))

    def _to_time_step(self, obs, reward, done, first):
        if first:
            step_type = StepType(0)
        elif done:
            step_type = StepType(2)
        else:
            step_type = StepType(1)

        discount = 1.0 - float(done)

        time_step = TimeStep(observation=self._convert_obs(obs),
                             reward=reward,
                             step_type=step_type,
                             discount=discount)
        return time_step

    def __getattr__(self, item):
        return getattr(self._env, item)


def make_dmc(name, frame_stack, action_repeat, reward_kwargs, dynamics_kwargs, seed, pixel_obs):
    environment_kwargs = dict()
    if not pixel_obs:
        environment_kwargs['flat_observation'] = True

    task_kwargs = {'random': seed,
                   'reward_kwargs': reward_kwargs}
    if not dynamics_kwargs['use_default']:
        task_kwargs['dynamics_kwargs'] = dynamics_kwargs

    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=task_kwargs,
                         environment_kwargs=environment_kwargs,
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'

    env = ActionDTypeWrapper(env, np.float32)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    # add wrappers for pixel or state obs
    if pixel_obs:
        env = ActionRepeatWrapper(env, action_repeat)
        # add renderings for classical tasks
        if (domain, task) in suite.ALL_TASKS:
            # zoom in camera for quadruped
            camera_id = dict(quadruped=2).get(domain, 0)
            render_kwargs = dict(height=84, width=84, camera_id=camera_id)
            env = pixels.Wrapper(env,
                                 pixels_only=True,
                                 render_kwargs=render_kwargs)
        # stack several frames
        env = FrameStackWrapper(env, frame_stack, pixels_key)
        env = ExtendedTimeStepWrapper(env)
    else:
        env = ObservationSpecWrapper(env)
        env = ExtendedTimeStepWrapper(env)

    return env


def make(name, frame_stack, action_repeat, reward_kwargs, dynamics_kwargs, seed, pixel_obs):
    return make_dmc(name, frame_stack, action_repeat, reward_kwargs, dynamics_kwargs, seed, pixel_obs)
