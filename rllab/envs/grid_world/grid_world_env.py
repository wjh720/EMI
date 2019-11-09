import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging

import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import *
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger

from rllab.envs.grid_world.pass_environment import Pass
from rllab.envs.grid_world.island_environment import Island
from rllab.envs.grid_world.pushball_environment import PushBall
from rllab.envs.grid_world.x_island_environment import x_Island

from rllab.envs.atari.atari_wrappers import wrap_deepmind, make_atari, get_wrapper_of_specific_type, FrameSaver


def get_venv(args):
	env_type = args.env
	if env_type == 'pass':
		env = make_multi_pass_env(env_id, env_type, num_env, seed, args)
	elif env_type == 'island':
		env = make_m_island_env(env_id, env_type, num_env, seed, args)
	elif env_type == 'x_island':
		env = make_m_x_island_env(env_id, env_type, num_env, seed, args)
	elif env_type == 'pushball':
		env = make_m_pushball_env(env_id, env_type, num_env, seed, args)
	venv = TfEnv(env)
	return venv


class Grid_World_Env(Env, Serializable):

	def __init__(self, args):

		Serializable.quick_init(self, locals())

		env = get_venv(args)
		self.env = env
		self.env_id = env.spec.id

		self._observation_space = convert_gym_space(env.observation_space)
		logger.log("observation space: {}".format(self._observation_space))
		self._action_space = convert_gym_space(env.action_space)
		logger.log("action space: {}".format(self._action_space))
		self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']

	@property
	def observation_space(self):
		return self._observation_space

	@property
	def action_space(self):
		return self._action_space

	@property
	def horizon(self):
		return self._horizon

	def reset(self):
		if self._force_reset and self.monitoring:
			from gym.wrappers.monitoring import Monitor
			assert isinstance(self.env, Monitor)
			recorder = self.env.stats_recorder
			if recorder is not None:
				recorder.done = True
		return self.env.reset()

	def step(self, action):
		next_obs, reward, done, info = self.env.step(action)
		return Step(next_obs, reward, done, **info)

	def render(self, mode='human'):
		return self.env.render(mode)

	def terminate(self):
		if self.monitoring:
			self.env._close()
			if self._log_dir is not None:
				print("""
	***************************
	Training finished! You can upload results to OpenAI Gym by running the following command:
	python scripts/submit_gym.py %s
	***************************
				""" % self._log_dir)

	def get_original_frames(self):
		if not self.save_original_frames:
			return None
		return self.original_frame_saver.get_frames()
