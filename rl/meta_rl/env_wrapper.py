from collections import namedtuple
import gym
import numpy as np
import tensorflow as tf

SpaceDescriptor = namedtuple('Space', ['shape'])

class MetaEnv(gym.Env):
    def __init__(self, env_list):
        if not isinstance(env_list, list):
            env_list = [env_list]

        self._env_list = env_list

        self.action_space = SpaceDescriptor(shape=env_list[0].action_shape)

        assert(len(self.action_space.shape) == 1)
        context = env_list[0].context
        if context is None:
            self._context_shape = (0,)
        else:
           self._context_shape = context.shape

        # This way we don't have to hard code the observation shape here
        dummy_act = np.zeros(self.action_space.shape)
        dummy_obs = self.get_observation(dummy_act, dummy_act,
                           np.zeros(self._context_shape))
        self.observation_space = SpaceDescriptor(shape=dummy_obs.shape)

        # TODO: rl.experimenter.mdps +194 needs this
        self._max_episode_steps = float('Inf')

    def step(self, np_action):
        action = tf.convert_to_tensor(np_action)
        oracle = self._env.get_oracle(action)
        loss = oracle.compute_loss().numpy()
        g = oracle.compute_grad().numpy()
        context = self._env.context
        ob = self.get_observation(action, g, context)
        return ob, -loss, False, None

    def get_observation(self, action, g, context):
        ''' inputs are numpy.array'''
        assert(action.shape == self.action_space.shape)
        assert(g.shape == self.action_space.shape)

        if context is None:
            context = np.zeors((0,))

        assert(context.shape == self._context_shape)
        return np.concatenate([action, g, context.ravel()])

    def reset(self):
        self._env = np.random.choice(self._env_list)
        dummy_act = np.zeros(self.action_space.shape)
        ob = self.get_observation(dummy_act, dummy_act, self._env.context)
        return ob

    @property
    def observation_decoder_fn(self):
        def decoder_fn(observation, has_time_info):
            ''' observation is a tf.tensor object '''
            if has_time_info:
                observation = observation[:-1]
            assert(tuple(observation.shape) == self.observation_space.shape)
            action, g, contex = tf.split(observation, [self.action_space.shape[0],
                                                       self.action_space.shape[0],
                                                       np.prod(self._context_shape)])
            contex = tf.reshape(contex, self._context_shape)
            return action, g, contex
        return decoder_fn
