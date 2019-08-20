import argparse
import tensorflow as tf

try: # Restrict TensorFlow to use limited memory
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
except:
    pass


import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import PolicyGradient
from rl.core.function_approximators.policies.tf2_policies import tfGaussian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP
from rl.experimenter.mdps import MDP

from rl.meta_rl.env_wrapper import MetaEnv
from rl.meta_rl.agent_wrapper import LearnerGaussianPolicy
from rl.meta_rl.pepg_wrapper import PEPGWrapper

from meta_rl.builders import (environment_builder, agent_builder)
from easydict import EasyDict
from functools import partial

def setup_mdp(c, seed):
    """ Set seed and then create an MDP. """
    c = EasyDict(c)

    assert(c.envid == 'Meta')

    # fix randomness
    if tf.__version__[0]=='2':
        tf.random.set_seed(seed)
    else:
        tf.set_random_seed(seed)  # graph-level seed
    np.random.seed(seed)

    train_envs, test_envs = environment_builder.build(c.env)

    del c['envid']
    del c['env']

    tr_env = MetaEnv(train_envs)
    train_mdp = MDP(tr_env, **c)
    test_mdp = MDP(MetaEnv(test_envs), **c)

    return train_mdp, test_mdp, tr_env.observation_decoder_fn

def setup_policy(agent_config, ob_shape, ac_shape, init_lstd,
                 batch_size, dim, state_decoder):

    agent_config = EasyDict(agent_config)
    agent_gen, hyper_parameters = agent_builder.build(agent_config, batch_size, dim)

    policy = LearnerGaussianPolicy(ob_shape, ac_shape, init_lstd=init_lstd,
                                   agent_gen=agent_gen, hyper_parameters=hyper_parameters,
                                   state_decoder=state_decoder)
    return policy

def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create mdp and fix randomness
    mdp, test_mdp, state_decoder = setup_mdp(c['mdp'], c['seed'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    if mdp.use_time_info:
        ob_shape = (np.prod(ob_shape)+1,)

    state_decoder = partial(state_decoder, has_time_info=mdp.use_time_info)

    # Define the learner
    policy = setup_policy(c['agent'], ob_shape, ac_shape,
                          init_lstd=c['init_lstd'],
                          batch_size=1,
                          dim=c['mdp']['env']['dim'],
                          state_decoder=state_decoder)

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function',
                              units=c['value_units'])

    distribution = tfGaussian((0,), policy.variable.shape, init_lstd=c['init_lstd'])
    distribution.mean_variable = policy.variable

    # Create algorithm
    alg = PEPGWrapper(distribution, policy, vfn,
                      horizon=mdp.horizon, gamma=mdp.gamma,
                      **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['rollout_kwargs'], mdp_test=test_mdp)
    exp.run(**c['experimenter']['run_kwargs'])

CONFIG = {
    'top_log_dir': 'log_meta_pg',
    'exp_name': 'meta_cp',
    'seed': 9,
    'mdp': {
        'envid': 'Meta',
        'env': {
            'type': 'cosine',
            'dim': 2,
            'ntrain': 100,
            'ntest': 20,
            'access': {
                'context': True,
                'fn': True,
                'derivative': True,
            },
        },
        'horizon': 200,  # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes':8,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'save_freq': 12,
        },
        'rollout_kwargs': {
            'min_n_samples': None,
            'max_n_rollouts': 20,
        },
    },
    'algorithm': {
        'optimizer':'adam',
        'lr':0.00,
        'max_kl':0.1,
        'delta':None,
        'lambd':0.99,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':1,
    },
    'agent': {
        'type': 'MD',
        'learn_init_variables': True,
        'scheduler': {
            'type': 'basic',
            'trainable': True,
            'eta': 0.1,
            'k': 0.5,
            'c': 0.001,
        },
        'balg': {
            'trainable': False,
            'beta1': 0.9,
            'beta2': 0.99,
            'eps': 1e-8,
        },
    },
    'value_units': (128,128),
    'init_lstd': -2,
}


if __name__ == '__main__':
    main(CONFIG)
