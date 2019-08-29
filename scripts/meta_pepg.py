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
from rl.meta_rl.agent_wrapper import LearnerPolicy
from rl.meta_rl.pepg_wrapper import PEPGWrapper
from meta_rl.experiments.experiment_utils import read_config
from meta_rl.builders import (environment_builder, agent_builder)

from easydict import EasyDict
from functools import partial
import os.path as osp
import copy
import pickle

def setup_mdp(c, seed):
    """ Set seed and then create an MDP. """

    c = EasyDict(c)
    assert(c.envid == 'Meta')

    # fix randomness
    tf.keras.backend.clear_session()
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

    agent_gen, hyper_parameters = agent_builder.build(agent_config, batch_size, dim)

    policy = LearnerPolicy(ob_shape, ac_shape, agent_gen=agent_gen,
                           hyper_parameters=hyper_parameters, state_decoder=state_decoder)
    return policy

def read_ref(ref):
    config_root = '../meta_rl/meta_rl/experiments/config'
    splits = ref.split(':')

    config_fn = osp.join(config_root, ':'.join(splits[:2]))
    default_config_fn = osp.join(config_root, 'default.yaml')
    config = read_config(config_fn, default_config_fn)

    for split in splits[2:]:
        config = config[split]

    return config

def resovle_references(c):
    def _fn(item):
        k, v = item
        if isinstance(v, dict):
            return [(k, resovle_references(v))]
        elif k != 'REFERENCE' or not isinstance(v, str):
            return [item]
        else: # k=='REFERENCE' and ininstance(v, str)
            ret = read_ref(v)
            if isinstance(ret, dict):
                return ret.items()
            else:
                return [(None, ret)]

    ret_items = [it for item in c.items() for it in _fn(item)]
    if len(ret_items) == 1 and ret_items[0][0] is None:
        return ret_items[0][1]
    return dict(ret_items)

def main(c):
    c = EasyDict(c)

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

    # Save the initial state
    r = osp.join(c.top_log_dir, c.exp_name, str(c.seed), 'initial_state')
    policy.save(r, name='policy')
    with open(osp.join(r, 'train_mdp'), 'wb') as f:
        pickle.dump(mdp, f)
    with open(osp.join(r, 'test_mdp'), 'wb') as f:
        pickle.dump(test_mdp, f)


    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function',
                              units=c['value_units'])

    distribution = tfGaussian((0,), policy.variable.shape,
                              init_lstd=c['init_lstd'])
    distribution.mean_variable = policy.variable

    # Create algorithm
    alg = PEPGWrapper(distribution, policy, vfn,
                      horizon=mdp.horizon, gamma=mdp.gamma,
                      **c['algorithm'])

    # Let's do some experiments!
    train_mdp = copy.deepcopy(mdp)

    rollout_kwargs = c['experimenter']['rollout_kwargs']
    rollout_kwargs_test0 = EasyDict(rollout_kwargs)
    rollout_kwargs_test0.max_n_rollouts = c.experimenter.test_rollout.steps

    rollout_kwargs_test1 = EasyDict(rollout_kwargs)
    rollout_kwargs_test1.max_n_rollouts = c.experimenter.test_rollout.steps


    exp = Exp.Experimenter(alg, mdp, rollout_kwargs,
                           mdp_test=[test_mdp, train_mdp],
                           ro_kwargs_test=[rollout_kwargs_test0, rollout_kwargs_test1])

    exp.run(**c['experimenter']['run_kwargs'])

CONFIG = {
    'top_log_dir': 'log_meta_pepg',
    'exp_name': 'meta_cp',
    'seed': 10,
    'mdp': {
        'envid': 'Meta',
        'env': { 'REFERENCE': 'toy.yaml:MD:environment' },
        'horizon': { 'REFERENCE': 'toy.yaml:MD:trainer:unroller:k'}, # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes': 8,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': False, # True,
            'final_eval': False,
            'eval_freq': 1,
            'save_freq': 12,
        },
        'rollout_kwargs': {
            'min_n_samples': None,
            'max_n_rollouts': 20,
        },

        'test_rollout': {'REFERENCE': 'toy.yaml:MD:test'}
    },
    'algorithm': {
        'optimizer':'adam',
        'lr': 1e-1,
        'c': 1e-1,
        'max_kl':0.1,
        'delta':None,
        'lambd':0.99,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':1,
    },
    'agent': { 'REFERENCE': 'toy.yaml:MD:agent' },
    'value_units': (128,128),
    'init_lstd': -2,
}


CONFIG = resovle_references(CONFIG)

if __name__ == '__main__':
    main(CONFIG)
