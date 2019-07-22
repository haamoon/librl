import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import AggreVaTeD
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP

def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create mdp and fix randomness
    mdp = ps.general_setup(c['general'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    if mdp.use_time_info:
        ob_shape = (np.prod(ob_shape)+1,)

    # define expert
    expert = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=-1,
                                   units=(64,))
    expert.restore('./experts', name='mlp_policy_64_at_50_iter_seed_9')
    expert.name = 'expert'

    # define the learner
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=-1,
                                   units=(64, 64))

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='expert value function',
                                   units=(128,128))

    # Create algorithm
    alg = AggreVaTeD(policy, expert, vfn,
                     horizon=mdp.horizon, gamma=mdp.gamma,
                     **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['rollout_kwargs'])
    exp.run(**c['experimenter']['run_kwargs'])


CONFIGS = {
    'general': {
        'top_log_dir': 'log',
        'envid': 'DartCartPole-v1',
        'seed': 0,
        'exp_name': 'cp',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 50,
            'pretrain': True,
            'final_eval': False,
            'eval_freq': 1,
        },
        'rollout_kwargs': {
            'min_n_samples': 2000,
            'max_n_rollouts': None,
        },
    },
    'algorithm': {
        'lr':1e-3,
        'delta':None,
        'lambd':0.5,
        'max_n_batches':1000,
        'n_pretrain_interactions':4,
    },
}

if __name__ == '__main__':
    main(CONFIGS)