from scripts.meta_pepg import *

CONFIG = {
    'top_log_dir': 'log_meta_pepg_pic',
    'exp_name': 'meta_cp',
    'seed': 10,
    'mdp': {
        'envid': 'Meta',
        'env': { 'REFERENCE': 'toy.yaml:MD_MAML:environment' },
        'horizon': { 'REFERENCE': 'toy.yaml:MD_MAML:trainer:unroller:k'}, # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes': 2,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': { 'REFERENCE': 'toy.yaml:MD_MAML:trainer:steps'},
            'pretrain': False, # True,
            'final_eval': True,
            'eval_freq': 10,
            'save_freq': 10,
        },
        'rollout_kwargs': {
            'min_n_samples': None,
            'max_n_rollouts': { 'REFERENCE': 'toy.yaml:MD_MAML:trainer:batch_size' },
        },

        'test_rollout': {'REFERENCE': 'toy.yaml:MD_MAML:test'}
    },
    'algorithm': {
        'optimizer':'adam',
        'lr': 5e-2,
        'lr_par': 2e-3,
        'c': 1e-1,
        'max_kl':0.1,
        'delta':None,
        'lambd':0.99,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':1,
    },
    'agent': { 'REFERENCE': 'toy.yaml:DPMD_MAML:agent' },
    'value_units': (128,128),
    'init_lstd': -2,
}


CONFIG = resovle_references(CONFIG)

if __name__ == '__main__':
    main(CONFIG)
