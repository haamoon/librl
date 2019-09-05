from meta_rl.experiments.experiment_utils import read_config
from meta_rl.experiments.new_toy_tests import main
import os


META_RL_HOME = '../meta_rl/meta_rl/experiments'


CONFIG = read_config(os.path.join(META_RL_HOME, 'config/toy.yaml:MD_MAML'),
                     os.path.join(META_RL_HOME, 'config/default.yaml'))



if __name__ == '__main__':
    main(CONFIG)
