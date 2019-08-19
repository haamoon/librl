from rl.core.function_approximators.tf2_function_approximators import tfFuncApp
from rl.core.function_approximators.policies.tf2_policies import (tfPolicy,
                                                                  tfGaussianPolicy)

from meta_rl.tools.utils.tf_utils import as_vars
import tensorflow as tf

class LearnerFuncApp(tfFuncApp):
    '''
        We have a slightly different termonology in meta rl code. Variable is what
        is being optimized inside an agent. Parameters and hyper parameters refer
        to parameters being optmized by the meta rl algorithm.
    '''

    def __init__(self, x_shape, y_shape, name, agent_gen=None, hyper_parameters=None,
                 state_decoder=None, **kwargs):
        assert(agent_gen is not None)
        assert(hyper_parameters is not None)
        assert(state_decoder is not None)

        self._agent_parameters = agent_gen(*hyper_parameters).parameters

        self._agent_gen = agent_gen
        self._hyper_parameters = hyper_parameters
        self._state_decoder = state_decoder
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    @property
    def ts_variables(self):
        # TODO: why do we need this? to compute gradients? Does it work properly
        # with respect to self._hyper_parameters? Can we use self._agent_parameters
        # instead?

        hyper_variables = as_vars(self._hyper_parameters)
        # parameters of the agent to be optimized
        return self._agent_parameters + hyper_variables

    def ts_predict(self, ts_xs, **kwargs):
        assert(ts_xs.shape[0] == 1)
        # or this one?
        #assert(tf.shape(ts_xs)[0].numpy() == 1)

        ts_xs = ts_xs[0]
        variable, g, context = self._state_decoder(ts_xs)
        self._agent.update(g)
        return self._agent.variable[tf.newaxis]

    def reset(self):
        self._agent = self._agent_gen(*self._hyper_parameters)
        if self._agent_parameters is not None:
            self._agent.parameters = self._agent_parameters

class LearnerPolicy(tfPolicy, LearnerFuncApp):
    def __init__(self, x_shape, y_shape, name, **kwargs):
        super().__init__(x_shape, y_shape, name, **kwargs)

class LearnerGaussianPolicy(tfGaussianPolicy, LearnerPolicy):
    def __init__(self, x_shape, y_shape, name='learner_gaussian_policy', **kwargs):
        """ The user needs to provide init_lstd and optionally min_std. """
        super().__init__(x_shape, y_shape, name=name, **kwargs)
