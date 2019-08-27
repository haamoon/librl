from rl.core.function_approximators.tf2_function_approximators import tfFuncApp
from rl.core.function_approximators.policies.tf2_policies import (tfPolicy,
                                                                  tfGaussianPolicy)

from meta_rl.tools.utils.tf_utils import as_vars, extract_vars, extract_values
from meta_rl.agents import md
import tensorflow as tf

import warnings

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


        self._agent_gen = agent_gen
        self._hyper_parameters = hyper_parameters
        self._set_variables()
        self._state_decoder = state_decoder
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def _set_variables(self):
        agent = self._agent_gen(*extract_values(self._hyper_parameters))
        self._agent_parameters = agent.parameters

        hyper_variables = as_vars(extract_vars(self._hyper_parameters))

        if isinstance(agent, md.MDAgent) or isinstance(agent, md.DPMDAgent):
            self._tf_variables = hyper_variables + self._agent_parameters
            self._tf_supervised_variables = []
        elif isinstance(agent, md.PMDAgent) or isinstance(agent, mp.DPMDAgent):
            self._tf_variables = hyper_variables
            self._tf_supervised_variables = self._agent_parameters
        else:
            raise Exception('Invalid Agent Type.')

    @property
    def ts_variables(self):
        ## parameters of the agent to be optimized by pepg, pg, or truncated backprop
        ## Do not try to take the gradient with respect to these variables. They might be isolated
        ## variables in the graph
        return self._tf_variables

    @property
    def tf_supervised_variables(self):
        ## self._tf_supervised_variables might be isolated variables in the graph.
        return self._tf_supervised_variables

    def ts_predict(self, ts_xs, **kwargs):
        try:
            assert(ts_xs.shape[0] == 1)
        except:
            ## TODO: Hopefully we only get here when log p is going to be estimated
            ## and we do not care about the returned value.
            warnings.warn('I\'m returning garbage. Make sure it is not important.')
            return self._agent.variable[tf.newaxis]

        if self._first_iteration:
            self._first_iteration = False
            return self._agent.variable[tf.newaxis]

        ts_xs = ts_xs[0]
        variable, g, context = self._state_decoder(ts_xs)
        self._agent.update(g)
        return self._agent.variable[tf.newaxis]

    def reset(self):
        self._first_iteration = True
        self._agent = self._agent_gen(*extract_values(self._hyper_parameters))
        if self._agent_parameters is not None:
            self._agent.parameters = self._agent_parameters

class LearnerPolicy(tfPolicy, LearnerFuncApp):
    def __init__(self, x_shape, y_shape, name='learner_policy', **kwargs):
        super().__init__(x_shape, y_shape, name, **kwargs)

class LearnerGaussianPolicy(tfGaussianPolicy, LearnerPolicy):
    def __init__(self, x_shape, y_shape, name='learner_gaussian_policy', **kwargs):
        """ The user needs to provide init_lstd and optionally min_std. """
        super().__init__(x_shape, y_shape, name=name, **kwargs)
