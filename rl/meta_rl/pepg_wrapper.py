from rl.algorithms.pepg import ParameterExploringPolicyGradient, PEPolicyAgent
from rl.algorithms.algorithm import PolicyAgent
from meta_rl.tools.utils.tf_utils import average
import tensorflow as tf
from rl.core.utils import logz

class PEPGWrapper(ParameterExploringPolicyGradient):
    def __init__(self, distribution,
                 policy, vfn,
                 optimizer_par='adam',
                 lr_par=1e-3,
                 optimizer='adam',
                 lr=1e-3, c=1e-3, max_kl=0.1,
                 horizon=None, gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2,
                 n_warm_up_itrs=None,
                 n_pretrain_itrs=1):

        assert(optimizer_par == 'adam')
        self._optimizer_par = tf.optimizers.Adam(lr_par)
        super().__init__(distribution,
                         policy, vfn,
                         optimizer=optimizer,
                         lr=lr, c=c, max_kl=max_kl,
                         horizon=horizon, gamma=gamma, delta=delta, lambd=lambd,
                         max_n_batches=max_n_batches,
                         n_warm_up_itrs=n_warm_up_itrs,
                         n_pretrain_itrs=n_pretrain_itrs)

    def agent(self, mode):
        if mode=='behavior':
            return PEPolicyAgentWrapper(self.policy, self.distribution)
        elif mode=='target':
            return PolicyAgentWrapper(self.get_policy())


    def update(self, ros, agents):
        super().update(ros, agents)
        logz.log_tabular('RawStepSize', self.learner._base_alg._scheduler.stepsize) # XXX

        if not self.policy.tf_supervised_variables:
            return

        # update other parameters by surrogate loss
        def loss_grad(agent):
            with tf.GradientTape() as tape:
                tape.watch(agent.parameters)
                loss = agent.loss
            return tape.gradient(loss, agent.parameters,
                                 unconnected_gradients='zero')

        online_agents = [r.online_agent  for ro in ros for r in ro]
        loss_grads = average(*list(map(loss_grad, online_agents)))
        # print(average(*[a.loss for a in agents]))
        self._optimizer_par.apply_gradients(zip(loss_grads, self.policy.tf_supervised_variables))




class PEPolicyAgentWrapper(PEPolicyAgent):
    def pi(self, ob, t, done):
        if t==0:
            self.policy.variable = self._sample()
            self.policy.reset()
        return self.policy(ob)

    def callback(self, rollout):
        super().callback(rollout)
        rollout.online_agent = self.policy._agent

class PolicyAgentWrapper(PolicyAgent):
    def pi(self, ob, t, done):
        if t==0:
            self.policy.reset()
        return self.policy(ob)

