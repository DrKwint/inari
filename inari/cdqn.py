import collections
import functools
import logging
from typing import Any, Tuple

import dopamine.jax.losses as losses
import dopamine.replay_memory.prioritized_replay_buffer as prioritized_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
import gin
import jax
import numpy as np
import optax
import tensorflow as tf
from jax import lax
from jax import numpy as jnp
from flax import linen as nn


@gin.configurable
def create_optimizer(name='adam',
                     learning_rate=6.25e-5,
                     beta1=0.9,
                     beta2=0.999,
                     eps=1.5e-4,
                     centered=False):
    """Create an optimizer for training.
    Currently, only the Adam and RMSProp optimizers are supported.
    Args:
        name: str, name of the optimizer to create.
        learning_rate: float, learning rate to use in the optimizer.
        beta1: float, beta1 parameter for the optimizer.
        beta2: float, beta2 parameter for the optimizer.
        eps: float, epsilon parameter for the optimizer.
        centered: bool, centered parameter for RMSProp.
    Returns:
        An optax optimizer.
    """
    if name == 'adam':
        logging.info(
            'Creating Adam optimizer with settings lr=%f, beta1=%f, '
            'beta2=%f, eps=%f', learning_rate, beta1, beta2, eps)
        return optax.adam(learning_rate, b1=beta1, b2=beta2, eps=eps)
    elif name == 'rmsprop':
        logging.info(
            'Creating RMSProp optimizer with settings lr=%f, beta2=%f, '
            'eps=%f', learning_rate, beta2, eps)
        return optax.rmsprop(learning_rate,
                             decay=beta2,
                             eps=eps,
                             centered=centered)
    else:
        raise ValueError('Unsupported optimizer {}'.format(name))


@gin.configurable
@functools.partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
        Begin at 1. until warmup_steps steps have been taken; then
        Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
        Use epsilon from there on.
    Args:
        decay_period: float, the period over which epsilon is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before epsilon is decayed.
        epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
        A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = jnp.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


@functools.partial(jax.jit, static_argnums=(0, 4))
def select_max_q_action(network_def,
                        params,
                        state: jnp.ndarray,
                        action_init: jnp.ndarray,
                        search_steps: int,
                        cost_coeff: float = 0.):
    #neg_q_fn = lambda a: -1 * network_def.apply(params, state, a)[0]
    def loss_fn(a):
        q_r, q_c = network_def.apply(params, state, a, cost_coeff)
        return (-q_r + cost_coeff * q_c) * (1 / (1 + cost_coeff))

    @jax.jit
    def step(carry: Tuple[jnp.ndarray, jnp.ndarray],
             step_input: Any) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Any]:
        action, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(action)
        updates, opt_state = optimizer.update(grads, opt_state, action)
        action = optax.apply_updates(action, updates)
        action = jnp.clip(action, -5, 5)
        carry = action, opt_state
        return carry, loss

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(action_init)
    carry, step_losses = lax.scan(step, (action_init, opt_state),
                                  None,
                                  length=search_steps,
                                  unroll=search_steps)
    action, _ = carry
    q = network_def.apply(params, state, action, cost_coeff)
    return action, q


@functools.partial(jax.jit, static_argnums=(0, 8))
def target_q(network_def, online_params, target_params, next_states, rewards,
             terminals, actions, cumulative_gamma, search_steps, costs,
             cost_coeff):
    """Compute the target Q-value."""

    def q_target(state, action):
        return network_def.apply(target_params, state, action, cost_coeff)

    select_q = functools.partial(select_max_q_action,
                                 network_def=network_def,
                                 params=online_params,
                                 search_steps=search_steps,
                                 cost_coeff=cost_coeff)
    # Do the DDQN thing of selecting the next q with online params
    # and evaluate its q-value with target params
    next_actions = jax.vmap(select_q)(state=next_states,
                                      action_init=actions)[0]
    q_vals = jax.vmap(q_target)(state=next_states, action=next_actions)
    rew_replay_next_qt_max = q_vals[:, 0]
    cst_replay_next_qt_max = q_vals[:, 1]
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return jax.lax.stop_gradient(
        rewards + cumulative_gamma * rew_replay_next_qt_max *
        (1. - terminals)), jax.lax.stop_gradient(costs + cumulative_gamma *
                                                 cst_replay_next_qt_max *
                                                 (1. - terminals))


@functools.partial(jax.jit, static_argnums=(0, 3, 11, 12, 13))
def train(network_def,
          online_params,
          target_params,
          optimizer,
          optimizer_state,
          rng,
          states,
          actions,
          next_states,
          rewards,
          terminals,
          cumulative_gamma,
          search_steps,
          loss_type='huber',
          costs=None,
          c_loss_coeff=0.25,
          cost_penalty=0.):
    """Run the training step."""

    def loss_fn(params, r_target, c_target):

        def q_online(state, action):
            return network_def.apply(params, state, action, cost_penalty)

        replay_chosen_q = jax.vmap(q_online)(states, actions)
        r_replay_chosen_q = replay_chosen_q[:, 0]
        c_replay_chosen_q = replay_chosen_q[:, 1]
        if loss_type == 'huber':
            return jnp.mean(
                jax.vmap(losses.huber_loss)(
                    r_target, r_replay_chosen_q)), c_loss_coeff * jnp.mean(
                        jax.vmap(losses.huber_loss)(c_target,
                                                    c_replay_chosen_q))
        return jnp.mean(
            jax.vmap(losses.mse_loss)(
                r_target, r_replay_chosen_q)), c_loss_coeff * jnp.mean(
                    jax.vmap(losses.mse_loss)(c_target, c_replay_chosen_q))

    r_target, c_target = target_q(network_def, online_params, target_params,
                                  jnp.squeeze(next_states), rewards, terminals,
                                  actions, cumulative_gamma, search_steps,
                                  costs, cost_penalty)
    grad_fn = jax.value_and_grad(lambda params, r_target, c_target: jnp.sum(
        jnp.array(loss_fn(params, r_target, c_target))))
    loss, grad = grad_fn(online_params, r_target, c_target)
    r_loss, c_loss = loss_fn(online_params, r_target, c_target)
    updates, optimizer_state = optimizer.update(grad,
                                                optimizer_state,
                                                params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    return optimizer_state, online_params, loss, r_loss, c_loss / c_loss_coeff


@gin.configurable
class CDQN():

    def __init__(self,
                 network,
                 observation_shape,
                 action_shape,
                 rng,
                 num_search_steps,
                 epsilon_fn=linearly_decaying_epsilon,
                 epsilon_train=0.05,
                 epsilon_eval=0.001,
                 epsilon_decay_period=1000000,
                 optimizer='adam',
                 loss_type='huber',
                 gamma=0.99,
                 update_horizon=1,
                 update_period=2,
                 min_replay_history=5000,
                 target_update_period=5000,
                 batch_size=64,
                 replay_capacity=1000000,
                 cost_penalty_init=0.,
                 base_dir=None,
                 checkpoint_dir=None,
                 summary_writing_frequency=500,
                 observation_dtype=np.float32,
                 action_dtype=np.float32):
        self.network_def = network
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.update_horizon = update_horizon
        self.gamma = gamma
        self.cumulative_gamma = gamma**update_horizon
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.num_search_steps = num_search_steps
        self.epsilon_fn = epsilon_fn
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.cost_penalty = cost_penalty_init
        self._optimizer_name = optimizer
        self._loss_type = loss_type
        self._replay = self._build_replay_buffer(batch_size, replay_capacity)
        self.summary_writer = tf.summary.create_file_writer('.')
        self.summary_writing_frequency = summary_writing_frequency

        self.lambd = 0.
        self.pid_prev_cost = 0.
        self.pid_integral = 0.

        self.training_steps = 0
        self.eval_mode = False
        self.state = np.zeros(observation_shape)
        self.action = np.zeros(action_shape)
        self._rng = rng
        self._build_networks_and_optimizer()
        self._pre_train()

    def pid_update(self, cost: float, p_param: float, i_param: float,
                   d_param: float, cost_limit: float):
        delta = cost - cost_limit
        print("delta:", delta)
        partial = nn.relu(cost - self.pid_prev_cost)
        print("partial:", partial)
        self.pid_integral = nn.relu(self.pid_integral + delta)
        print("integral:", self.pid_integral)
        self.cost_penalty = nn.relu(p_param * delta +
                                    i_param * self.pid_integral +
                                    d_param * partial)
        print("cost_penalty:", self.cost_penalty)
        self.pid_prev_cost = cost
        return self.cost_penalty

    def _build_networks_and_optimizer(self):
        self._rng, rng = jax.random.split(self._rng)
        self.online_params = self.network_def.init(rngs={'params': rng},
                                                   state=self.state,
                                                   action=self.action,
                                                   lambd=0.)
        self.optimizer = create_optimizer(self._optimizer_name)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def select_action(self, state, last_action) -> Tuple[jnp.ndarray, float]:
        action, est_q = select_max_q_action(self.network_def,
                                            self.online_params, state,
                                            last_action, self.num_search_steps,
                                            self.cost_penalty)
        epsilon = self.epsilon_fn(self.epsilon_decay_period,
                                  self.training_steps, self.min_replay_history,
                                  self.epsilon_train)
        self._rng, coin_flip_rng, noise_rng = jax.random.split(self._rng, 3)
        if jax.random.bernoulli(coin_flip_rng, epsilon):
            action += 0.5 * jax.random.normal(noise_rng, action.shape)
        return action, est_q

    def _build_replay_buffer(self, batch_size, replay_capacity):
        """Creates the replay buffer used by the agent."""
        return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            observation_shape=self.observation_shape,
            observation_dtype=self.observation_dtype,
            action_shape=self.action_shape,
            action_dtype=self.action_dtype,
            stack_size=1,
            update_horizon=self.update_horizon,
            extra_storage_types=[ReplayElement('cost', (), float)],
            gamma=self.gamma,
            reward_shape=(),
            terminal_dtype=bool)

    def _sync_weights(self):
        """Syncs the target_network_params with online_params."""
        self.target_network_params = self.online_params

    def store_transition(self,
                         last_observation,
                         action,
                         reward,
                         is_terminal,
                         *args,
                         priority=None,
                         episode_end=False):
        """Stores a transition when in training mode.
    Stores the following tuple in the replay buffer (last_observation, action,
    reward, is_terminal, priority).
    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      *args: Any, other items to be added to the replay buffer.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
      episode_end: bool, whether this transition is the last for the episode.
        This can be different than terminal when ending the episode because
        of a timeout, for example.
    """
        is_prioritized = isinstance(
            self._replay,
            prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer)
        if is_prioritized and priority is None:
            priority = self._replay.sum_tree.max_recorded_priority

        if not self.eval_mode:
            self._replay.add(last_observation,
                             action,
                             reward,
                             is_terminal,
                             *args,
                             priority=priority,
                             episode_end=episode_end)

    def train_step(self):
        """Runs a single training step.
        Runs training if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_steps` is a multiple of `update_period`.
        Also, syncs weights from online_params to target_network_params if training
        steps is a multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sample_from_replay_buffer()

                self._rng, train_rng = jax.random.split(self._rng)
                self.optimizer_state, self.online_params, loss, r_loss, c_loss = train(
                    network_def=self.network_def,
                    online_params=self.online_params,
                    target_params=self.target_network_params,
                    optimizer=self.optimizer,
                    optimizer_state=self.optimizer_state,
                    rng=train_rng,
                    states=self.replay_elements['state'],
                    actions=self.replay_elements['action'],
                    next_states=self.replay_elements['next_state'],
                    rewards=self.replay_elements['reward'],
                    terminals=self.replay_elements['terminal'],
                    cumulative_gamma=self.cumulative_gamma,
                    search_steps=self.num_search_steps,
                    loss_type=self._loss_type,
                    costs=self.replay_elements['cost'],
                    cost_penalty=self.cost_penalty)
                if (self.summary_writer is not None and self.training_steps > 0
                        and self.training_steps %
                        self.summary_writing_frequency == 0):
                    with self.summary_writer.as_default():
                        tf.summary.scalar("train/" + self._loss_type,
                                          loss,
                                          step=self.training_steps)
                        tf.summary.scalar('train/reward_q_loss',
                                          r_loss,
                                          step=self.training_steps)
                        tf.summary.scalar("train/cost_q_loss",
                                          c_loss,
                                          step=self.training_steps)
            if self.training_steps % self.target_update_period == 0:
                self._sync_weights()

            self.training_steps += 1

    def _sample_from_replay_buffer(self):
        samples = self._replay.sample_transition_batch()
        types = self._replay.get_transition_elements()
        self.replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            self.replay_elements[element_type.name] = element

    def _pre_train(self, num_steps=100000):
        opt_state = self.optimizer.init(self.online_params)

        loss = lambda params, obs, act, lambd: jnp.mean(
            losses.mse_loss(
                -1 * jnp.sqrt(jnp.sum(jnp.square(act), axis=-1)),
                jax.vmap(lambda o, a: self.network_def.apply(
                    params, o, a, lambd)[0])(obs, act)))

        @jax.jit
        def step(params, opt_state, rng):
            state_rng, action_rng, lambd_rng = jax.random.split(rng, num=3)
            batch_obs = 10. * jax.random.normal(
                state_rng, (128, ) + self.observation_shape)
            batch_act = 10. * jax.random.normal(action_rng,
                                                (128, ) + self.action_shape)
            batch_lambd = jax.random.normal(lambd_rng, ())
            loss_value, grads = jax.value_and_grad(loss)(params, batch_obs,
                                                         batch_act,
                                                         batch_lambd)
            updates, opt_state = self.optimizer.update(grads, opt_state,
                                                       params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for _ in range(num_steps):
            self._rng, step_rng = jax.random.split(self._rng)
            self.online_params, opt_state, loss_value = step(
                self.online_params, opt_state, step_rng)
        self.target_network_params = self.online_params