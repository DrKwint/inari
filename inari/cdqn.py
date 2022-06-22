import collections
import itertools
import tensorboard
import tensorflow as tf
import functools
import jax
from flax import linen as nn
import gin
from jax import numpy as jnp
import rlax
import dopamine.replay_memory.prioritized_replay_buffer as prioritized_replay_buffer
import numpy as np
import dopamine.jax.losses as losses
import optax
import logging


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


def select_max_q_action(q_network,
                        state: jnp.ndarray,
                        rng,
                        action_shape,
                        max_steps=10):
    q_fn = lambda a: q_network(state, a)

    @jax.jit
    def step(opt_state, action):
        neg_q, grads = jax.value_and_grad(q_fn)(action)
        updates, opt_state = optimizer.update(grads, opt_state, action)
        action = optax.apply_updates(action, updates)
        return action, opt_state, neg_q

    optimizer = optax.adam(learning_rate=1e-2)
    action = jax.random.normal(key=rng, shape=action_shape)
    opt_state = optimizer.init(action)
    for _ in range(max_steps):
        action, opt_state, neg_q = step(opt_state, action)
    return action, -1 * neg_q


def target_q(target_network, next_states, rewards, terminals, actions, rng,
             cumulative_gamma):
    """Compute the target Q-value."""
    calculate_q = functools.partial(select_max_q_action,
                                    target_network=target_network,
                                    action_shape=(actions.shape[-1], ),
                                    max_steps=100)
    q_vals = jax.vmap(calculate_q)(state=next_states,
                                   rng=jax.random.split(rng,
                                                        len(next_states)))[1]
    replay_next_qt_max = q_vals
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    return jax.lax.stop_gradient(rewards +
                                 cumulative_gamma * replay_next_qt_max *
                                 (1. - terminals))


@functools.partial(jax.jit, static_argnums=(0, 3, 11, 12))
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
          loss_type='huber'):
    """Run the training step."""

    def loss_fn(params, target):

        def q_online(state, action):
            return network_def.apply(params, state, action)[0]

        q_values = jax.vmap(q_online)(states, actions)
        replay_chosen_q = q_values
        if loss_type == 'huber':
            return jnp.mean(
                jax.vmap(losses.huber_loss)(target, replay_chosen_q))
        return jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

    def q_target(state, action):
        return network_def.apply(target_params, state, action)[0]

    target = target_q(q_target, jnp.squeeze(next_states), rewards, terminals,
                      actions, rng, cumulative_gamma)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(online_params, target)
    updates, optimizer_state = optimizer.update(grad,
                                                optimizer_state,
                                                params=online_params)
    online_params = optax.apply_updates(online_params, updates)
    return optimizer_state, online_params, loss


class CDQN():

    def __init__(self,
                 network,
                 observation_shape,
                 action_shape,
                 base_dir,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=30000,
                 update_period=4,
                 target_update_period=30000,
                 summary_writing_frequency=500,
                 optimizer='adam',
                 loss_type='huber',
                 observation_dtype=np.float32,
                 action_dtype=np.float32,
                 seed=0):
        self.network_def = network()
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
        self._optimizer_name = optimizer
        self._loss_type = loss_type
        self._replay = self._build_replay_buffer()
        self.summary_writer = tf.summary.create_file_writer(base_dir)
        self.summary_writing_frequency = summary_writing_frequency

        self.training_steps = 0
        self.eval_mode = False
        self.state = np.zeros(observation_shape)
        self.action = np.zeros(action_shape)
        self._rng = jax.random.PRNGKey(seed)
        self._build_networks_and_optimizer()

    def _build_networks_and_optimizer(self):
        self._rng, rng = jax.random.split(self._rng)
        self.online_params = self.network_def.init(rngs={'params': rng},
                                                   state=self.state,
                                                   action=self.action)
        self.optimizer = create_optimizer(self._optimizer_name)
        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = self.online_params

    def select_action(self, state, rng, max_steps=10):

        def q_online(action):
            return self.network_def.apply(self.online_params, state, action)[0]

        @jax.jit
        def step(opt_state, action):
            neg_q, grads = jax.value_and_grad(q_online)(action)
            updates, opt_state = optimizer.update(grads, opt_state, action)
            action = optax.apply_updates(action, updates)
            return action, opt_state, neg_q

        optimizer = optax.adam(learning_rate=1e-2)
        action = jax.random.normal(key=rng, shape=self.action_shape)
        opt_state = optimizer.init(action)
        for _ in range(max_steps):
            action, opt_state, neg_q = step(opt_state, action)
        return action, -1 * neg_q

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
            observation_shape=self.observation_shape,
            observation_dtype=self.observation_dtype,
            action_shape=self.action_shape,
            action_dtype=self.action_dtype,
            stack_size=1,
            batch_size=32,
            replay_capacity=1000000,
            update_horizon=self.update_horizon,
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
                self.optimizer_state, self.online_params, loss = train(
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
                    loss_type=self._loss_type)
                if (self.summary_writer is not None and self.training_steps > 0
                        and self.training_steps %
                        self.summary_writing_frequency == 0):
                    with self.summary_writer.as_default():
                        tf.summary.scalar(self._loss_type,
                                          loss,
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