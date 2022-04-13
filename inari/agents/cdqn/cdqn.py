import jax
import gin
import tqdm
import optax
import logging
import pathlib
import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from functools import partial
from collections import OrderedDict
import dopamine.jax.losses as losses
from typing import Any, Callable, Optional, Tuple
from dopamine.replay_memory.prioritized_replay_buffer import (
    OutOfGraphPrioritizedReplayBuffer,
)


@gin.configurable
class CDQN:
    """
    Continuous Deep Q-Network
    """

    def __init__(
        self,
        network: Callable,
        observation_shape: np.dtype.shape,
        action_shape: np.dtype.shape,
        base_dir: str,
        rng: Any,
        gamma: float = 0.99,
        update_horizon: int = 1,
        min_replay_history: int = 5000,
        update_period: int = 4,
        target_update_period: int = 5000,
        summary_writing_frequency: int = 500,
        optimizer: Any = optax.adam(
            learning_rate=6.25e-5, b1=0.9, b2=0.999, eps=1.5e-4
        ),
        loss_type: str = "huber",
        observation_dtype=np.float32,
        action_dtype=np.float32,
        stack_size: int = 1,
        pretrain_steps: int = 100000,
        epsilon_fn: Optional[Callable] = None,
        epsilon_fn_target: float = 0.01,
        epsilon_train: float = 0.25,
        epsilon_eval: float = 0.001,
        epsilon_decay_period: int = 250000,
        eval_mode: bool = False,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Args:
            network (Callable): Network to train
            observation_shape (np.dtype.shape): The shape of the observations
            action_shape (np.dtype.shape): The shape of the actions
            base_dir (str): Location for the summary writer to log to
            rng (KeyArray): Pseudo-random number generator key
            gamma (float, optional): Discount factor. Defaults to 0.99.
            update_horizon (int, optional): Length of an update. Defaults to 1.
            min_replay_history (int, optional): Minimum replay history. Defaults to
                5000.
            update_period (int, optional): The period at which an agent should be
                trained. Defaults to 4.
            target_update_period (int, optional): Frequency that the target network
                params should be synced with the online params. Defaults to 5000.
            summary_writing_frequency (int, optional): Summary writing frequency.
                Defaults to 500.
            optimizer (Any, optional): Optimizer to use for search. Defaults to
                optax.adam( learning_rate=6.25e-5, b1=0.9, b2=0.999, eps=1.5e-4 ).
            loss_type (str, optional): The loss type to use for training. Defaults to
                "huber".
            observation_dtype (optional): Datatype of the observation. Defaults to
                np.float32.
            action_dtype (optional): Datatype of the action. Defaults to np.float32.
            stack_size (int, optional): Number of frames to use in replay buffer state
                stack. Defaults to 1.
            pretrain_steps (int, optional): The number of steps to pre-train the model
                over. Defaults to 100000.
            epsilon_fn (Optional[Callable], optional): Function used to manipulate the
                epsilon value during training. Defaults to None.
            epsilon_fn_target (float, optional): Target epsilon value that the epsilon
                function is attempting to achieve. Defaults to 0.01.
            epsilon_train (float, optional): Static epsilon value to use if the epsilon
                function is not provided. Defaults to 0.25.
            epsilon_eval (float, optional): Epsilon value to use during testing.
                Defaults to 0.001.
            epsilon_decay_period (int, optional): The number of timesteps over which the
                epsilon value should decay. Defaults to 250000.
            eval_mode (bool, optional): Flag indicating whether to test the policy.
                Defaults to False.
            log_level (int, optional): The level of log messages that should be
                displayed. Defaults to logging.INFO.
        """
        self.__logger = self.__init_logger("CDQN", log_level=log_level)
        self.__logger.debug("Initializing the CDQN")

        self.__network = network()
        self.__observation_shape = observation_shape
        self.__observation_dtype = observation_dtype
        self.__action_shape = action_shape
        self.__action_dtype = action_dtype
        self.__cumulative_gamma = gamma**update_horizon
        self.__min_replay_history = min_replay_history
        self.__update_period = update_period
        self.__target_update_period = target_update_period
        self.__optimizer = optimizer
        self.__loss_type = loss_type
        self.__replay = OutOfGraphPrioritizedReplayBuffer(
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            stack_size=stack_size,
            update_horizon=update_horizon,
            gamma=gamma,
            reward_shape=(),
            terminal_dtype=bool,
        )
        self.__training_steps = 0
        self.__eval_mode = eval_mode
        self.__rng, net_rng = jax.random.split(rng)
        self.__online_params = self.__network.init(
            rngs={"params": net_rng},
            state=np.zeros(observation_shape),
            action=np.zeros(action_shape),
        )
        self.__optimizer_state = self.__optimizer.init(self.__online_params)
        self.__target_network_params = self.__online_params
        self.__epsilon_fn = epsilon_fn
        self.__epsilon_train = epsilon_train
        self.__epsilon_eval = epsilon_eval
        self.__epsilon_fn_target = epsilon_fn_target
        self.__epsilon_decay_period = epsilon_decay_period

        self.__summary_writer = tf.summary.create_file_writer(
            str(pathlib.Path(base_dir) / "update_freq={}".format(update_period))
        )
        self.__summary_writing_frequency = summary_writing_frequency

        # Perform pre-training
        self.__pre_train(num_steps=pretrain_steps)

        return

    @property
    def __summary_writer(self) -> tf.summary.SummaryWriter:
        """
        Summary file writer for the given log directory (base_dir)

        Returns:
            tf.summary.SummaryWriter
        """
        return self.__summary_writer

    def __init_logger(self, name: str, log_level: int = logging.INFO) -> logging.Logger:
        """
        Initialize a new logger for the class

        Args:
            log_level (int, optional): Log level for the system. Defaults to logging.INFO.

        Returns:
            logging.Logger: New logger
        """
        logging.basicConfig()
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        return logger

    def __pre_train(self, num_steps: int) -> None:
        """
        Pre-train the network

        Args:
            num_steps (int, optional): Number of steps over which the model should be
                pre-trained.
        """
        # Initialize the optimization state
        opt_state = self.__optimizer.init(self.__online_params)

        # Create an MSE loss function
        loss = lambda params, obs, act: jnp.mean(
            losses.mse_loss(
                -1 * jnp.sqrt(jnp.sum(jnp.square(act), axis=-1)),
                jax.vmap(lambda o, a: self.__network.apply(params, o, a)[0])(obs, act),
            )
        )

        @jax.jit
        def step(params: Any, opt_state: Any, rng: Any) -> tuple:
            """
            Step function used to pre-train the model

            Args:
                params (Any): Online params
                opt_state (Any): Current state of the optimizer
                rng (Any): PRNG key

            Returns:
                tuple: Parameters, optimizer state, loss value
            """
            state_rng, action_rng = jax.random.split(rng)

            batch_obs = 10.0 * jax.random.normal(
                state_rng,
                (128,) + self.__observation_shape,
                dtype=self.__observation_dtype,
            )

            batch_act = 10.0 * jax.random.normal(
                action_rng, (128,) + self.__action_shape, dtype=self.__action_dtype
            )

            # Compute the loss and gradient
            loss_value, grads = jax.value_and_grad(loss)(params, batch_obs, batch_act)

            # Perform search
            updates, opt_state = self.__optimizer.update(grads, opt_state, params)

            # Apply the optimization results
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss_value

        self.__logger.debug(f"Pre-training the network using {num_steps} steps")

        # Perform the pre-training
        for _ in tqdm(range(num_steps)):
            self.__rng, step_rng = jax.random.split(self.__rng)

            self.__online_params, opt_state, loss_value = step(
                self.__online_params, opt_state, step_rng
            )

            self.__logger.debug(f"Loss: {loss_value}")

        # Update the target params
        self.__target_network_params = self.__online_params

        return

    @partial(jax.jit, static_argnums=(0,))
    def select_max_q_action(
        self,
        network: Any,
        params: Any,
        state: jnp.ndarray,
        action_init: jnp.ndarray,
        max_action_range: float,
        min_action_range: float,
        max_steps: int = 1000,
    ) -> Tuple[jnp.ndarray, Any]:
        """
        Perform search to identify the action with the maximum Q value

        Args:
            network (Any): Network being trained
            params (Any): Online parameters
            state (jnp.ndarray): Current state
            action_init (jnp.ndarray): The initial action (last action)
            max_action_range (float): Maximum value that an action may be
            min_action_range (float): Minimum value that an action may be
            max_steps (int, optional): Maximum number of steps to perform when
                conducting search. Defaults to 1000.

        Returns:
            tuple: The action with the maximum Q value identified and the respective Q
                value

        TODO: Integrate support for early stopping in the search
        """
        # Negative Q Function
        neg_q_fn = lambda a: -1 * network.apply(params, state, a)[0]

        @jax.jit
        def step(
            carry: Tuple[jnp.ndarray, jnp.ndarray], step_input: Any
        ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Any]:
            """
            Step function used when selecting the action with the max Q value

            Args:
                carry (Tuple[jnp.ndarray, jnp.ndarray]): Loop carry
                step_input (Any): Slice of xs along the leading axis

            Returns:
                Tuple[Tuple[jnp.ndarray, jnp.ndarray], Any]: _description_
            """
            # Unpack the carry
            action, opt_state = carry

            # Perform search using the negative Q function
            loss, grads = jax.value_and_grad(neg_q_fn)(action)
            updates, opt_state = optimizer.update(grads, opt_state, action)
            action = optax.apply_updates(action, updates)

            # Update the carry for the next iteration
            carry = action, opt_state

            return carry, loss

        # Initialize a new optimizer to use for searching
        optimizer = optax.adam(learning_rate=1e-2)
        opt_state = optimizer.init(action_init)

        # Perform search over 100 steps
        carry, _ = jax.lax.scan(
            step, (action_init, opt_state), xs=None, length=max_steps, unroll=100
        )

        # Get the optimal action
        action, _ = carry

        # Clip the action to the acceptable range
        action = jnp.clip(action, min_action_range, max_action_range)

        # Get the respective Q value
        q = network.apply(params, state, action)[0]

        return action, q

    def select_action(
        self, state: Any, last_action: np.ndarray, rng: Any, max_steps: int = 100
    ) -> jnp.ndarray:
        """
        Select the action with the max Q value and randomly inject noise

        Args:
            state (Any): Current state observations
            last_action (np.ndarray): Last action performed
            rng (Any): PRNG key
            max_steps (int, optional): Maximum steps to perform search over. Defaults
                to 100.

        Returns:
            jnp.ndarray: Found action with randomly injected noise
        """

        # Perform search to identify the max Q action
        action, _ = self.select_max_q_action(
            self.__network, self.__online_params, state, last_action, max_steps
        )

        coin_flip_rng, noise_rng = jax.random.split(rng)

        # Get the epsilon value (This is the mean of the random variables used to
        # determine whether to generate noise)
        if self.__eval_mode:
            # Use the evaluation noise
            epsilon = self.__epsilon_eval
        else:
            if self.__epsilon_fn is not None:
                # Compute the epsilon value according to the decay function
                epsilon = self.__epsilon_fn(
                    self.__epsilon_decay_period,
                    self.__training_steps,
                    self.__min_replay_history,
                    self.__epsilon_fn_target,
                )
            else:
                # Use the static epsilon training value
                epsilon = self.__epsilon_train

        # Flip a coin to determine whether to inject noise
        if jax.random.bernoulli(coin_flip_rng, epsilon):
            action += jax.random.normal(noise_rng, action.shape)

        return action

    def store_transition(
        self,
        last_observation: Any,
        action: np.ndarray,
        reward: float,
        is_terminal: bool,
        *args: Any,
        priority: float = None,
        episode_end: bool = False,
    ) -> None:
        """
        Stores a transition when in training mode.

        Stores the following tuple in the replay buffer (last_observation, action,
        reward, is_terminal, priority).

        Args:
            last_observation (Any): Last observation, type determined via
                observation_type
            parameter in the replay_memory constructor.
            action (np.ndarray): The action taken.
            reward (float): The reward.
            is_terminal (bool): Boolean indicating if the current state is a terminal
                state.
            *args (Any): Other items to be added to the replay buffer.
            priority (float): Priority of sampling the transition. If None, the default
                priority will be used. If replay scheme is uniform, the default priority
                is 1. If the replay scheme is prioritized, the default priority is the
                maximum ever seen [Schaul et al., 2015].
            episode_end (bool): Whether this transition is the last for the episode.
                This can be different than terminal when ending the episode because
                of a timeout, for example.
        """
        # Get the priority if using a prioritized replay buffer
        if (
            isinstance(self.__replay, OutOfGraphPrioritizedReplayBuffer)
            and priority is None
        ):
            priority = self.__replay.sum_tree.max_recorded_priority

        # If we are not evaluating, save the result to the replay buffer
        if not self.__eval_mode:
            self.__replay.add(
                last_observation,
                action,
                reward,
                is_terminal,
                *args,
                priority=priority,
                episode_end=episode_end,
            )

        return

    def __sample_from_replay_buffer(self, replay: Any) -> OrderedDict:
        """
        Get a sample from the replay buffer

        Args:
            replay (Any): The replay buffer to sample from

        Returns:
            OrderedDict: Transition sample
        """
        samples = replay.sample_transition_batch()
        types = replay.get_transition_elements()

        replay_elements = OrderedDict()

        for element, element_type in zip(samples, types):
            replay_elements[element_type.name] = element

        return replay_elements

    @partial(jax.jit, static_argnums=(0,))
    def __target_q(
        self,
        network: Any,
        online_params: Any,
        target_params: Any,
        next_states: Any,
        rewards,
        terminals,
        actions,
        cumulative_gamma,
    ):
        def q_target(state, action):
            return network.apply(target_params, state, action)[0]

        select_q = partial(
            self.select_max_q_action,
            network_def=network,
            params=online_params,
            max_steps=100,
        )

        # Do the DDQN thing of selecting the next q with online params
        # and evaluate its q-value with target params
        next_actions = jax.vmap(select_q)(state=next_states, action_init=actions)[0]
        q_vals = jax.vmap(q_target)(state=next_states, action=next_actions)
        replay_next_qt_max = q_vals

        # Calculate the Bellman target value.
        #   Q_t = R_t + \gamma^N * Q'_t+1
        # where,
        #   Q'_t+1 = \argmax_a Q(S_t+1, a)
        #          (or) 0 if S_t is a terminal state,
        # and
        #   N is the update horizon (by default, N=1).

        return jax.lax.stop_gradient(
            rewards + cumulative_gamma * replay_next_qt_max * (1.0 - terminals)
        )

    @partial(jax.jit, static_argnums=(0,))
    def train(
        self,
        network,
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
        loss_type="huber",
    ):
        # Define a loss function to use for training
        def loss_fn(params, target):
            def q_online(state, action):
                return network.apply(params, state, action)[0]

            replay_chosen_q = jax.vmap(q_online)(states, actions)
            if loss_type == "huber":
                return jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))

            return jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

        target = self.__target_q(
            network,
            online_params,
            target_params,
            jnp.squeeze(next_states),
            rewards,
            terminals,
            actions,
            cumulative_gamma,
        )

        grad_fn = jax.value_and_grad(loss_fn)

        loss, grad = grad_fn(online_params, target)

        updates, optimizer_state = optimizer.update(
            grad, optimizer_state, params=online_params
        )

        online_params = optax.apply_updates(online_params, updates)

        return optimizer_state, online_params, loss

    def train_step(self) -> None:
        """
        Runs a single training step.

        Runs training if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_params to target_network_params if training
        steps is a multiple of target update period.
        """
        # Run a train op at the rate of update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self.__replay.add_count > self.__min_replay_history:
            if self.__training_steps % self.__update_period == 0:
                # Get the replay elements
                replay_elements = self.__sample_from_replay_buffer(self.__replay)

                self._rng, train_rng = jax.random.split(self._rng)

                self.optimizer_state, self.online_params, loss = self.train(
                    network_def=self.__network,
                    online_params=self.__online_params,
                    target_params=self.__target_network_params,
                    optimizer=self.__optimizer,
                    optimizer_state=self.__optimizer_state,
                    rng=train_rng,
                    states=replay_elements["state"],
                    actions=replay_elements["action"],
                    next_states=replay_elements["next_state"],
                    rewards=replay_elements["reward"],
                    terminals=replay_elements["terminal"],
                    cumulative_gamma=self.__cumulative_gamma,
                    loss_type=self.__loss_type,
                )

                if (
                    self.__summary_writer is not None
                    and self.__training_steps > 0
                    and self.__training_steps % self.__summary_writing_frequency == 0
                ):
                    with self.__summary_writer.as_default():
                        tf.summary.scalar(
                            self.__loss_type, loss, step=self.__training_steps
                        )

            # Sync the target network params with the online params
            if self.__training_steps % self.__target_update_period == 0:
                self.__target_network_params = self.__online_params

            self.__training_steps += 1

            return
