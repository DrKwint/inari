import jax
import optax
import logging
import pathlib
import numpy as np
import tensorflow as tf
from jax import numpy as jnp
from typing import Any, Union
import dopamine.jax.losses as losses
from dopamine.replay_memory.prioritized_replay_buffer import (
    OutOfGraphPrioritizedReplayBuffer,
)


class CDQN:
    def __init__(
        self,
        network: Any,
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
        observation_dtype = np.float32,
        action_dtype = np.float32,
        stack_size: int = 1,
        pretrain_steps: int = 100000,
        log_level: int = logging.INFO,
    ) -> None:
        self.__logger = self.__init_logger("CDQN", log_level=log_level)

        self.__logger.debug("Initializing the CDQN")

        self.__network = network()
        self.__observation_shape = observation_shape
        self.__observation_dtype = observation_dtype
        self.__action_shape = action_shape
        self.__action_dtype = action_dtype
        self.__update_horizon = update_horizon
        self.__gamma = gamma
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

        self.__summary_writer = tf.summary.create_file_writer(
            str(pathlib.Path(base_dir) / "update_freq={}".format(update_period))
        )

        self.__summary_writing_frequency = summary_writing_frequency
        self.__training_steps = 0
        self.__eval_mode = False
        self.__state = np.zeros(observation_shape)
        self.__action = np.zeros(action_shape)
        self.__rng, net_rng = jax.random.split(rng)
        self.__online_params = self.__network.init(
            rngs={"params": net_rng}, state=self.__state, action=self.__action
        )
        self.__optimizer_state = self.__optimizer.init(self.__online_params)
        self.__target_network_params = self.__online_params

        # Perform pre-training
        self.__pre_train(num_steps=pretrain_steps)

        return

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
                pretrained. Defaults to 100000.
        """
        # Initialize the optimization state
        opt_state = self.__optimizer.init(self.__online_params)

        # Create a loss function using MSE
        loss = lambda params, obs, act: jnp.mean(
            losses.mse_loss(
                -1 * jnp.sqrt(jnp.sum(jnp.square(act), axis=-1)),
                jax.vmap(lambda o, a: self.__network.apply(params, o, a)[0])(obs, act),
            )
        )

        @jax.jit
        def step(params: Any, opt_state: Any, rng: Any) -> tuple:
            """
            Step function used to train the model

            Args:
                params (Any): _description_
                opt_state (Any): _description_
                rng (Any): _description_

            Returns:
                tuple: _description_
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
            updates, opt_state = self.optimizer.update(grads, opt_state, params)

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


    def __select_action(self):
        """
        TODO: Need to integrate epsilon and epsilon decay function
        """
        return

    def __store_transition(self):
        """
        TODO
        """
        return

    def __train_step(self):
        """
        TODO
        """
        return

    def __sample_from_replay_buffer(self):
        """
        TODO
        """
        return

    def train(self):
        """
        TODO: Going to abstract some of the training functionality to improve usability
        """