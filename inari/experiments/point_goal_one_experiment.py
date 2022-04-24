import os
import gin
import gym
import jax
import random
import argparse
import safety_gym
import numpy as np
from ray import tune
import tensorflow as tf
from typing import Optional
from jax import numpy as jnp
from flax import linen as nn
from inari.agents.cdqn import CDQN


class CDQNNetwork(nn.Module):
    """
    Network to be trained using continuous deep Q learning
    """

    num_layers: int = 2
    hidden_units: int = 128

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        kernel_initializer = jax.nn.initializers.glorot_uniform()

        # Preprocess inputs
        a = action.reshape(-1)  # flatten
        s = state.astype(jnp.float32)
        s = state.reshape(-1)  # flatten
        x = jnp.concatenate((s, a))

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_units, kernel_init=kernel_initializer)(x)
            x = nn.relu(x)

        return nn.Dense(features=1, kernel_init=kernel_initializer)(x)


def run_agent(
    config: dict,
    checkpoint_dir: Optional[str] = None,
    epochs: int = 333,
    timesteps: int = 30000,
) -> float:
    """
    Train an agent using the configured hyperparameters

    Args:
        config (dict): The hyperparameters to use for the current trial
        checkpoint_dir (Optional[str], optional): The directory that checkpoints should
            be saved to. Defaults to None.
        epochs (int, optional): The number of epochs to train for
        timesteps (int, optional): The total number of timesteps per epoch to train for

    Returns:
        float: The maximum reward obtained during training
    """
    # seed everything
    seed = 0
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rng = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Configure the environment
    robot = "point".capitalize()
    task = "goal1".capitalize()
    env_name = "Safexp-" + robot + task + "-v0"
    env = gym.make(env_name)

    # Get the action and observation shapes
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Construct a new CDQN
    cdqn = CDQN(
        CDQNNetwork, obs_shape, act_shape, base_dir="./tests/", rng=rng, **config
    )

    # Reset the environment
    obs = env.reset()

    # Initialize the action
    act = np.zeros(act_shape)

    # Flag indicating whether the episode is complete
    done = False

    # Store the rewards for each epoch
    epoch_rewards = []

    for epoch in range(epochs):
        ep_rewards = []
        cum_reward = 0
        cum_cost = 0

        for step in range(timesteps):
            # Choose an action and step
            act = cdqn.select_action(obs, act)
            prev_obs = obs
            obs, rew, done, info = env.step(act)
            cost = info["cost"]

            # Store transition and train
            cum_reward += rew
            cum_cost += cost
            cdqn.store_transition(prev_obs, np.array(act), rew, done)
            cdqn.train_step()

            # Episode conclusion
            if done:
                with cdqn.summary_writer.as_default():
                    tf.summary.scalar(
                        "ep_reward", cum_reward, step=epoch * 30000 + step
                    )
                    tf.summary.scalar("ep_cost", cum_cost, step=epoch * 30000 + step)

                obs = env.reset()
                ep_rewards.append(cum_reward)
                cum_reward = 0
                cum_cost = 0

        tune.report(episode_reward_mean=np.mean(ep_rewards))
        print("Mean episode reward at epoch {}:".format(epoch), np.mean(ep_rewards))

    return max(epoch_rewards)

def hyperparam_search() -> None:
    """
    Perform hyperparameter search to determine the optimal hyperparameters
    """

    search_space = {
        "epsilon_train": tune.quniform(0.001, 0.1, 0.001),
        "num_search_steps": tune.qlograndint(10, 1000, 10),
        "target_update_period": tune.qrandint(1000, 10000, 1000),
        "update_period": tune.randint(1, 5),
    }

    analysis = tune.run(
        run_agent,
        config=search_space,
        resources_per_trial={"gpu": 1},
        local_dir="./results",
        name="pointgoal1",
        scheduler=tune.schedulers.ASHAScheduler(
            mode="max", metric="episode_reward_mean", max_t=333
        ),
    )

    print(analysis.get_best_config(metric="score", mode="max"))

    return


def main() -> None:
    # Configure the arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("network_configs", type=str, help="The path to the CDQN gin configurations")
    args = parser.parse_args()

    # Load the configurations
    gin.parse_config_file("network_configs")
    
    # Perform hyperparameter search
    hyperparam_search()

    return


if __name__ == "__main__":
    main()