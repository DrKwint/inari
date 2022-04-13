from inari.cqdn import CDQN
import safety_gym
import gym
import jax
from flax import linen as nn
import gin
from jax import numpy as jnp
import rlax
import numpy as np
from flax import linen as nn
import gin
import jax
from tqdm import tqdm
import tensorflow as tf
import os
import random


class CDQNNetwork(nn.Module):
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


def run_agent(seed=0):
    gin.parse_config_file("./inari/cdqn.gin")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rng = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    robot = "point".capitalize()
    task = "goal1".capitalize()
    env_name = "Safexp-" + robot + task + "-v0"
    env = gym.make(env_name)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cdqn = CDQN(CDQNNetwork, obs_shape, act_shape, base_dir="./tests/", rng=rng)

    obs = env.reset()
    act = np.zeros(act_shape)
    done = False
    for epoch in range(333):
        epoch_ep_rewards = []
        cum_ep_reward = 0
        with cdqn.summary_writer.as_default():
            for step in tqdm(range(30000)):
                rng, action_sample_rng = jax.random.split(rng)
                act = cdqn.select_action(obs, act, action_sample_rng, 0.25)
                prev_obs = obs
                obs, rew, done, info = env.step(act)
                cost = info["cost"]
                cum_ep_reward += rew
                cdqn.store_transition(prev_obs, np.array(act), rew, done)
                cdqn.train_step()
                tf.summary.scalar(
                    "act_magnitude",
                    np.sqrt(np.sum(np.square(act))),
                    step=epoch * 30000 + step,
                )
                if done:
                    tf.summary.scalar(
                        "ep_reward", cum_ep_reward, step=epoch * 30000 + step
                    )
                    obs = env.reset()
                    epoch_ep_rewards.append(cum_ep_reward)
                    cum_ep_reward = 0
        print("Mean episode reward:", np.mean(epoch_ep_rewards))


if __name__ == "__main__":
    run_agent()
