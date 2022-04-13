import os
import random

import gin
import gym
import jax
import numpy as np
import safety_gym
import tensorflow as tf
from flax import linen as nn
from jax import numpy as jnp
from ray import tune

from inari.cqdn import CDQN


def run_agent(config, checkpoint_dir=None):
    import safety_gym

    # seed everything
    seed = 0
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    rng = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    robot = 'point'.capitalize()
    task = 'goal1'.capitalize()
    env_name = 'Safexp-' + robot + task + '-v0'
    env = gym.make(env_name)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    class CDQNNetwork(nn.Module):
        num_layers: int = 2
        hidden_units: int = 128

        @nn.compact
        def __call__(self, state: jnp.ndarray,
                     action: jnp.ndarray) -> jnp.ndarray:
            kernel_initializer = jax.nn.initializers.glorot_uniform()

            # Preprocess inputs
            a = action.reshape(-1)  # flatten
            s = state.astype(jnp.float32)
            s = state.reshape(-1)  # flatten
            x = jnp.concatenate((s, a))

            for _ in range(self.num_layers):
                x = nn.Dense(features=self.hidden_units,
                             kernel_init=kernel_initializer)(x)
                x = nn.relu(x)

            return nn.Dense(features=1, kernel_init=kernel_initializer)(x)

    cdqn = CDQN(CDQNNetwork,
                obs_shape,
                act_shape,
                base_dir='./tests/',
                rng=rng,
                **config)

    obs = env.reset()
    act = np.zeros(act_shape)
    done = False

    epoch_rewards = []
    for epoch in range(333):
        ep_rewards = []
        cum_reward = 0
        cum_cost = 0
        for step in range(30000):
            # choose an action and step
            act = cdqn.select_action(obs, act)
            prev_obs = obs
            obs, rew, done, info = env.step(act)
            cost = info['cost']

            # store transition and train
            cum_reward += rew
            cum_cost += cost
            cdqn.store_transition(prev_obs, np.array(act), rew, done)
            cdqn.train_step()

            # episode conclusion
            if done:
                with cdqn.summary_writer.as_default():
                    tf.summary.scalar("ep_reward",
                                      cum_reward,
                                      step=epoch * 30000 + step)
                    tf.summary.scalar("ep_cost",
                                      cum_cost,
                                      step=epoch * 30000 + step)

                obs = env.reset()
                ep_rewards.append(cum_reward)
                cum_reward = 0
                cum_cost = 0

        tune.report(episode_reward_mean=np.mean(ep_rewards))
        print("Mean episode reward at epoch {}:".format(epoch),
              np.mean(ep_rewards))
    return max(epoch_rewards)


def hyperparam_search():

    search_space = {
        "epsilon_train": tune.quniform(0.001, 0.1, 0.001),
        "num_search_steps": tune.qlograndint(10, 1000, 10),
        "target_update_period": tune.qrandint(1000, 10000, 1000),
        "update_period": tune.randint(1, 5),
    }
    analysis = tune.run(run_agent,
                        config=search_space,
                        resources_per_trial={'gpu': 1},
                        local_dir="./results",
                        name="pointgoal1",
                        scheduler=tune.schedulers.ASHAScheduler(
                            mode='max',
                            metric='episode_reward_mean',
                            max_t=333))
    print(analysis.get_best_config(metric="score", mode="max"))


if __name__ == '__main__':
    gin.parse_config_file('/home/equint/GitHub/inari/inari/cdqn.gin')
    hyperparam_search()
