import os
import pickle
import random
from tabnanny import check

import gin
import gym
import jax
import numpy as np
import safety_gym
import tensorflow as tf
from flax import linen as nn
from jax import numpy as jnp
from ray import tune
import ray
from ray.tune.suggest.hyperopt import HyperOptSearch

from inari.cdqn import CDQN
from inari.networks import CDQNNetwork


class Trainable(tune.Trainable):

    def setup(self, config):

        import safety_gym

        # seed everything
        seed = 0
        random.seed(seed)
        rng = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        robot = 'point'.capitalize()
        task = 'goal1'.capitalize()
        env_name = 'Safexp-' + robot + task + '-v0'
        env = gym.make(env_name)

        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        cdqn = CDQN(CDQNNetwork(parent=None),
                    obs_shape,
                    act_shape,
                    rng=rng,
                    **config)

        self.env = env
        self.cdqn = cdqn
        self.act_shape = act_shape
        self.epoch = 0
        self.epoch_rewards = []
        self.epoch_costs = []
        self.total_cost = 0

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pkl")
        with open(checkpoint_path, 'wb') as checkpoint_file:
            pickle.dump(
                (self.cdqn.get_full_state(), self.epoch, self.epoch_rewards,
                 self.epoch_costs, np.random.get_state()), checkpoint_file)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pkl")
        with open(checkpoint_path, 'rb') as checkpoint_file:
            cdqn_state, self.epoch, self.epoch_rewards, self.epoch_costs, rng_state = pickle.load(
                checkpoint_file)
        self.cdqn.set_full_state(cdqn_state)
        np.random.set_state(rng_state)

    def step(self):  # This is called iteratively.
        obs = self.env.reset()
        act = np.zeros(self.act_shape)
        done = False

        ep_rewards = []
        ep_costs = []
        cum_reward = 0
        cum_cost = 0
        for step in range(30000):
            # choose an action and step
            act, qs = self.cdqn.select_action(obs, act)
            est_rew_q, est_cost_q = qs[0], qs[1]
            prev_obs = obs
            obs, rew, done, info = self.env.step(act)
            cost = info['cost']

            # store transition and train
            cum_reward += rew
            cum_cost += cost
            self.cdqn.store_transition(prev_obs, np.array(act), rew, done,
                                       cost)
            self.cdqn.train_step()

            with self.cdqn.summary_writer.as_default():
                tf.summary.scalar("step_q_est/reward",
                                  est_rew_q,
                                  step=self.epoch * 30000 + step)
                tf.summary.scalar("step_q_est/cost",
                                  est_cost_q,
                                  step=self.epoch * 30000 + step)

            # episode conclusion
            if done:
                self.total_cost += cum_cost
                with self.cdqn.summary_writer.as_default():
                    tf.summary.scalar("episode/reward",
                                      cum_reward,
                                      step=self.epoch * 30000 + step)
                    tf.summary.scalar("episode/cost",
                                      cum_cost,
                                      step=self.epoch * 30000 + step)
                    tf.summary.scalar("cost_rate",
                                      self.total_cost /
                                      (self.epoch * 30000 + step),
                                      step=self.epoch * 30000 + step)
                    tf.summary.scalar("cost_penalty_coeff",
                                      self.cdqn.cost_penalty,
                                      step=self.epoch * 30000 + step)
                self.cdqn.pid_update(cum_cost,
                                     p_param=0.5 * 1e-2,
                                     i_param=1e-3 * 1e-2,
                                     d_param=1. * 1e-2,
                                     cost_limit=25)
                if self.cdqn.cost_penalty < 0.:
                    self.cdqn.cost_penalty = 0.

                obs = self.env.reset()
                ep_rewards.append(cum_reward)
                ep_costs.append(cum_cost)
                cum_reward = 0
                cum_cost = 0

        self.epoch += 1
        self.epoch_rewards.append(np.mean(ep_rewards))
        self.epoch_costs.append(np.mean(ep_costs))
        return dict(episode_reward_mean=np.mean(ep_rewards),
                    episode_cost_mean=np.mean(ep_costs),
                    total_cum_cost=np.sum(self.epoch_costs) *
                    30)  # 30 episodes per epoch


def hyperparam_search():
    #ray.init(address=None, local_mode=True, num_gpus=1)
    search_space = {
        "epsilon_train": .095,  #tune.quniform(0.001, 0.1, 0.001),
        "num_search_steps": 290,  #tune.qlograndint(10, 1000, 10),
        "target_update_period": 3000,  #tune.qrandint(1000, 6000, 1000),
        "update_period": 2  #tune.randint(1, 5),
    }
    analysis = tune.run(
        Trainable,
        config=search_space,
        resources_per_trial={
            'cpu': 1,
            'gpu': 1
        },
        local_dir="./test_results",
        name="pointgoal1",
        num_samples=-1,
        #search_alg=HyperOptSearch(metric='episode_reward_mean', mode='max'),
        scheduler=tune.schedulers.ASHAScheduler(mode='max',
                                                metric='episode_reward_mean',
                                                max_t=333),
        #sync_config=ray.tune.syncer.SyncConfig(),
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=1,
        checkpoint_freq=100,
        resume='AUTO')
    print(analysis.get_best_config(metric="episode_reward_mean", mode="max"))
    #ray.shutdown()


if __name__ == '__main__':
    gin.parse_config_file('/home/equint/GitHub/inari/inari/cdqn.gin')
    hyperparam_search()
    """
    exp = Trainable()
    exp.setup(
        config={
            "epsilon_train": .095,  #tune.quniform(0.001, 0.1, 0.001),
            "num_search_steps": 290,  #tune.qlograndint(10, 1000, 10),
            "target_update_period": 3000,  #tune.qrandint(1000, 6000, 1000),
            "update_period": 2  #tune.randint(1, 5),
        })
    exp.step()
    """
