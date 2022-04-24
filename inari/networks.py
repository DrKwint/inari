import os
import pickle
import random
from tabnanny import check

import gin
import gym
import jax
from flax import linen as nn
from jax import numpy as jnp


class CDQNNetwork(nn.Module):
    num_layers: int = 2
    hidden_units: int = 128

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray,
                 lambd: float) -> jnp.ndarray:
        kernel_initializer = jax.nn.initializers.glorot_uniform()

        # Preprocess inputs
        a = action.reshape(-1)  # flatten
        s = state.astype(jnp.float32)
        s = state.reshape(-1)  # flatten
        lambd = jnp.reshape(lambd, [-1])
        lambd = lambd / (lambd + 1)
        x = jnp.concatenate((s, a, lambd))

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_units,
                         kernel_init=kernel_initializer)(x)
            x = nn.relu(x)

        return nn.Dense(features=2, kernel_init=kernel_initializer)(x)