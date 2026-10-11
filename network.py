import jax
import jax.numpy as jnp
import flax.linen as nn
 
# -------------------------------------------------------------------
# THE AI BRAIN NETWORK (Actor-Critic Architecture)
# -------------------------------------------------------------------
 
class ActorCritic(nn.Module):
    """Neural network containing both the Actor (decides actions) and the Critic (estimates value)."""
 
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        x is the observation array: [cart_x, cart_speed, pole_angle, pole_speed]
        """
 
        # 1. Shared base layers
        dense = nn.Dense(features=64)(x)
        dense = nn.tanh(dense)
 
        dense = nn.Dense(features=64)(dense)
        dense = nn.tanh(dense)
 
        # 2. Actor branch: mean push force, squashed to [-1, 1]
        actor_mean = nn.Dense(features=1)(dense)
        actor_mean = jnp.squeeze(actor_mean, axis=-1)
        actor_mean = jnp.tanh(actor_mean)
 
        # 3. Critic branch: expected score from the current state
        critic_value = nn.Dense(features=1)(dense)
        critic_value = jnp.squeeze(critic_value, axis=-1)
 
        return actor_mean, critic_value
 