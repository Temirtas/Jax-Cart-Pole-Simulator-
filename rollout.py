import jax
import jax.numpy as jnp
from environment import EnvParams, reset, step
from network import ActorCritic

# -------------------------------------------------------------------
# 1. VECTORIZED GAME FUNCTIONS (Running thousands of games at once)
# -------------------------------------------------------------------

# vmap automatically duplicates the single-game reset and step functions
# so they process a batch of environments in parallel.
vmap_reset = jax.vmap(reset, in_axes=(0, None))
vmap_step = jax.vmap(step, in_axes=(0, 0, 0, None))

NOISE_STD = 0.3  # Exploration noise; train.py uses the same value in the log-prob


# -------------------------------------------------------------------
# 2. PARALLEL TRAJECTORY COLLECTOR
# -------------------------------------------------------------------

def collect_rollout(
    rng,
    params_network,
    states,
    obs,
    num_steps=100,
    env_params=EnvParams(),
):
    """
    Runs num_steps of physics across ALL parallel games at once on the GPU.
    Finished environments are automatically reset so they keep producing
    useful data.
    """
    model = ActorCritic()

    def step_fn(carry, _):
        """Executed once per time step inside jax.lax.scan."""
        obs, states, rng = carry

        # Split random keys for: exploration noise, env step, env reset
        rng, action_key, step_key, reset_key = jax.random.split(rng, 4)

        # 1. Ask the network for action means and critic values
        actor_means, values = model.apply(params_network, obs)

        # 2. Add random noise to the actions so the AI explores
        num_envs = obs.shape[0]
        step_keys = jax.random.split(step_key, num_envs)

        noise = jax.random.normal(action_key, shape=actor_means.shape) * NOISE_STD
        actions = jnp.clip(actor_means + noise, -1.0, 1.0)

        # 3. Step all parallel games forward at once
        next_obs, next_states, rewards, dones, _ = vmap_step(
            step_keys, states, actions, env_params
        )

        # 4. AUTO-RESET: replace finished environments with fresh ones
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_states = vmap_reset(reset_keys, env_params)

        next_obs = jnp.where(dones[:, None], reset_obs, next_obs)
        next_states = jax.tree_util.tree_map(
            lambda fresh, old: jnp.where(dones, fresh, old),
            reset_states,
            next_states,
        )

        # Save this step's data
        transition = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
        }

        return (next_obs, next_states, rng), transition

    # jax.lax.scan runs the time loop entirely on the GPU
    (final_obs, final_states, _), trajectory = jax.lax.scan(
        step_fn, (obs, states, rng), None, length=num_steps
    )

    return final_obs, final_states, trajectory