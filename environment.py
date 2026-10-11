import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class EnvParams(NamedTuple):
    """Holds fixed physics rules (gravity, weights, lengths)."""
    gravity: float = 9.81
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    length: float = 0.5          # Half-length of the pole
    dt: float = 0.02             # Time step interval
    max_steps: int = 500
    force_mag: float = 10.0      # Standard CartPole force (1.0 was too weak to recover the pole)


class EnvState(NamedTuple):
    """Holds dynamic variables that change every step."""
    x: jax.Array                 # Cart position
    x_dot: jax.Array             # Cart velocity
    theta: jax.Array             # Pole angle
    theta_dot: jax.Array         # Pole angular velocity
    time_step: jax.Array         # Current step counter


def reset(rng: jax.Array, params: EnvParams = EnvParams()) -> Tuple[jax.Array, EnvState]:
    """Resets the environment with a small random start so the policy sees recovery situations."""
    rng, key = jax.random.split(rng)

    noise = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)

    state = EnvState(
        x=noise[0],
        x_dot=noise[1],
        theta=noise[2],
        theta_dot=noise[3],
        time_step=jnp.array(0, dtype=jnp.int32),
    )

    obs = jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])
    return obs, state


@jax.jit
def step(rng: jax.Array, state: EnvState, action: jax.Array, params: EnvParams = EnvParams()):
    """Performs a single physics step using semi-implicit Euler integration."""

    # Handles both 0D scalars and 1D arrays safely
    force = jnp.clip(jnp.squeeze(action), -1.0, 1.0) * params.force_mag

    x = state.x
    x_dot = state.x_dot
    theta = state.theta
    theta_dot = state.theta_dot

    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    # Equations of motion for Cart-Pole
    temp = (force + params.mass_pole * params.length * theta_dot**2 * sintheta) / (
        params.mass_cart + params.mass_pole
    )

    theta_acc = (params.gravity * sintheta - costheta * temp) / (
        params.length * (4.0 / 3.0 - params.mass_pole * costheta**2 / (params.mass_cart + params.mass_pole))
    )

    x_acc = temp - params.mass_pole * params.length * theta_acc * costheta / (
        params.mass_cart + params.mass_pole
    )

    # Update state via semi-implicit Euler integration
    x_dot = x_dot + params.dt * x_acc
    x = x + params.dt * x_dot

    theta_dot = theta_dot + params.dt * theta_acc
    theta = theta + params.dt * theta_dot

    time_step = state.time_step + 1

    state = EnvState(
        x=x,
        x_dot=x_dot,
        theta=theta,
        theta_dot=theta_dot,
        time_step=time_step,
    )

    # Failure conditions (track bounds or pole falling past ~12 degrees)
    done_x = jnp.abs(x) > 2.4
    done_theta = jnp.abs(theta) > 0.2095  # ~12 degrees
    done_time = time_step >= params.max_steps
    done = jnp.logical_or(jnp.logical_or(done_x, done_theta), done_time)

    # Smooth reward shaping: rewards staying centered and upright
    x_reward = 1.0 - jnp.abs(x) / 2.4
    theta_reward = 1.0 - jnp.abs(theta) / 0.2095
    reward = jnp.where(done, 0.0, x_reward + theta_reward)

    obs = jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])
    info = {}

    return obs, state, reward, done, info