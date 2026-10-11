import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment import EnvParams, reset, step
from network import ActorCritic
from train import train, load_params, PARAMS_PATH

# The physics "length" is the HALF-length of the pole, so the drawn pole is twice that.
POLE_DRAW_LENGTH = 2.0 * EnvParams().length


def run_single_episode(params_network, max_steps=500, seed=99):
    """Runs one episode with the trained model and records cart/pole history."""
    model = ActorCritic()
    env_params = EnvParams()

    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    obs, state = reset(reset_key, env_params)

    xs = []
    thetas = []

    for _ in range(max_steps):
        xs.append(float(state.x))
        thetas.append(float(state.theta))

        # Deterministic action from the trained network (no exploration noise)
        actor_mean, _ = model.apply(params_network, obs)
        action = jnp.clip(actor_mean, -1.0, 1.0)

        rng, step_key = jax.random.split(rng)
        obs, state, reward, done, _ = step(step_key, state, action, env_params)

        if done:
            break

    return xs, thetas


def make_animation(xs, thetas, title="Trained JAX Cart-Pole Agent", interval=20):
    """Builds a matplotlib animation from recorded cart positions and pole angles."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.2, 1.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)

    ax.plot([-2.4, 2.4], [0, 0], "k-", lw=4)  # Track
    cart_rect = plt.Rectangle((-0.2, -0.1), 0.4, 0.2, fc="royalblue")
    ax.add_patch(cart_rect)
    (pole_line,) = ax.plot([], [], "r-", lw=5, marker="o", markersize=10)

    def init():
        cart_rect.set_xy((-0.2, -0.1))
        pole_line.set_data([], [])
        return cart_rect, pole_line

    def animate(i):
        x = xs[i]
        theta = thetas[i]

        cart_rect.set_xy((x - 0.2, -0.1))

        pole_x = x + POLE_DRAW_LENGTH * np.sin(theta)
        pole_y = POLE_DRAW_LENGTH * np.cos(theta)
        pole_line.set_data([x, pole_x], [0, pole_y])
        return cart_rect, pole_line

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(xs), interval=interval, blit=True
    )
    return fig, ani


def get_trained_params():
    """Loads saved weights if they exist, otherwise trains a new model."""
    if os.path.exists(PARAMS_PATH):
        print(f"Loading saved weights from '{PARAMS_PATH}'...")
        return load_params()

    print("No saved weights found. Training a new model first...")
    return train()


def generate_gif(gif_path="cartpole_trained.gif"):
    params_network = get_trained_params()

    print("Running a recorded test episode...")
    xs, thetas = run_single_episode(params_network, max_steps=500)
    print(f"Episode lasted {len(xs)} steps.")

    print("Rendering animation GIF...")
    fig, ani = make_animation(xs, thetas)
    ani.save(gif_path, writer="pillow", fps=30)
    plt.close(fig)
    print(f"Animation successfully saved to '{gif_path}'!")


if __name__ == "__main__":
    generate_gif()