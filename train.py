import jax
import jax.numpy as jnp
import optax
import time
import pickle

from environment import EnvParams
from network import ActorCritic
from rollout import vmap_reset, collect_rollout, NOISE_STD

# -------------------------------------------------------------------
# CONFIGURATION & HYPERPARAMETERS
# -------------------------------------------------------------------
NUM_ENVS = 2048        # Run 2,048 parallel games at once
NUM_STEPS = 100        # Each rollout lasts 100 timesteps
UPDATES = 500          # Number of learning update cycles (30 was far too few)
LEARNING_RATE = 1e-3   # Adam learning rate
REWARD_SCALE = 0.1    # Shrinks returns so the critic loss doesn't swamp the actor loss
PRINT_EVERY = 10       # Print progress every N updates
PARAMS_PATH = "trained_params.pkl"  # Where trained weights are saved

# -------------------------------------------------------------------
# LOSS FUNCTION (Policy Gradient / Advantage Learning)
# -------------------------------------------------------------------

def compute_loss(params_network, trajectory):
    """Calculates policy gradients using Gaussian log-probabilities."""
    model = ActorCritic()

    # Re-evaluate the trajectory with current parameters.
    # obs shape: (NUM_STEPS, NUM_ENVS, 4). Dense layers handle the extra dims.
    actor_means, values = model.apply(params_network, trajectory["obs"])

    rewards = trajectory["rewards"] * REWARD_SCALE       # (NUM_STEPS, NUM_ENVS)
    dones = trajectory["dones"].astype(jnp.float32)      # (NUM_STEPS, NUM_ENVS)

    # 1. Discounted returns-to-go, computed backwards.
    #    (1 - done) stops the return from flowing across an episode reset.
    def scan_fn(running_return, x):
        r, d = x
        running_return = r + 0.99 * (1.0 - d) * running_return
        return running_return, running_return

    # Start from the critic's estimate of the state after the last step,
    # so the rollout cut-off isn't treated as "the future is worth zero".
    last_value = jax.lax.stop_gradient(trajectory["last_value"])
    _, returns = jax.lax.scan(
        scan_fn, last_value, (rewards, dones), reverse=True
    )

    # 2. Advantage: how much better the real return was than the critic expected.
    #    stop_gradient keeps the actor loss from changing the critic.
    advantages = jax.lax.stop_gradient(returns - values)
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # 3. Actor loss (Gaussian log-probability, fixed std = 0.5 to match rollout noise)
    std = NOISE_STD
    log_probs = -0.5 * jnp.square((trajectory["actions"] - actor_means) / std)
    actor_loss = -jnp.mean(log_probs * advantages)

    # 4. Critic loss (value prediction error)
    critic_loss = jnp.mean(jnp.square(returns - values))

    return actor_loss + 0.5 * critic_loss


# -------------------------------------------------------------------
# MAIN TRAINING LOOP
# -------------------------------------------------------------------

def train():
    print("==================================================")
    print("      JAX MINI-BRAX ROBOTICS TRAINING LOOP        ")
    print("==================================================")
    print(f"Parallel Environments: {NUM_ENVS}")
    print(f"Steps Per Rollout:     {NUM_STEPS}")
    print(f"Total Steps Per Epoch: {NUM_ENVS * NUM_STEPS:,}")
    print("--------------------------------------------------\n")

    # 1. Initialize random keys
    rng = jax.random.PRNGKey(42)
    rng, init_key, reset_key = jax.random.split(rng, 3)

    # 2. Initialize network and optimizer
    model = ActorCritic()
    dummy_obs = jnp.zeros((1, 4))
    params_network = model.init(init_key, dummy_obs)

    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params_network)

    # 3. Reset all parallel environments
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    obs, states = vmap_reset(reset_keys, EnvParams())

    # Compile the training step for speed
    @jax.jit
    def train_step(params_network, opt_state, obs, states, rng):
        rng, rollout_key = jax.random.split(rng)

        # Collect rollouts across all parallel environments
        obs, states, trajectory = collect_rollout(
            rollout_key, params_network, states, obs, num_steps=NUM_STEPS
        )

        # Critic's value of the state we ended on (for bootstrapping the returns)
        _, last_value = ActorCritic().apply(params_network, obs)
        trajectory["last_value"] = last_value

        # Loss and gradients
        loss, grads = jax.value_and_grad(compute_loss)(params_network, trajectory)

        # Update weights with Adam
        updates, opt_state = optimizer.update(grads, opt_state, params_network)
        params_network = optax.apply_updates(params_network, updates)

        avg_reward = jnp.mean(trajectory["rewards"])
        done_rate = jnp.mean(trajectory["dones"].astype(jnp.float32))
        return params_network, opt_state, obs, states, rng, loss, avg_reward, done_rate

    # --- Run training iterations ---
    start_time = time.time()

    for update in range(1, UPDATES + 1):
        (params_network, opt_state, obs, states, rng,
         loss, avg_reward, done_rate) = train_step(
            params_network, opt_state, obs, states, rng
        )

        if update % PRINT_EVERY == 0 or update == 1:
            steps_processed = update * NUM_ENVS * NUM_STEPS
            elapsed = time.time() - start_time
            sps = steps_processed / elapsed

            print(
                f"Epoch {update:03d}/{UPDATES} | Loss: {loss:8.4f} | "
                f"Avg Step Reward: {avg_reward:.3f} | Done Rate: {done_rate:.4f} | "
                f"Speed: {sps:,.0f} Steps/Sec"
            )

    print("\n--------------------------------------------------")
    print("Training Complete!")
    print(f"Total Time Elapsed: {time.time() - start_time:.2f} seconds")
    print("==================================================")

    # Save the trained weights so visualize.py / runall.py can reuse them
    with open(PARAMS_PATH, "wb") as f:
        pickle.dump(jax.device_get(params_network), f)
    print(f"Saved trained weights to '{PARAMS_PATH}'")

    return params_network


def load_params(path=PARAMS_PATH):
    """Loads previously saved network weights."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train()