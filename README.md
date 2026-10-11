# JAX Cart-Pole Simulator

A reinforcement learning project written from scratch in **JAX**. A small neural network learns to balance a pole on a moving cart by pushing the cart left or right. The physics, the parallel game runner, and the learning algorithm are all written in plain JAX and Flax, with no RL library. Thousands of games run at the same time, so training takes under a minute.

![Trained agent balancing the pole](cartpole_demo.gif)

## Results

After 500 training updates (about 45 seconds on the author's laptop), the agent keeps the pole upright for the full 500-step episode (10 seconds of simulated time).

| Metric | Start of training | End of training |
|--------|-------------------|-----------------|
| Average reward per step (max 2.0) | ~1.5 | ~1.78 |
| Fraction of steps where a game ends | ~0.01 | ~0.002 (almost only the time limit) |

A done rate of about 0.002 means roughly one game ending per 500 steps, which is the time limit. In other words, the agent almost never fails during training, even with exploration noise switched on.

## What the project does

The agent controls one thing: the horizontal push on the cart. It sees four numbers each step:

| Observation | Meaning |
|-------------|---------|
| `x` | cart position |
| `x_dot` | cart velocity |
| `theta` | pole angle (0 = upright) |
| `theta_dot` | pole angular velocity |

It outputs a push between -1 and +1, which is multiplied by 10 N. A game ends when:

- the pole tilts more than about 12 degrees (0.2095 rad), or
- the cart leaves the track (`|x| > 2.4`), or
- 500 steps have passed (time limit).

**Reward.** Each step gives up to 2 points: up to 1 for being near the center of the track and up to 1 for the pole being upright. A step that ends the game gives 0, so the agent learns to avoid failing.

## Project structure

| File | What it does |
|------|--------------|
| `environment.py` | Cart-pole physics, `reset`, `step`, and the reward function |
| `network.py` | Actor-critic neural network (Flax) |
| `rollout.py` | Runs many games in parallel and automatically resets finished ones |
| `train.py` | Loss function and training loop; saves the weights to `trained_params.pkl` |
| `visualize.py` | Plays one episode with the trained network and saves a GIF |
| `run_all.py` | Trains, then opens a live animation window |
| `requirements.txt` | Python dependencies |

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/Temirtas/Jax-Cart-Pole-Simulator-.git
cd Jax-Cart-Pole-Simulator-

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py       # train the agent and save weights to trained_params.pkl
python visualize.py   # play one episode with the saved weights and save a GIF
python run_all.py     # train, then open a live animation window
```

`visualize.py` reuses `trained_params.pkl` if it exists, so you don't need to retrain every time. If you change any training settings, delete `trained_params.pkl` first so the old weights are not reused.

## How it works

### 1. The environment (`environment.py`)
The cart-pole equations of motion are integrated with semi-implicit Euler steps of 0.02 seconds. Every game starts with small random values (about +/-0.05) for all four state variables, so the agent sees situations where it has to recover and not only a perfect start. The `step` function is compiled with `jax.jit`.

### 2. The network (`network.py`)
One shared body with two hidden layers of 64 neurons (tanh activation), then two outputs:

- **Actor:** the average push, squashed to [-1, 1] with `tanh`.
- **Critic:** a single number estimating how many (discounted) points the agent will collect from the current state.

### 3. Parallel rollouts (`rollout.py`)
`jax.vmap` turns the single-game `reset` and `step` functions into versions that run 2,048 games at once, and `jax.lax.scan` runs the 100-step time loop without leaving compiled code. Random noise (standard deviation 0.3) is added to the actor's push so the agent explores. Any game that ends is **reset immediately**, so every environment keeps producing useful experience.

### 4. Learning (`train.py`)
This is a simple advantage actor-critic method:

1. **Returns.** Compute discounted returns (gamma = 0.99) backwards through the 100 collected steps. The return does not flow across an episode reset, and the last step is bootstrapped with the critic's own estimate.
2. **Advantage.** `advantage = return - critic_value`, normalized and detached from the critic's gradient. A positive advantage means the action did better than the critic expected.
3. **Actor loss.** `-mean(log_prob * advantage)` with a Gaussian policy of fixed standard deviation. Actions that beat expectations become more likely.
4. **Critic loss.** Mean squared error between the return and the critic's prediction, weighted by 0.5.
5. **Update.** One Adam step on the combined loss. Rewards are multiplied by `REWARD_SCALE = 0.1` inside the loss so the critic's large numbers don't drown out the actor's gradient.

Repeat for 500 updates.

## Hyperparameters

All of these are at the top of `train.py` and `rollout.py`:

| Name | Value | Meaning |
|------|-------|---------|
| `NUM_ENVS` | 2048 | parallel games |
| `NUM_STEPS` | 100 | steps collected per update |
| `UPDATES` | 500 | number of learning updates |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `REWARD_SCALE` | 0.1 | scales rewards inside the loss |
| `NOISE_STD` | 0.3 | exploration noise (also used in the log-probability) |
| gamma | 0.99 | discount factor |

Physics settings (gravity, masses, force, time step, track limits) are in `EnvParams` in `environment.py`.

## Ideas to experiment with

- Increase the starting noise in `reset` (for example from 0.05 to 0.15) to train a more robust agent.
- Raise `max_steps` in `EnvParams` for longer episodes.
- Add random pushes on the pole during a game to teach the agent to handle disturbances.
- Lower `force_mag` to make the task harder.
- Save and reload weights to continue training instead of starting from scratch.

## Troubleshooting

- **The GIF shows the pole falling:** delete `trained_params.pkl` and the old GIF, then run `python train.py` and `python visualize.py` again. `visualize.py` should print `Episode lasted 500 steps.`
- **No animation window opens:** you may be on a machine without a display. Use `python visualize.py`, which saves a GIF instead.
- **Training is slow:** JAX runs on the CPU by default unless a GPU build is installed. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Tech stack

[JAX](https://github.com/jax-ml/jax) for fast numerical code and parallel simulation, [Flax](https://github.com/google/flax) for the neural network, [Optax](https://github.com/google-deepmind/optax) for the Adam optimizer, and Matplotlib with Pillow for the animation.
