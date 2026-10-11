import matplotlib.pyplot as plt

from train import train
from visualize import run_single_episode, make_animation


def main():
    print("==================================================")
    print("      JAX TRAIN + LIVE VISUALIZATION PIPELINE     ")
    print("==================================================")

    # Phase 1: train (uses the settings in train.py and saves the weights)
    print("\n[Phase 1] Training neural network...")
    params_network = train()

    # Phase 2: run one test episode and show it live
    print("\n[Phase 2] Launching live interactive visualizer...")
    xs, thetas = run_single_episode(params_network, max_steps=500)
    print(f"Opening live animation window for {len(xs)} steps...")

    fig, ani = make_animation(
        xs, thetas, title="Trained JAX Cart-Pole Robot (Live Animation)", interval=20
    )
    plt.show()


if __name__ == "__main__":
    main()