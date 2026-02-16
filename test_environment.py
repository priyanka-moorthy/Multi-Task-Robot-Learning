"""
Test PyBullet Environment
=========================
Run this to see the robot environment in action!

This demonstrates:
1. Environment reset
2. Random actions
3. Image observations
4. Task-specific rewards
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from environments import Action

# Auto-detect available simulator
try:
    from environments import PYBULLET_AVAILABLE, MUJOCO_AVAILABLE
except ImportError:
    PYBULLET_AVAILABLE = False
    MUJOCO_AVAILABLE = False

# Import the appropriate environment
if MUJOCO_AVAILABLE:
    from environments import FrankaMuJoCoEnv as RobotEnv
    SIMULATOR = "MuJoCo"
elif PYBULLET_AVAILABLE:
    from environments import FrankaPickAndPlaceEnv as RobotEnv
    SIMULATOR = "PyBullet"
else:
    raise ImportError("No simulator available! Install either 'mujoco' or 'pybullet'")


def test_environment_visual():
    """
    Test environment with GUI visualization.
    Watch the robot perform random actions with live matplotlib display!
    """
    print("=" * 60)
    print(f"Testing {SIMULATOR} Environment - GUI Mode")
    print("=" * 60)

    # Create environment (headless rendering, we'll show with matplotlib)
    env = RobotEnv(
        task_name="pick",
        render_mode="rgb_array",
        max_steps=50
    )

    print(f"\n✓ Environment created")
    print(f"  Available tasks: {len(env.available_tasks)}")
    print(f"  Image size: {env.image_size}")
    print(f"  Action space: {env.action_space}")

    # Setup matplotlib for live visualization
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Test 3 different tasks
    tasks_to_test = [
        "pick up the red block",
        "push the blue block forward",
        "place the green block on the target"
    ]

    for task in tasks_to_test:
        print(f"\n{'='*60}")
        print(f"Task: '{task}'")
        print(f"{'='*60}")

        obs = env.reset(task_instruction=task)
        print(f"✓ Environment reset")
        print(f"  Image shape: {obs.image.shape}")
        print(f"  Proprioception shape: {obs.proprioception.shape}")
        print(f"  Task: {obs.task_instruction}")

        # Run episode with random actions
        total_reward = 0
        for step in range(20):
            # Random action
            action = Action(
                delta_position=np.random.uniform(-0.02, 0.02, size=3),  # Small movements
                gripper=np.random.choice([0.0, 1.0])  # Randomly open/close
            )

            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Update visualization every step
            ax.clear()
            ax.imshow(obs.image)
            ax.set_title(f"{task}\nStep: {step}, Reward: {reward:+.3f}", fontsize=12)
            ax.axis('off')
            plt.pause(0.1)  # Small delay to see animation

            if step % 5 == 0:
                print(f"  Step {step:2d}: reward={reward:+.3f}, done={done}, success={info['success']}")

            if done:
                print(f"  ✓ Episode finished at step {step}")
                plt.pause(1.0)  # Pause longer on completion
                break

        print(f"  Total reward: {total_reward:.3f}")

    plt.ioff()
    plt.close()
    env.close()
    print("\n✓ Test completed!\n")


def test_environment_headless():
    """
    Test environment in headless mode and visualize observations.
    This is what training will use.
    """
    print("=" * 60)
    print(f"Testing {SIMULATOR} Environment - Headless Mode")
    print("=" * 60)

    # Create environment without GUI (faster)
    env = RobotEnv(
        task_name="pick",
        render_mode="rgb_array",  # No GUI, just images
        max_steps=50
    )

    # Reset with random task
    obs = env.reset()
    print(f"\n✓ Environment created in headless mode")
    print(f"  Task: {obs.task_instruction}")

    # Collect observations from a few steps
    observations = []
    for _ in range(5):
        action = Action(
            delta_position=np.random.uniform(-0.01, 0.01, size=3),
            gripper=0.0
        )
        obs, _, _, _ = env.step(action)
        observations.append(obs.image)

    # Visualize collected images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, img in enumerate(observations):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Step {i}")

    plt.suptitle(f"Task: {obs.task_instruction}")
    plt.tight_layout()

    # Save figure
    output_path = "test_observations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved observations to: {output_path}")

    env.close()
    print("✓ Test completed!\n")


if __name__ == "__main__":
    import argparse

    print(f"\n🤖 Using {SIMULATOR} simulator\n")

    parser = argparse.ArgumentParser(description=f"Test {SIMULATOR} robot environment")
    parser.add_argument(
        "--mode",
        choices=["gui", "headless"],
        default="headless",
        help="Visualization mode"
    )

    args = parser.parse_args()

    if args.mode == "gui":
        test_environment_visual()
    else:
        test_environment_headless()
