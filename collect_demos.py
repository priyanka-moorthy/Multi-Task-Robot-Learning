"""
Demo Collection Script
======================
Collects expert demonstrations for training the VLA model.

Usage:
    python collect_demos.py --num_demos 10 --output data/demos.pkl
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from pathlib import Path

# Import our modules
from environments import FrankaMuJoCoEnv
from data.expert_policy import ScriptedExpertPolicy
from data.demo_collector import DemoCollector
from data.dataset import RobotDemoDataset, create_dataloader


def main():
    parser = argparse.ArgumentParser(description="Collect robot demonstrations")
    parser.add_argument(
        "--num_demos",
        type=int,
        default=5,
        help="Number of demos per task"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/demos.pkl",
        help="Output path for demos"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum steps per demo"
    )
    parser.add_argument(
        "--test_dataset",
        action="store_true",
        help="Test loading dataset after collection"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Demo Collection for VLA Training")
    print("=" * 60)

    # Create environment
    print("\n[1/3] Creating environment...")
    env = FrankaMuJoCoEnv(
        task_name="pick",
        render_mode="rgb_array",
        image_size=(224, 224),
        max_steps=args.max_steps,
    )
    print(f"  Available tasks: {len(env.available_tasks)}")

    # Create expert policy
    print("\n[2/3] Creating expert policy...")
    expert = ScriptedExpertPolicy(
        max_action=0.03,
        grasp_height=0.02,
        lift_height=0.15,
    )

    # Collect demos
    print("\n[3/3] Collecting demonstrations...")
    collector = DemoCollector(
        env=env,
        expert_policy=expert,
        max_steps_per_demo=args.max_steps,
    )

    # Use subset of tasks for faster collection
    tasks = [
        "pick up the red block",
        "pick up the blue block",
        "push the red block forward",
        "push the blue block forward",
        "place the red block on the target",
    ]

    dataset = collector.collect(
        num_demos_per_task=args.num_demos,
        tasks=tasks,
        show_progress=True,
    )

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(output_path))

    # Print stats
    stats = dataset.get_stats()
    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    print(f"  Output: {output_path}")
    print(f"  Total demos: {stats['total_demos']}")
    print(f"  Total transitions: {stats['total_transitions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")

    # Test loading as PyTorch dataset
    if args.test_dataset:
        print("\n[Testing Dataset Loading]")
        torch_dataset = RobotDemoDataset.from_file(str(output_path))
        dataloader = create_dataloader(torch_dataset, batch_size=4)

        batch = next(iter(dataloader))
        print(f"  Batch images: {batch['image'].shape}")
        print(f"  Batch tasks: {batch['task'][:2]}")
        print(f"  Batch actions: {batch['action']['delta_position'].shape}")
        print("  ✓ Dataset loading successful!")

    env.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
