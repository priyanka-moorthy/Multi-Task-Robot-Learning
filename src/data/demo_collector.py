"""
Demonstration Collector
=======================
Collects expert demonstrations from the robot environment.

These demonstrations are used for behavior cloning:
- Expert performs task
- We record (observation, action) pairs
- Model learns to mimic expert actions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from tqdm import tqdm

try:
    from ..environments.base import Observation, Action
except ImportError:
    # When running as script
    from environments.base import Observation, Action


@dataclass
class Demonstration:
    """
    A single demonstration trajectory.

    Contains a sequence of (observation, action) pairs
    from expert performing a task.
    """
    task: str                         # Task instruction
    observations: List[np.ndarray]    # Images (T, H, W, 3)
    actions: List[Dict]               # Actions [{delta_pos, gripper}, ...]
    proprioceptions: List[np.ndarray] # Joint states (T, num_joints)
    rewards: List[float]              # Rewards per step
    success: bool                     # Whether task was completed

    def __len__(self):
        return len(self.observations)


@dataclass
class DemoDataset:
    """
    Collection of demonstrations for multiple tasks.
    """
    demos: List[Demonstration] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add(self, demo: Demonstration):
        """Add a demonstration to the dataset."""
        self.demos.append(demo)

    def __len__(self):
        return len(self.demos)

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.demos:
            return {}

        tasks = {}
        for demo in self.demos:
            task_type = demo.task.split()[0]  # "pick", "push", "place"
            if task_type not in tasks:
                tasks[task_type] = {"count": 0, "success": 0, "total_steps": 0}
            tasks[task_type]["count"] += 1
            tasks[task_type]["success"] += int(demo.success)
            tasks[task_type]["total_steps"] += len(demo)

        return {
            "total_demos": len(self.demos),
            "total_transitions": sum(len(d) for d in self.demos),
            "tasks": tasks,
            "success_rate": sum(d.success for d in self.demos) / len(self.demos),
        }

    def save(self, path: str):
        """Save dataset to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved {len(self)} demos to {path}")

    @classmethod
    def load(cls, path: str) -> 'DemoDataset':
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class DemoCollector:
    """
    Collects demonstrations using an expert policy.

    Usage:
        collector = DemoCollector(env, expert)
        dataset = collector.collect(num_demos=100)
        dataset.save("demos.pkl")
    """

    def __init__(
        self,
        env,
        expert_policy,
        max_steps_per_demo: int = 50,
    ):
        """
        Args:
            env: Robot environment
            expert_policy: Expert policy for generating actions
            max_steps_per_demo: Maximum steps per demonstration
        """
        self.env = env
        self.expert = expert_policy
        self.max_steps = max_steps_per_demo

    def collect_one(self, task: str) -> Demonstration:
        """
        Collect a single demonstration for a task.

        Args:
            task: Task instruction string

        Returns:
            Demonstration object with trajectory data
        """
        # Reset environment and expert
        obs = self.env.reset(task_instruction=task)
        self.expert.reset(task)

        # Storage for trajectory
        observations = [obs.image.copy()]
        proprioceptions = [obs.proprioception.copy()]
        actions = []
        rewards = []

        # Collect trajectory
        done = False
        success = False

        for step in range(self.max_steps):
            # Get expert action (needs simulator state - we'll extract it)
            expert_action = self._get_expert_action(obs, task)

            # Convert to environment Action format
            env_action = Action(
                delta_position=expert_action.delta_position,
                gripper=expert_action.gripper,
            )

            # Execute action
            next_obs, reward, done, info = self.env.step(env_action)

            # Store transition
            actions.append({
                'delta_position': expert_action.delta_position.copy(),
                'gripper': expert_action.gripper,
            })
            rewards.append(reward)

            if not done:
                observations.append(next_obs.image.copy())
                proprioceptions.append(next_obs.proprioception.copy())

            obs = next_obs
            success = info.get('success', False)

            if done:
                break

        return Demonstration(
            task=task,
            observations=observations,
            actions=actions,
            proprioceptions=proprioceptions,
            rewards=rewards,
            success=success,
        )

    def _get_expert_action(self, obs: Observation, task: str):
        """
        Get expert action from current observation.

        Note: This is a simplified version that extracts state from env.
        In real scenarios, you'd use perception or teleoperation.
        """
        # Extract gripper position from proprioception or env state
        # For MuJoCo env, we need to get positions from the data

        # Get gripper position (end effector)
        try:
            gripper_pos = self.env.data.site_xpos[self.env.ee_site_id].copy()
        except AttributeError:
            # Fallback: estimate from proprioception
            gripper_pos = np.array([0.4, 0.0, 0.2])

        # Get block positions
        block_positions = {}
        for color in ["red", "blue", "green"]:
            try:
                body_id = self.env.block_body_ids[color]
                block_positions[color] = self.env.data.xpos[body_id].copy()
            except (AttributeError, KeyError):
                # Fallback
                block_positions[color] = np.array([0.5, 0.0, 0.02])

        # Get target position
        target_pos = np.array([0.5, 0.2, 0.02])

        return self.expert.get_action(
            gripper_pos=gripper_pos,
            block_positions=block_positions,
            target_position=target_pos,
            task=task,
        )

    def collect(
        self,
        num_demos_per_task: int = 10,
        tasks: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> DemoDataset:
        """
        Collect demonstrations for multiple tasks.

        Args:
            num_demos_per_task: Number of demos per task
            tasks: List of tasks (or use env's default tasks)
            show_progress: Show progress bar

        Returns:
            DemoDataset with all collected demos
        """
        if tasks is None:
            tasks = self.env.available_tasks

        dataset = DemoDataset(metadata={
            "num_demos_per_task": num_demos_per_task,
            "max_steps": self.max_steps,
            "tasks": tasks,
        })

        total_demos = len(tasks) * num_demos_per_task

        if show_progress:
            pbar = tqdm(total=total_demos, desc="Collecting demos")

        for task in tasks:
            for _ in range(num_demos_per_task):
                demo = self.collect_one(task)
                dataset.add(demo)

                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix({"success": f"{dataset.get_stats()['success_rate']:.1%}"})

        if show_progress:
            pbar.close()

        # Print summary
        stats = dataset.get_stats()
        print("\nCollection Summary:")
        print(f"  Total demos: {stats['total_demos']}")
        print(f"  Total transitions: {stats['total_transitions']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")

        return dataset


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing Demo Collector")
    print("=" * 40)

    # This would require the environment to be set up
    print("Note: Full test requires MuJoCo environment")
    print("Run 'python -m src.data.collect_demos' for full collection")
