"""
Expert Policy for Demonstration Collection
==========================================
A scripted policy that knows how to perform manipulation tasks.

This is used to collect training data for behavior cloning.
In a real scenario, you might use:
- Human teleoperation
- Motion planning algorithms
- Pre-trained RL policies

For this tutorial, we use a simple state-based policy that
cheats by reading the simulator state directly.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExpertAction:
    """Action from expert policy."""
    delta_position: np.ndarray  # (3,) end-effector delta
    gripper: float              # 0=open, 1=closed


class ScriptedExpertPolicy:
    """
    Scripted expert that performs pick/push/place tasks.

    This policy has access to ground truth object positions
    (cheating!) to generate optimal actions.

    In real robotics, you'd use:
    - Human demonstrations via teleoperation
    - Motion planning (RRT, trajectory optimization)
    - Reinforcement learning
    """

    def __init__(
        self,
        max_action: float = 0.03,
        grasp_height: float = 0.02,
        lift_height: float = 0.15,
        approach_distance: float = 0.05,
    ):
        """
        Args:
            max_action: Maximum action magnitude per step
            grasp_height: Height for grasping objects
            lift_height: Height to lift objects
            approach_distance: Distance threshold for "close enough"
        """
        self.max_action = max_action
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self.approach_distance = approach_distance

        # State machine for multi-step tasks
        self.state = "approach"
        self.target_color = None

    def reset(self, task: str):
        """Reset policy state for new episode."""
        self.state = "approach"
        self.target_color = self._parse_color(task)

    def _parse_color(self, task: str) -> str:
        """Extract target color from task instruction."""
        task_lower = task.lower()
        for color in ["red", "blue", "green"]:
            if color in task_lower:
                return color
        return "red"  # Default

    def get_action(
        self,
        gripper_pos: np.ndarray,
        block_positions: Dict[str, np.ndarray],
        target_position: Optional[np.ndarray] = None,
        task: str = "pick",
    ) -> ExpertAction:
        """
        Compute expert action based on current state.

        Args:
            gripper_pos: Current gripper position (3,)
            block_positions: Dict mapping color → position (3,)
            target_position: Target zone position for "place" task
            task: Task type ("pick", "push", "place")

        Returns:
            ExpertAction with delta_position and gripper command
        """
        # Get target block position
        block_pos = block_positions.get(self.target_color, np.array([0.5, 0, 0.02]))

        # Determine task type
        if "push" in task.lower():
            return self._push_action(gripper_pos, block_pos)
        elif "place" in task.lower():
            return self._pick_and_place_action(gripper_pos, block_pos, target_position)
        else:  # pick
            return self._pick_action(gripper_pos, block_pos)

    def _pick_action(
        self,
        gripper_pos: np.ndarray,
        block_pos: np.ndarray,
    ) -> ExpertAction:
        """
        Pick up action sequence:
        1. Approach (move above block)
        2. Descend (lower to grasp height)
        3. Grasp (close gripper)
        4. Lift (raise block)
        """
        # Position above block for approach
        approach_pos = block_pos.copy()
        approach_pos[2] = self.lift_height

        # Position for grasping
        grasp_pos = block_pos.copy()
        grasp_pos[2] = self.grasp_height

        if self.state == "approach":
            # Move above the block
            delta = self._move_towards(gripper_pos, approach_pos)
            if np.linalg.norm(gripper_pos[:2] - block_pos[:2]) < self.approach_distance:
                self.state = "descend"
            return ExpertAction(delta_position=delta, gripper=0.0)  # Open gripper

        elif self.state == "descend":
            # Lower to grasp height
            delta = self._move_towards(gripper_pos, grasp_pos)
            if gripper_pos[2] < self.grasp_height + 0.02:
                self.state = "grasp"
            return ExpertAction(delta_position=delta, gripper=0.0)

        elif self.state == "grasp":
            # Close gripper and prepare to lift
            self.state = "lift"
            return ExpertAction(delta_position=np.zeros(3), gripper=1.0)  # Close gripper

        elif self.state == "lift":
            # Lift the block
            lift_target = gripper_pos.copy()
            lift_target[2] = self.lift_height
            delta = self._move_towards(gripper_pos, lift_target)
            return ExpertAction(delta_position=delta, gripper=1.0)  # Keep closed

        # Default: stay still
        return ExpertAction(delta_position=np.zeros(3), gripper=1.0)

    def _push_action(
        self,
        gripper_pos: np.ndarray,
        block_pos: np.ndarray,
    ) -> ExpertAction:
        """
        Push action sequence:
        1. Move behind block (relative to push direction)
        2. Lower to block level
        3. Push forward
        """
        # Push direction is forward (+y)
        push_direction = np.array([0, 1, 0])

        # Position behind block for pushing
        behind_pos = block_pos.copy()
        behind_pos[1] -= 0.08  # Behind in y direction
        behind_pos[2] = block_pos[2] + 0.01  # Slightly above block

        if self.state == "approach":
            # Move behind the block
            delta = self._move_towards(gripper_pos, behind_pos)
            if np.linalg.norm(gripper_pos - behind_pos) < self.approach_distance:
                self.state = "push"
            return ExpertAction(delta_position=delta, gripper=1.0)  # Closed gripper

        elif self.state == "push":
            # Push forward
            delta = push_direction * self.max_action
            return ExpertAction(delta_position=delta, gripper=1.0)

        return ExpertAction(delta_position=np.zeros(3), gripper=1.0)

    def _pick_and_place_action(
        self,
        gripper_pos: np.ndarray,
        block_pos: np.ndarray,
        target_pos: Optional[np.ndarray],
    ) -> ExpertAction:
        """
        Pick and place action sequence:
        1. Pick up (same as pick)
        2. Move to target
        3. Lower and release
        """
        if target_pos is None:
            target_pos = np.array([0.5, 0.2, 0.02])

        # First, complete the pick sequence
        if self.state in ["approach", "descend", "grasp", "lift"]:
            action = self._pick_action(gripper_pos, block_pos)
            # Check if we've lifted high enough
            if self.state == "lift" and gripper_pos[2] > self.lift_height - 0.02:
                self.state = "move_to_target"
            return action

        elif self.state == "move_to_target":
            # Move to above target
            target_above = target_pos.copy()
            target_above[2] = self.lift_height
            delta = self._move_towards(gripper_pos, target_above)
            if np.linalg.norm(gripper_pos[:2] - target_pos[:2]) < self.approach_distance:
                self.state = "lower"
            return ExpertAction(delta_position=delta, gripper=1.0)

        elif self.state == "lower":
            # Lower to target
            target_low = target_pos.copy()
            target_low[2] = self.grasp_height + 0.02
            delta = self._move_towards(gripper_pos, target_low)
            if gripper_pos[2] < self.grasp_height + 0.04:
                self.state = "release"
            return ExpertAction(delta_position=delta, gripper=1.0)

        elif self.state == "release":
            # Open gripper to release
            return ExpertAction(delta_position=np.zeros(3), gripper=0.0)

        return ExpertAction(delta_position=np.zeros(3), gripper=0.0)

    def _move_towards(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute delta to move towards target, clamped to max_action.
        """
        delta = target - current
        distance = np.linalg.norm(delta)

        if distance > self.max_action:
            delta = delta / distance * self.max_action

        return delta


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing Expert Policy")
    print("=" * 40)

    expert = ScriptedExpertPolicy()

    # Simulate a pick task
    expert.reset("pick up the red block")

    gripper_pos = np.array([0.3, 0.0, 0.3])
    block_positions = {
        "red": np.array([0.5, 0.0, 0.02]),
        "blue": np.array([0.5, 0.1, 0.02]),
        "green": np.array([0.5, -0.1, 0.02]),
    }

    print("Simulating pick task:")
    for step in range(10):
        action = expert.get_action(gripper_pos, block_positions, task="pick up the red block")
        gripper_pos = gripper_pos + action.delta_position
        print(f"  Step {step}: state={expert.state}, pos={gripper_pos}, gripper={action.gripper}")

    print("\n✓ Expert policy working!")
