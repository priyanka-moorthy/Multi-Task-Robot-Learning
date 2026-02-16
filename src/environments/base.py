"""
Base Environment Interface
==========================
Abstract base class that defines the contract ALL simulators must follow.
This allows us to swap PyBullet ↔ Isaac Sim without changing model code.

Key Concept: Dependency Inversion Principle
- High-level modules (VLA model) don't depend on low-level modules (simulators)
- Both depend on abstractions (this interface)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np


@dataclass
class Observation:
    """
    Standardized observation from any simulator.

    Attributes:
        image: RGB image from camera (H, W, 3), values in [0, 255]
        proprioception: Robot joint positions/velocities (optional)
        task_instruction: Natural language task description
    """
    image: np.ndarray          # Shape: (H, W, 3)
    proprioception: np.ndarray # Shape: (num_joints,) - joint angles, gripper state
    task_instruction: str      # e.g., "Pick up the red block"


@dataclass
class Action:
    """
    Standardized action for robot control.

    We use end-effector control (easier to learn than joint control):
    - delta_position: Move gripper by (dx, dy, dz) in workspace
    - delta_rotation: Rotate gripper (simplified to single axis for now)
    - gripper: 0.0 = open, 1.0 = closed
    """
    delta_position: np.ndarray  # Shape: (3,) - dx, dy, dz
    gripper: float              # 0.0 to 1.0


class BaseRobotEnv(ABC):
    """
    Abstract base class for robot manipulation environments.

    Any simulator (PyBullet, Isaac Sim, MuJoCo) must implement these methods.
    This ensures our VLA model works with ANY simulator.
    """

    def __init__(self, task_name: str, render_mode: str = "rgb_array"):
        """
        Args:
            task_name: Name of the manipulation task
            render_mode: "rgb_array" for training, "human" for visualization
        """
        self.task_name = task_name
        self.render_mode = render_mode
        self._tasks: List[str] = []  # Available tasks

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """Define the observation space dimensions."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """Define the action space dimensions."""
        pass

    @abstractmethod
    def reset(self, task_instruction: str = None) -> Observation:
        """
        Reset environment to initial state.

        Args:
            task_instruction: Optional task to set. If None, randomly sample.

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute action and return results.

        Args:
            action: Action to execute

        Returns:
            observation: New observation after action
            reward: Reward signal (for RL training)
            done: Whether episode is complete
            info: Additional information (success, metrics, etc.)
        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render current state as RGB image."""
        pass

    @abstractmethod
    def close(self):
        """Clean up simulator resources."""
        pass

    @property
    def available_tasks(self) -> List[str]:
        """List of tasks this environment supports."""
        return self._tasks

    def get_task_instruction(self) -> str:
        """Get current task as natural language instruction."""
        return self.task_name


# =============================================================================
# ISAAC SIM PATTERN (for future reference)
# =============================================================================
#
# In Isaac Sim, you would implement BaseRobotEnv like this:
#
# from omni.isaac.core import World
# from omni.isaac.core.robots import Robot
# from omni.isaac.core.utils.stage import add_reference_to_stage
#
# class IsaacSimEnv(BaseRobotEnv):
#     def __init__(self, task_name: str, ...):
#         super().__init__(task_name)
#
#         # Isaac Sim specific setup
#         self.world = World(stage_units_in_meters=1.0)
#         self.world.scene.add_default_ground_plane()
#
#         # Load robot USD (Universal Scene Description)
#         add_reference_to_stage(
#             usd_path="/path/to/franka.usd",
#             prim_path="/World/Franka"
#         )
#         self.robot = self.world.scene.add(
#             Robot(prim_path="/World/Franka", name="franka")
#         )
#
#         # Isaac Sim uses USD for scenes, PhysX for physics
#         self.world.reset()
#
#     def step(self, action: Action):
#         # Convert our Action to Isaac Sim format
#         # Isaac Sim uses ArticulationAction for joint control
#         self.robot.apply_action(...)
#         self.world.step(render=True)
#         return self._get_observation(), reward, done, info
#
# The key insight: Same interface, different implementation!
# =============================================================================
