"""
MuJoCo Robot Environment
========================
Implements BaseRobotEnv using MuJoCo physics simulator.
Apple Silicon (M1/M2/M3) compatible!

Robot: Franka Panda (7-DOF arm + 2-finger parallel gripper)
Tasks: Pick, Push, Place colored blocks

Key differences from PyBullet:
- Uses XML (MJCF) format instead of URDF
- Better physics accuracy and performance
- Native macOS ARM support
"""

import numpy as np
import mujoco
from typing import Tuple, Dict, Any, List
import os
from pathlib import Path

from .base import BaseRobotEnv, Observation, Action


class FrankaMuJoCoEnv(BaseRobotEnv):
    """
    Multi-task manipulation environment with Franka Panda using MuJoCo.

    Same interface as FrankaPickAndPlaceEnv but uses MuJoCo instead of PyBullet.
    This demonstrates the power of our simulator-agnostic design!
    """

    def __init__(
        self,
        task_name: str = "pick",
        render_mode: str = "rgb_array",
        image_size: Tuple[int, int] = (224, 224),
        max_steps: int = 100,
    ):
        """
        Args:
            task_name: One of ["pick", "push", "place"]
            render_mode: "rgb_array" for headless, "human" for GUI
            image_size: Camera image resolution (height, width)
            max_steps: Maximum steps per episode
        """
        super().__init__(task_name, render_mode)

        self.image_size = image_size
        self.max_steps = max_steps
        self.current_step = 0

        # Available tasks (same as PyBullet version)
        self._tasks = [
            "pick up the red block",
            "pick up the blue block",
            "pick up the green block",
            "push the red block forward",
            "push the blue block forward",
            "push the green block forward",
            "place the red block on the target",
            "place the blue block on the target",
            "place the green block on the target",
        ]

        # Create MuJoCo scene XML
        self.xml_path = self._create_scene_xml()

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Setup renderer
        self.viewer = None
        if render_mode == "human":
            # For GUI mode, we need to handle viewer separately
            # MuJoCo viewer API varies by version, so we'll use a simple approach
            print("Note: GUI mode will show camera feed only (not interactive 3D viewer)")
            print("      Use headless mode for training")

        # Create offscreen renderer for camera images
        self.renderer = mujoco.Renderer(self.model, height=image_size[0], width=image_size[1])

        # Get important body/geom IDs from model
        self._cache_model_ids()

        # Initial block positions (will be randomized in reset)
        self.initial_block_positions = {
            "red": np.array([0.5, -0.1, 0.02]),
            "blue": np.array([0.5, 0.0, 0.02]),
            "green": np.array([0.5, 0.1, 0.02]),
        }

        # Workspace bounds
        self.workspace_bounds = np.array([
            [0.3, 0.7],   # x
            [-0.2, 0.2],  # y
            [0.0, 0.5],   # z
        ])

    def _create_scene_xml(self) -> str:
        """
        Create MuJoCo scene XML file.

        MuJoCo uses XML (MJCF format) to define robots and scenes.
        This is simpler than we thought - we can define everything here!
        """

        # Create XML for the scene
        # Using a simplified Franka model (we'll use a built-in model)
        xml_content = """
<mujoco model="franka_pick_place">
    <compiler angle="radian" meshdir="assets"/>

    <option timestep="0.002" gravity="0 0 -9.81"/>

    <asset>
        <!-- Materials for colored blocks -->
        <material name="red" rgba="1 0 0 1"/>
        <material name="blue" rgba="0 0 1 1"/>
        <material name="green" rgba="0 1 0 1"/>
        <material name="yellow" rgba="1 1 0 0.5"/>
        <material name="wood" rgba="0.6 0.4 0.2 1"/>
        <material name="metal" rgba="0.7 0.7 0.7 1"/>
    </asset>

    <worldbody>
        <!-- Lighting -->
        <light directional="true" pos="0 0 3" dir="0 0 -1"/>
        <light directional="true" pos="0.5 0.5 2" dir="0 0 -1"/>

        <!-- Table -->
        <body name="table" pos="0.5 0 -0.25">
            <geom type="box" size="0.5 0.5 0.25" material="wood" mass="0"/>
        </body>

        <!-- Simple Franka-like Robot Arm (7-DOF + gripper) -->
        <body name="robot_base" pos="0 0 0">
            <!-- Link 0: Base -->
            <geom type="cylinder" size="0.06 0.05" rgba="0.3 0.3 0.3 1" mass="0"/>

            <!-- Joint 1: Rotation around Z -->
            <body name="link1" pos="0 0 0.05">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973" damping="0.5"/>
                <geom type="cylinder" size="0.05 0.1" rgba="0.9 0.9 0.9 1"/>

                <!-- Joint 2: Rotation around Y -->
                <body name="link2" pos="0 0 0.15">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-1.7628 1.7628" damping="0.5"/>
                    <geom type="box" size="0.04 0.04 0.1" rgba="0.9 0.9 0.9 1"/>

                    <!-- Joint 3: Rotation around Z -->
                    <body name="link3" pos="0 0 0.15">
                        <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973" damping="0.5"/>
                        <geom type="cylinder" size="0.04 0.08" rgba="0.9 0.9 0.9 1"/>

                        <!-- Joint 4: Rotation around Y -->
                        <body name="link4" pos="0 0 0.12">
                            <joint name="joint4" type="hinge" axis="0 1 0" range="-3.0718 -0.0698" damping="0.5"/>
                            <geom type="box" size="0.03 0.03 0.08" rgba="0.9 0.9 0.9 1"/>

                            <!-- Joint 5: Rotation around Z -->
                            <body name="link5" pos="0 0 0.12">
                                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973" damping="0.5"/>
                                <geom type="cylinder" size="0.03 0.06" rgba="0.9 0.9 0.9 1"/>

                                <!-- Joint 6: Rotation around Y -->
                                <body name="link6" pos="0 0 0.1">
                                    <joint name="joint6" type="hinge" axis="0 1 0" range="-0.0175 3.7525" damping="0.5"/>
                                    <geom type="box" size="0.025 0.025 0.05" rgba="0.9 0.9 0.9 1"/>

                                    <!-- Joint 7: Rotation around Z -->
                                    <body name="link7" pos="0 0 0.08">
                                        <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973" damping="0.5"/>
                                        <geom type="cylinder" size="0.02 0.04" rgba="0.9 0.9 0.9 1"/>

                                        <!-- End Effector / Gripper Base -->
                                        <body name="gripper_base" pos="0 0 0.08">
                                            <geom type="box" size="0.02 0.03 0.02" rgba="0.3 0.3 0.3 1"/>

                                            <!-- Camera (for observation) -->
                                            <camera name="overhead_cam" pos="0 0 0.6" quat="1 0 0 0" fovy="60"/>

                                            <!-- Left Gripper Finger -->
                                            <body name="left_finger" pos="0 0.02 0.03">
                                                <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.04" damping="0.1"/>
                                                <geom name="left_finger_geom" type="box" size="0.01 0.005 0.025" rgba="0.2 0.2 0.2 1"/>
                                            </body>

                                            <!-- Right Gripper Finger -->
                                            <body name="right_finger" pos="0 -0.02 0.03">
                                                <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.04" damping="0.1"/>
                                                <geom name="right_finger_geom" type="box" size="0.01 0.005 0.025" rgba="0.2 0.2 0.2 1"/>
                                            </body>

                                            <!-- End effector site (for IK target) -->
                                            <site name="ee_site" pos="0 0 0.05" size="0.01"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <!-- Red Block -->
        <body name="red_block" pos="0.5 -0.1 0.02">
            <joint type="free"/>
            <geom name="red_geom" type="box" size="0.02 0.02 0.02" material="red" mass="0.05"/>
        </body>

        <!-- Blue Block -->
        <body name="blue_block" pos="0.5 0.0 0.02">
            <joint type="free"/>
            <geom name="blue_geom" type="box" size="0.02 0.02 0.02" material="blue" mass="0.05"/>
        </body>

        <!-- Green Block -->
        <body name="green_block" pos="0.5 0.1 0.02">
            <joint type="free"/>
            <geom name="green_geom" type="box" size="0.02 0.02 0.02" material="green" mass="0.05"/>
        </body>

        <!-- Target zone (visual only) -->
        <body name="target" pos="0.5 0.2 0.001">
            <geom type="cylinder" size="0.08 0.001" material="yellow" mass="0" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <actuator>
        <!-- Joint actuators (position control) -->
        <position name="joint1_actuator" joint="joint1" kp="100" ctrlrange="-2.8973 2.8973"/>
        <position name="joint2_actuator" joint="joint2" kp="100" ctrlrange="-1.7628 1.7628"/>
        <position name="joint3_actuator" joint="joint3" kp="100" ctrlrange="-2.8973 2.8973"/>
        <position name="joint4_actuator" joint="joint4" kp="100" ctrlrange="-3.0718 -0.0698"/>
        <position name="joint5_actuator" joint="joint5" kp="100" ctrlrange="-2.8973 2.8973"/>
        <position name="joint6_actuator" joint="joint6" kp="100" ctrlrange="-0.0175 3.7525"/>
        <position name="joint7_actuator" joint="joint7" kp="100" ctrlrange="-2.8973 2.8973"/>

        <!-- Gripper actuators -->
        <position name="left_finger_actuator" joint="left_finger_joint" kp="20" ctrlrange="0 0.04"/>
        <position name="right_finger_actuator" joint="right_finger_joint" kp="20" ctrlrange="0 0.04"/>
    </actuator>
</mujoco>
"""

        # Save to file
        xml_path = Path(__file__).parent / "franka_scene.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_content)

        return str(xml_path)

    def _cache_model_ids(self):
        """Cache important MuJoCo model IDs for fast access."""

        # Joint IDs
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ]
        self.gripper_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_joint"),
        ]

        # Body IDs
        self.block_body_ids = {
            "red": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"),
            "blue": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_block"),
            "green": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "green_block"),
        }

        # End effector site ID
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        # Camera ID
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")

        # Actuator IDs
        self.arm_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}_actuator")
            for i in range(1, 8)
        ]
        self.gripper_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger_actuator"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger_actuator"),
        ]

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Define observation space dimensions."""
        return {
            "image": {
                "shape": (self.image_size[0], self.image_size[1], 3),
                "dtype": np.uint8,
                "range": [0, 255]
            },
            "proprioception": {
                "shape": (9,),  # 7 arm joints + 2 gripper
                "dtype": np.float32,
                "description": "Joint positions"
            },
            "task_instruction": {
                "type": str,
                "options": self._tasks
            }
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        """Define action space dimensions."""
        return {
            "delta_position": {
                "shape": (3,),
                "dtype": np.float32,
                "range": [-0.05, 0.05],
                "description": "End-effector delta (dx, dy, dz)"
            },
            "gripper": {
                "shape": (1,),
                "dtype": np.float32,
                "range": [0.0, 1.0],
                "description": "Gripper closure"
            }
        }

    def reset(self, task_instruction: str = None) -> Observation:
        """Reset environment to initial state."""

        # Select task
        if task_instruction is None:
            task_instruction = np.random.choice(self._tasks)
        self.current_task = task_instruction
        self.current_step = 0

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to neutral pose
        neutral_qpos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, qpos in enumerate(neutral_qpos):
            self.data.qpos[i] = qpos

        # Randomize block positions slightly
        for color, base_pos in self.initial_block_positions.items():
            noise = np.random.uniform(-0.03, 0.03, size=2)
            new_pos = base_pos + np.array([noise[0], noise[1], 0])

            # Get body ID and set position
            body_id = self.block_body_ids[color]
            # In MuJoCo, free joint has 7 qpos values: 3 pos + 4 quat
            # Find the qpos index for this body
            body_jntadr = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[body_jntadr]

            self.data.qpos[qpos_adr:qpos_adr+3] = new_pos
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  # Identity quaternion

        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

        # Let physics settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        return self._get_observation()

    def _get_observation(self) -> Observation:
        """Capture current observation from simulator."""

        # 1. Render camera image
        image = self.render()

        # 2. Get proprioception (joint positions)
        # First 7 joints are arm, next 2 are gripper
        joint_positions = self.data.qpos[:9].copy()

        return Observation(
            image=image,
            proprioception=joint_positions.astype(np.float32),
            task_instruction=self.current_task
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Execute action using end-effector control."""

        # Get current end-effector position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        # Compute target position
        target_pos = ee_pos + action.delta_position
        target_pos = np.clip(
            target_pos,
            self.workspace_bounds[:, 0],
            self.workspace_bounds[:, 1]
        )

        # Simple IK: Set target and let actuators track
        # In a real system, you'd use mujoco.mj_jac or analytical IK
        # For simplicity, we'll use a proportional controller

        # Get current joint positions
        current_qpos = self.data.qpos[:7].copy()

        # Compute Jacobian for end-effector
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Use damped least squares IK
        delta_pos = target_pos - ee_pos
        J = jacp[:, :7]  # Only arm joints
        lambda_damping = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(3))
        delta_qpos = J_pinv @ delta_pos

        # Apply joint commands
        target_qpos = current_qpos + delta_qpos * 0.1  # Scale for stability
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            self.data.ctrl[actuator_id] = target_qpos[i]

        # Apply gripper control
        gripper_openness = 0.04 * (1 - action.gripper)
        self.data.ctrl[self.gripper_actuator_ids[0]] = gripper_openness
        self.data.ctrl[self.gripper_actuator_ids[1]] = gripper_openness

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Get new observation
        obs = self._get_observation()

        # Compute reward
        reward, done, info = self._compute_reward()

        # Episode timeout
        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, info

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """Task-specific reward computation (same logic as PyBullet)."""

        info = {"success": False}
        reward = 0.0
        done = False

        # Parse task
        task_lower = self.current_task.lower()

        # Extract color
        target_color = None
        for color in ["red", "blue", "green"]:
            if color in task_lower:
                target_color = color
                break

        if target_color is None:
            return 0.0, False, info

        # Get block position
        body_id = self.block_body_ids[target_color]
        block_pos = self.data.xpos[body_id].copy()

        # Get gripper position
        gripper_pos = self.data.site_xpos[self.ee_site_id].copy()

        # Task-specific success conditions
        if "pick" in task_lower:
            if block_pos[2] > 0.1:
                reward = 1.0
                done = True
                info["success"] = True
            else:
                distance = np.linalg.norm(gripper_pos - block_pos)
                reward = -distance

        elif "push" in task_lower:
            initial_y = self.initial_block_positions[target_color][1]
            if block_pos[1] > initial_y + 0.1:
                reward = 1.0
                done = True
                info["success"] = True
            else:
                reward = block_pos[1] - initial_y

        elif "place" in task_lower:
            target_pos = np.array([0.5, 0.2, 0.02])
            distance_to_target = np.linalg.norm(block_pos - target_pos)
            if distance_to_target < 0.05 and block_pos[2] < 0.05:
                reward = 1.0
                done = True
                info["success"] = True
            else:
                reward = -distance_to_target

        return reward, done, info

    def render(self) -> np.ndarray:
        """Render camera view as RGB image."""

        # Update renderer
        self.renderer.update_scene(self.data, camera=self.camera_id)

        # Render and get pixels
        pixels = self.renderer.render()

        # MuJoCo returns (H, W, 3) RGB array in [0, 255]
        return pixels

    def close(self):
        """Clean up resources."""
        # Renderer cleanup is automatic with Python GC
        pass
