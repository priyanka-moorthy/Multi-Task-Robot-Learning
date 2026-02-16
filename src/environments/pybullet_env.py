"""
PyBullet Robot Environment
==========================
Implements BaseRobotEnv using PyBullet physics simulator.

Robot: Franka Panda (7-DOF arm + 2-finger parallel gripper)
Tasks: Pick, Push, Place colored blocks
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Tuple, Dict, Any, List
import time

from .base import BaseRobotEnv, Observation, Action


class FrankaPickAndPlaceEnv(BaseRobotEnv):
    """
    Multi-task manipulation environment with Franka Panda robot.

    Workspace:
    - Table: 1m x 1m at z=0
    - Objects: Colored blocks (red, blue, green)
    - Camera: Fixed overhead view

    Tasks:
    1. "pick up the [color] block" - Grasp and lift object
    2. "push the [color] block forward" - Push without grasping
    3. "place the [color] block on the target" - Pick and place
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

        # Available tasks
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

        # Connect to PyBullet
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)  # Headless mode

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # IDs for objects in simulation (will be set in _load_scene)
        self.robot_id = None
        self.table_id = None
        self.blocks = {}  # {color: block_id}
        self.target_id = None

        # Robot parameters
        self.num_joints = 9  # 7 arm + 2 gripper
        self.end_effector_index = 11  # Link index for gripper center

        # Workspace bounds (Franka's reachable area)
        self.workspace_bounds = np.array([
            [0.3, 0.7],   # x: 30cm to 70cm in front
            [-0.2, 0.2],  # y: ±20cm left/right
            [0.0, 0.5],   # z: table to 50cm above
        ])

        # Load the scene
        self._load_scene()

    def _load_scene(self):
        """Load robot, table, objects, and camera setup."""

        # 1. Load table (simple box)
        table_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.25]  # 1m x 1m x 0.5m table
        )
        table_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.25],
            rgbaColor=[0.6, 0.4, 0.2, 1]  # Brown wood color
        )
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=table_collision_shape,
            baseVisualShapeIndex=table_visual_shape,
            basePosition=[0.5, 0, -0.25]  # Center at (0.5, 0), top at z=0
        )

        # 2. Load Franka Panda robot
        # PyBullet includes Franka URDF in pybullet_data
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True  # Robot base is fixed to ground
        )

        # Set robot to neutral position (arm upright)
        self.reset_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, self.reset_joint_positions[i])

        # Enable joint force/torque sensors for proprioception
        for i in range(self.num_joints):
            p.enableJointForceTorqueSensor(self.robot_id, i, enableSensor=True)

        # 3. Create colored blocks
        colors = {
            "red": [1, 0, 0, 1],
            "blue": [0, 0, 1, 1],
            "green": [0, 1, 0, 1],
        }

        block_size = 0.04  # 4cm cubes
        for i, (color_name, rgba) in enumerate(colors.items()):
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[block_size/2] * 3
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[block_size/2] * 3,
                rgbaColor=rgba
            )

            # Place blocks in a row on table
            block_id = p.createMultiBody(
                baseMass=0.05,  # 50g blocks
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0.5 + i*0.1, 0, block_size/2]  # On table surface
            )
            self.blocks[color_name] = block_id

        # 4. Create target zone (for "place" task)
        target_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.08,
            length=0.001,
            rgbaColor=[1, 1, 0, 0.5]  # Yellow, semi-transparent
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_shape,
            basePosition=[0.5, 0.2, 0.001]  # Side of table
        )

        # 5. Setup camera (overhead view)
        # Camera looks down at workspace
        self.camera_params = {
            "cameraEyePosition": [0.5, 0, 0.8],      # Above workspace
            "cameraTargetPosition": [0.5, 0, 0],     # Looking at table center
            "cameraUpVector": [0, 1, 0],             # Y-axis is up in view
            "width": self.image_size[1],
            "height": self.image_size[0],
            "fov": 60,                                # Field of view
            "nearVal": 0.1,
            "farVal": 2.0,
        }

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_params["cameraEyePosition"],
            cameraTargetPosition=self.camera_params["cameraTargetPosition"],
            cameraUpVector=self.camera_params["cameraUpVector"]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_params["fov"],
            aspect=self.image_size[1] / self.image_size[0],
            nearVal=self.camera_params["nearVal"],
            farVal=self.camera_params["farVal"]
        )

        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix

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
                "shape": (self.num_joints,),
                "dtype": np.float32,
                "description": "Joint angles"
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
                "range": [-0.05, 0.05],  # Max 5cm movement per step
                "description": "End-effector delta (dx, dy, dz)"
            },
            "gripper": {
                "shape": (1,),
                "dtype": np.float32,
                "range": [0.0, 1.0],  # 0=open, 1=closed
                "description": "Gripper closure"
            }
        }

    def reset(self, task_instruction: str = None) -> Observation:
        """
        Reset environment to initial state.

        Args:
            task_instruction: Specific task, or None for random

        Returns:
            Initial observation
        """
        # Select task
        if task_instruction is None:
            task_instruction = np.random.choice(self._tasks)
        self.current_task = task_instruction
        self.current_step = 0

        # Reset robot to neutral position
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, self.reset_joint_positions[i])

        # Randomize block positions slightly (to avoid overfitting)
        for i, color in enumerate(["red", "blue", "green"]):
            noise = np.random.uniform(-0.03, 0.03, size=2)  # ±3cm
            position = [0.5 + i*0.1 + noise[0], noise[1], 0.02]
            p.resetBasePositionAndOrientation(
                self.blocks[color],
                position,
                p.getQuaternionFromEuler([0, 0, 0])
            )

        # Let physics settle
        for _ in range(50):
            p.stepSimulation()

        return self._get_observation()

    def _get_observation(self) -> Observation:
        """Capture current observation from simulator."""

        # 1. Get camera image
        image = self.render()

        # 2. Get proprioception (joint positions)
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)

        return Observation(
            image=image,
            proprioception=joint_positions,
            task_instruction=self.current_task
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute action using inverse kinematics for end-effector control.

        This is the key control method:
        1. Get current end-effector position
        2. Add action deltas to get target position
        3. Use IK to compute joint angles
        4. Apply joint control to reach target
        """

        # Get current end-effector state
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        current_position = np.array(ee_state[0])
        current_orientation = ee_state[1]

        # Compute target position (current + delta)
        target_position = current_position + action.delta_position

        # Clip to workspace bounds for safety
        target_position = np.clip(
            target_position,
            self.workspace_bounds[:, 0],
            self.workspace_bounds[:, 1]
        )

        # Inverse Kinematics: position → joint angles
        # PyBullet's IK solver computes joint angles to reach target
        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=target_position.tolist(),
            targetOrientation=current_orientation,  # Keep orientation fixed
            maxNumIterations=100,
            residualThreshold=1e-5
        )

        # Apply joint control to arm (first 7 joints)
        for i in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=100,  # Max force (Newtons)
                maxVelocity=1.0
            )

        # Apply gripper control (last 2 joints - symmetric)
        gripper_position = 0.04 * (1 - action.gripper)  # 0.04 = fully open
        for i in [7, 8]:  # Gripper finger joints
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=gripper_position,
                force=20
            )

        # Step physics simulation
        p.stepSimulation()
        self.current_step += 1

        # Get new observation
        obs = self._get_observation()

        # Compute reward and check if done
        reward, done, info = self._compute_reward()

        # Episode timeout
        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, info

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """
        Task-specific reward computation.

        This is where multi-task learning happens - different tasks
        have different success criteria!
        """

        info = {"success": False}
        reward = 0.0
        done = False

        # Parse task to extract action and target
        task_lower = self.current_task.lower()

        # Extract color from task
        target_color = None
        for color in ["red", "blue", "green"]:
            if color in task_lower:
                target_color = color
                break

        if target_color is None:
            return 0.0, False, info

        block_id = self.blocks[target_color]
        block_pos, _ = p.getBasePositionAndOrientation(block_id)
        block_pos = np.array(block_pos)

        # Get gripper position
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        gripper_pos = np.array(ee_state[0])

        # Task-specific success conditions
        if "pick" in task_lower:
            # Success: Block lifted above table (z > 0.1)
            if block_pos[2] > 0.1:
                reward = 1.0
                done = True
                info["success"] = True
            else:
                # Dense reward: negative distance to block
                distance = np.linalg.norm(gripper_pos - block_pos)
                reward = -distance

        elif "push" in task_lower:
            # Success: Block moved forward by >10cm
            initial_y = 0.0  # Blocks start at y≈0
            if block_pos[1] > initial_y + 0.1:
                reward = 1.0
                done = True
                info["success"] = True
            else:
                # Reward for pushing forward
                reward = block_pos[1] - initial_y

        elif "place" in task_lower:
            # Success: Block on target (within 5cm)
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

        # Get image from PyBullet camera
        img_arr = p.getCameraImage(
            width=self.image_size[1],
            height=self.image_size[0],
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # Faster rendering
        )

        # Extract RGB (ignore alpha and depth)
        rgb_array = np.array(img_arr[2], dtype=np.uint8)
        rgb_array = rgb_array.reshape(self.image_size[0], self.image_size[1], 4)
        rgb_array = rgb_array[:, :, :3]  # Drop alpha channel

        return rgb_array

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(self.physics_client)
