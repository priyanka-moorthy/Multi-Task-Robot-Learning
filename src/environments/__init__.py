from .base import BaseRobotEnv, Observation, Action

# Try importing both simulators
try:
    from .pybullet_env import FrankaPickAndPlaceEnv
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

try:
    from .mujoco_env import FrankaMuJoCoEnv
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Export what's available
__all__ = ["BaseRobotEnv", "Observation", "Action"]

if PYBULLET_AVAILABLE:
    __all__.append("FrankaPickAndPlaceEnv")

if MUJOCO_AVAILABLE:
    __all__.append("FrankaMuJoCoEnv")
