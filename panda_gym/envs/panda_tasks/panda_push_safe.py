import numpy as np

from panda_gym.envs.core_safe import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.push_safe import Push
from panda_gym.pybullet import PyBullet


class PandaPushSafeEnv(RobotTaskEnv):
    """Push task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "dense", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Push(sim, reward_type="dense", get_ee_position=robot.get_ee_position)
        super().__init__([robot], task)
