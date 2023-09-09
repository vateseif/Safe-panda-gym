import numpy as np

from panda_gym.envs.core_multi_task import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.move_table import MoveTable
from panda_gym.pybullet import PyBullet


class PandaMoveTable(RobotTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot1 = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), base_orientation=np.array([0, 0, 0]), control_type=control_type)
        #robot2 = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), base_orientation=np.array([0, 0, 0]), control_type=control_type)
        robots = [robot1]
        task = MoveTable(sim, reward_type=reward_type)
        super().__init__(robots, task)