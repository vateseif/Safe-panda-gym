import numpy as np

from panda_gym.envs.core_multi_task import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.cubes import Cubes
from panda_gym.pybullet import PyBullet


class PandaCubes(RobotTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool=False, debug: bool= False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, debug=debug, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type, body_name="")
        task = Cubes(sim, debug=debug)
        super().__init__([robot], task)
