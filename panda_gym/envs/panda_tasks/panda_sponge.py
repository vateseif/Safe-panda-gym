import numpy as np

from panda_gym.envs.core_multi_task import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.sponge import Sponge
from panda_gym.pybullet import PyBullet


class PandaSponge(RobotTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot1 = Panda(sim, block_gripper=False, base_position=np.array([-0.8, 0.0, 0.0]), base_orientation=np.array([0, 0, 0.]), control_type=control_type, body_name="_left")
        robot2 = Panda(sim, block_gripper=False, base_position=np.array([0.6, 0.0, 0.0]), base_orientation=np.array([0, 0, np.pi]), control_type=control_type, body_name="_right", base_gripper_orientation=np.array([np.pi,0.,np.pi]))
        robots = [robot1, robot2]
        task = Sponge(sim, reward_type=reward_type)
        super().__init__(robots, task)