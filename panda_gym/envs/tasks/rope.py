from fileinput import filename
from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class Rope(Task):
    def __init__(
        self,
        sim : PyBullet,
        reward_type="sparse",
        distance_threshold=0.1
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)


        #self.sim.loadURDF(body_name="rope", fileName="/Users/seifboss/thesis/Safe-panda-gym/panda_gym/assets/rope/rope_gpt.urdf",  basePosition=np.array([0.0, -0.2, 0.1]))
        self.sim.create_rope2()
        self.sim.loadURDF(body_name="cup", fileName="/Users/seifboss/thesis/Safe-panda-gym/panda_gym/assets/yellow_cup/model.urdf", basePosition=np.array([0.0, 0.1, 0.1]))


    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        cup_position = np.array(self.sim.get_base_position("cup"))
        cup_rotation = np.array(self.sim.get_base_rotation("cup"))
        cup_velocity = np.array(self.sim.get_base_velocity("cup"))
        cup_angular_velocity = np.array(self.sim.get_base_angular_velocity("cup"))

        observation = np.concatenate(
            [
                cup_position,
                cup_rotation,
                cup_velocity,
                cup_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:

        return np.array([10, 10, 10])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        
        self.sim.set_base_pose("cup",   np.array([0.0, 0.1, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("rope",   np.array([0., -0.2, 0.1]), np.array([0.0, 0.5, 0.0, 1.0]))


    def _sample_goal(self) -> np.ndarray:

        return np.array([10, 10, 10])

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # 
        return 

    def _get_object_orietation(self):
        
        return

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return np.array((d < self.distance_threshold), dtype=np.float64)
    def compute_cost(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = ...) -> Union[np.ndarray, float]:
        
        cost = np.array([1000, 1000, 1000])

        if self.reward_type == "sparse":
            cost_sparse = cost.copy()
            for key in cost_sparse:
                sparse_co = np.array((cost_sparse[key] > self.distance_threshold), dtype=np.float64)
                cost_sparse[key] = sparse_co
            return cost_sparse
        else:
            return cost 

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
