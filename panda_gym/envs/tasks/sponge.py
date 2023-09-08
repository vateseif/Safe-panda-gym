from fileinput import filename
from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class Sponge(Task):
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
        self.sim.create_table(length=2.2, width=0.7, height=0.4, x_offset=-0.3)


        self.sim.loadURDF(body_name="plate", fileName="/Users/seifboss/thesis/Safe-panda-gym/panda_gym/assets/urdf_models/models/blue_plate/model.urdf",
                            basePosition=np.array([0.0, 0.1, 0.01]),
                            useFixedBase=True) # plate cannot be moved

        h, w, l = (0.03, 0.02, 0.02)
        self.sim.create_box(
            body_name="sponge",
            half_extents= np.array([h, w, l]),
            mass=1.0,
            position=np.array([0.0, 0.0, l/2]),
            rgba_color=np.array([1, 1, 0, 1.0]),
        )

        
        self.sim.create_sink(base_position=np.array([0, -0.5, 0.]))

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        plate_position = np.array(self.sim.get_base_position("plate"))
        plate_rotation = np.array(self.sim.get_base_rotation("plate"))
        plate_velocity = np.array(self.sim.get_base_velocity("plate"))
        plate_angular_velocity = np.array(self.sim.get_base_angular_velocity("plate"))

        observation = np.concatenate(
            [
                plate_position,
                plate_rotation,
                plate_velocity,
                plate_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:

        return np.array([10, 10, 10])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        
        #self.sim.set_base_pose("plate",   np.array([0.0, 0.1, 0.1]), np.array([0.0, 0.0, 0.0, 0.0]))
        self.sim.set_base_pose("sponge",   np.array([0.0, -0.1, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))


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
