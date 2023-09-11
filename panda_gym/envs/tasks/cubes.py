from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet


class  Cubes(Task):
    def __init__(
        self,
        sim : PyBullet,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="cube_1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )

        self.sim.create_box(
            body_name="cube_2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="cube_3",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )


        self.sim.create_box(
            body_name="cube_4",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )


    def get_obs(self) -> Dict[str, np.ndarray]:
        # position of objects
        obs = {
            "cube_1": self.sim.get_base_position("cube_1"),
            "cube_2": self.sim.get_base_position("cube_2"),
            "cube_3": self.sim.get_base_position("cube_3"),
            "cube_4": self.sim.get_base_position("cube_4"),
        }

        return obs

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:
        object1_position, object2_position, object3_position, object4_position = self._sample_objects()
        
        self.sim.set_base_pose("cube_1",  object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("cube_2",  object2_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("cube_3",  object3_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("cube_4",  object4_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        # harcoded
        return np.zeros(1)

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.3, self.object_size / 2])
        object2_position = np.array([0.0, 0.1, self.object_size / 2])
        object3_position = np.array([0.0, -0.1, self.object_size / 2])
        object4_position = np.array([0.0, -0.3, self.object_size / 2])
        
        #noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #noise3 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #noise4 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position, object2_position, object3_position, object4_position

    def _get_object_orietation(self):
        object1_rotation = np.array(self.sim.get_base_rotation("cube_1", "quaternion"))
        object2_rotation = np.array(self.sim.get_base_rotation("cube_2", "quaternion"))
        object3_rotation = np.array(self.sim.get_base_rotation("cube_3", "quaternion"))
        object4_rotation = np.array(self.sim.get_base_rotation("cube_4", "quaternion"))
        return object1_rotation, object2_rotation, object3_rotation, object4_rotation

    def is_success(self):
        # harcoded to False
        return False
    
    def compute_cost(self):
        # hardcoded to 0
        return 0

    def compute_reward(self):
        # harcoded to 0
        return 0
