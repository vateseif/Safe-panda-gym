from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

'''
to do:
figure out the cost function to subtract if a point is in the 3d onject

1 check if goal isn in 3d object during reset if it is resample
2 check if gripper is in un safe state and return reduced cost funtion
'''
class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.object_size = 0.04
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.max_base_unsafe = 0.15 # max x pos to put unsafe space
        self.min_base_unsafe = -0.25
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

       
        # pre_size = np.ones(3) * self.object_size 
        self.unsafe_size_region  = np.array([0.1, 0.1, 0.1])
        self.sim.create_box(
            body_name="unsafe_region_1",
            half_extents = self.unsafe_size_region,
            # half_extents = np.array([ 1, 1.0, self.object_size / 2]) ,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="unsafe_region_2",
            half_extents = self.unsafe_size_region,
            # half_extents = np.array([ 1, 1.0, self.object_size / 2]) ,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position
    def _sample_unsafe_state_x_boundary(self):
        
        return np.random.uniform(self.max_base_unsafe,  self.min_base_unsafe )

    def _sample_unsafe_state_left(self):
        # sample boundary to right after base of robot
          
        random_x_base = self._sample_unsafe_state_x_boundary()
        return np.array([random_x_base, -0.25, self.object_size])

    def _check_if_goal_in_cuboid(self):
        # if
        pos_x , pos_y , pos_z = self.goal
        goal_radius = 0.02 

        pos_box_x, pos_box_y, pos_box_z = self.unsafe_state_pos

        displacement_box_x, displacement_box_y, displacement_box_z = \
                self.unsafe_size_region[0], self.unsafe_size_region[0],self.unsafe_size_region[0]

        # compute for left unsafe box
        if pos_x + goal_radius > pos_box_x + (displacement_box_x / 2) and \
            pos_x + goal_radius > pos_box_x + (displacement_box_y /2) and 
        
        

    def _sample_unsafe_state_right(self):
        random_x_base = self._sample_unsafe_state_x_boundary()
        return np.array([random_x_base, 0.25, self.object_size])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

        self.unsafe_state_pos = self._sample_unsafe_state_left()
        self.sim.set_base_pose("unsafe_region_1", self.unsafe_state_pos, np.array([0.0, 0.0, 0.0, 1.0]))

        self.unsafe_state_2_pos = self._sample_unsafe_state_right()
        self.sim.set_base_pose("unsafe_region_2", self.unsafe_state_2_pos, np.array([0.0, 0.0, 0.0, 1.0]))

        self._check_if_goal_in_cuboid()

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
