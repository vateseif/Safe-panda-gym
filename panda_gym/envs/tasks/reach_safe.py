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
        self.unsafe_region_radius = 0.1
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
        self.sim.create_sphere(
            body_name = "unsafe_region_1",
            radius = self.unsafe_region_radius,
            mass = 0.0,
            ghost = True,
            position = np.array([0.0, 0.0, -self.object_size / 2]),
            rgba_color = np.array([0.1, 0.9, 0.1, 0.3]),
        )

        self.sim.create_sphere(
            body_name = "unsafe_region_2",
            radius = self.unsafe_region_radius,
            mass = 0.0,
            ghost = True,
            position = np.array([0.0, 0.0, -self.object_size / 2]),
            rgba_color = np.array([0.1, 0.9, 0.1, 0.3]),
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
        return np.array([random_x_base, -0.25, - self.object_size /2])

    def _goal_in_unsafe_area(self):

        d1 = distance(self.goal, self.unsafe_state_1_pos)
        d2 = distance(self.goal, self.unsafe_state_2_pos)
        min_tresh_distane = self.unsafe_region_radius + 0.02# 0.02 is radius of goal
        if (d1<min_tresh_distane) or (d2<min_tresh_distane):
            return True
        else:
             return False

     
       
       
    def _sample_unsafe_state_right(self):
        random_x_base = self._sample_unsafe_state_x_boundary()
        return np.array([random_x_base, 0.25, -self.object_size / 2])

    def _compute_cost_safe_space(self, achieved_goal):
        d1 = distance( achieved_goal, self.unsafe_state_1_pos)
        d2 = distance(achieved_goal, self.unsafe_state_2_pos)
        min_tresh_distane = self.unsafe_region_radius
        cost_value =  (d1> min_tresh_distane) and (d2> min_tresh_distane)
        if (d1<min_tresh_distane) or (d2<min_tresh_distane):
            return 1.0
        else:
            return 0.0
        
     

    def reset(self) -> None:
       

        self.unsafe_state_1_pos = self._sample_unsafe_state_left()
        self.sim.set_base_pose("unsafe_region_1", self.unsafe_state_1_pos, np.array([0.0, 0.0, 0.0, 1.0]))

        self.unsafe_state_2_pos = self._sample_unsafe_state_right()
        self.sim.set_base_pose("unsafe_region_2", self.unsafe_state_2_pos, np.array([0.0, 0.0, 0.0, 1.0]))

        self.goal = self._sample_goal()

        while self._goal_in_unsafe_area():
            print("initalisation is not fine, change goal")
            self.goal = self._sample_goal()
        # note to self : recursive funtions are slower use loop instead
        # if not self._goal_in_safe_area():
        #     print("initalisation is not fine")
        #     self.reset()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

       

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        cost_safe = self._compute_cost_safe_space(achieved_goal)
        if self.reward_type == "sparse":
            # print("sparse")
            # print("cost safe : ", cost_safe)
            # print("normal_cost : ", np.array(d > self.distance_threshold, dtype=np.float64) )
            return np.array(d > self.distance_threshold, dtype=np.float64) + cost_safe
        else:
            # return d + cost_safe d + cost_safe
            return d + cost_safe
