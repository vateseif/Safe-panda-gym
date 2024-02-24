from typing import List, Tuple

import os
import numpy as np

from panda_gym import BASE_DIR
from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class CleanPlate(Task):
    def __init__(
        self,
        sim : PyBullet,
        debug: bool = False,
        reward_type="sparse",
        distance_threshold=0.1
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.h, self.w, self.l = (0.03, 0.02, 0.015)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)


        self.sim.loadURDF(body_name="plate", fileName=os.path.join(BASE_DIR, "assets/blue_plate/model.urdf"),
                            basePosition=np.array([0.0, 0.1, 0.01]),
                            useFixedBase=True) # plate cannot be moved

        sponge_position, sponge_scrub_offset = self._sample_objects()   

        self.sim.create_box(
            body_name="sponge",
            half_extents= np.array([self.h, self.w, self.l]),
            mass=1.0,
            position=sponge_position,
            rgba_color=np.array([1, 1, 0, 1.0])
        )

        # sponge scrub
        self.sim.create_box(
            body_name="sponge_scrub",
            half_extents= np.array([self.h, self.w, self.l/3]),
            mass=1.0,
            position=sponge_position+sponge_scrub_offset, 
            rgba_color=np.array([1/255, 50/255, 32/255, 1.0])
        )

        # create constraint between sponge and sponge scrub
        self.sim.create_fixed_constraint("sponge", 
                                        "sponge_scrub", 
                                        sponge_scrub_offset, 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.zeros(3))

    def get_obs(self) -> np.ndarray:
        # position of objects
        obs = {
            "plate": {
                "position": np.array(self.sim.get_base_position("plate")),
                "orientation": np.array(self.sim.get_base_orientation("plate")),
                "size": self.h
            },
            "sponge": {
                "position": np.array(self.sim.get_base_position("sponge")),
                "orientation": np.array(self.sim.get_base_orientation("sponge")),
                "size": self.l
            }
        }
        return obs

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:   
        sponge_position, sponge_scrub_offset = self._sample_objects()   
        self.sim.set_base_pose("sponge", sponge_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("sponge_scrub", sponge_position+sponge_scrub_offset, np.array([0.0, 0.0, 0.0, 1.0]))
        # create constraint between sponge and sponge scrub
        self.sim.create_fixed_constraint("sponge", 
                                        "sponge_scrub", 
                                        sponge_scrub_offset, 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.zeros(3))

    def _sample_goal(self) -> np.ndarray:
        return np.array([10, 10, 10])

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        sponge_position = np.array([0.0, -0.1, 0.1])
        sponge_scrub_offset = np.array([0.,0.,self.l*(1+1/3)])
        return sponge_position, sponge_scrub_offset

    def _get_object_orietation(self):
        return

    def is_success(self):
        # harcoded to False
        return False
    
    def compute_cost(self):
        # hardcoded to 0
        return 0

    def compute_reward(self):
        # harcoded to 0
        return 0
