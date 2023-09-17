from fileinput import filename
from typing import Any, Dict, Tuple, Union

import os
import numpy as np

from panda_gym import BASE_DIR
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

        # sponge dimensions
        self.h, self.w, self.l = (0.03, 0.02, 0.015)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.2, width=0.7, height=0.4, x_offset=-0.3)

        # get object positions
        plate_position, plate_handle_offset, plate_handle_orientation, sponge_position, sink_position  = self._sample_objects()
        # plate
        self.sim.loadURDF(body_name="plate", mass=1., fileName=os.path.join(BASE_DIR, "assets/blue_plate/model.urdf"),
                            basePosition=plate_position,
                            useFixedBase=False) # plate cannot be moved
        # plate handle
        self.sim.create_cylinder(
            body_name="plate_handle",
            radius=0.01,
            height=0.1,
            mass=0.01,
            position=plate_position+plate_handle_offset,
            orientation=plate_handle_orientation,
            rgba_color=np.array([96/255, 59/255, 42/255, 1.0])
        )
        # pedestal
        self.sim.create_box(
            body_name="pedestal",
            half_extents= np.array([0.1, 0.1, plate_position[2]/2]),
            mass=0,
            position=plate_position - np.array([0., 0., plate_position[2]]), 
            rgba_color=np.array([189/255, 174/255, 153/255, 1]),
        )
        # sponge
        self.sim.create_box(
            body_name="sponge",
            half_extents= np.array([self.h, self.w, self.l]),
            mass=1.0,
            position=sponge_position, 
            rgba_color=np.array([1, 1, 0, 1.0]),
        )
        # sponge scrub
        self.sim.create_box(
            body_name="sponge_scrub",
            half_extents= np.array([self.h, self.w, self.l/3]),
            mass=1.0,
            position=sponge_position+np.array([0.,0.,self.l*(1+1/3)]), 
            rgba_color=np.array([1/255, 50/255, 32/255, 1.0])
        )
        # sink
        self.sim.create_sink(base_position=sink_position)

        # create constraint between plate and handle
        self.sim.create_fixed_constraint("plate", 
                                        "plate_handle", 
                                        plate_handle_offset, 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        [0, -1, 0, 1])

        # create constraint between sponge and sponge scrub
        self.sim.create_fixed_constraint("sponge", 
                                        "sponge_scrub", 
                                        np.array([0.,0.,self.l*(1+1/3)]), 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.zeros(3))

    def get_obs(self) -> np.ndarray:
        # position of objects
        obs = {
            "sink": np.array(self.sim.get_base_position("sink")),
            "plate": np.array(self.sim.get_base_position("plate")),
            "sponge": np.array(self.sim.get_base_position("sponge"))
        }
        return obs

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:  
        plate_position, plate_handle_offset, plate_handle_orientation, sponge_position, sink_position  = self._sample_objects()       
        #self.sim.set_base_pose("plate",   np.array([0.0, 0.1, 0.1]), np.array([0.0, 0.0, 0.0, 0.0]))
        self.sim.set_base_pose("sponge",  sponge_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        return np.zeros(1)

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        plate_position = np.array([0.0, -0.1, 0.03])
        plate_handle_offset = np.array([0.14, 0.0, 0.007])
        sponge_position = np.array([0.0, 0.15, 0.1])
        sink_position = np.array([0, -0.5, 0.])
        plate_handle_orientation = np.array([0., 1., 0., 1.])
        return plate_position, plate_handle_offset, plate_handle_orientation, sponge_position, sink_position 

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
