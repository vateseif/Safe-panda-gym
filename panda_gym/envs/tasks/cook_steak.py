from typing import Tuple

import os
import numpy as np

from panda_gym import BASE_DIR
from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class CookSteak(Task):
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

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.8, width=0.6, height=0.4, x_offset=-0.15, y_offset=-0.2)

        self.handle_offsets = [np.array([-0.15, 0.0, 0.02]), np.array([0.15, 0.0, 0.02])]

        pan_position, oven_position, steak_position = self._sample_objects()

        # load table URDF
        self.sim.loadURDF(body_name="pan", 
                        mass=2.0, 
                        fileName=os.path.join(BASE_DIR, "assets/blue_plate/model.urdf"), 
                        basePosition=pan_position, 
                        baseOrientation=[0,0,1,1],
                        globalScaling=1.5)

        # load handle 
        self.sim.create_handle("pan_handle_left", base_position=pan_position+ np.array([0.0, self.handle_offsets[0][0], self.handle_offsets[0][2]]))
        self.sim.create_handle("pan_handle_right", base_position=pan_position+ np.array([0.0, self.handle_offsets[1][0], self.handle_offsets[1][2]]))

        # load oven
        self.sim.create_oven("oven", base_position=oven_position)
        # load steak
        self.sim.create_steak("steak", base_position=steak_position)

        # create constraints between handles and table
        self.sim.create_fixed_constraint("pan", 
                                        "pan_handle_left", 
                                        self.handle_offsets[0], 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.array([0., -1., 1., 0.]))

        self.sim.create_fixed_constraint("pan", 
                                        "pan_handle_right", 
                                        self.handle_offsets[1], 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.array([0., -1., 1., 0.]))


    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        obs = {
            "pan": {
                "position": np.array(self.sim.get_base_position("pan")),
                "orientation": np.array(self.sim.get_base_orientation("pan")),
                "size": 0.06
            },
            "pan_handle_left": {
                "position": np.array(self.sim.get_base_position("pan_handle_left")) + np.array([0.0, 0.0, 0.015]),
                "orientation": np.array(self.sim.get_base_orientation("pan_handle_left")),
                "size": 0.0
            },
            "pan_handle_right": {
                "position": np.array(self.sim.get_base_position("pan_handle_right")) + np.array([0.0, 0.0, 0.015]),
                "orientation": np.array(self.sim.get_base_orientation("pan_handle_right")),
                "size": 0.0
            },
            "oven": {
                "position": np.array(self.sim.get_base_position("oven")),
                "orientation": np.array(self.sim.get_base_orientation("oven")),
                "size": 0.65
            },
            "burner_plate": {
                "position": np.array(self.sim.get_base_position("oven")) + np.array([-0.05, -0.27, 0.7]),
                "orientation": np.array(self.sim.get_base_orientation("oven")),
                "size": 0.0
            },
            "steak": {
                "position": np.array(self.sim.get_base_position("steak")),
                "orientation": np.array(self.sim.get_base_orientation("steak")),
                "size": 0.035
            }
        }

        return obs
    
    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:        
        # NOTE: for some reason set_base_pose of the table causes it to disappear. You can't use reset() but have to run again the environment
        pan_position, _, steak_position = self._sample_objects()

        self.sim.set_base_pose("pan_handle_left", pan_position+np.array([0.0, self.handle_offsets[0][0], self.handle_offsets[0][2]]), np.array([-1, 1, -1, 1]))
        self.sim.set_base_pose("pan_handle_right", pan_position+np.array([0.0, self.handle_offsets[1][0], self.handle_offsets[1][2]]), np.array([-1, 1, -1, 1]))

        self.sim.set_base_pose("pan", pan_position, np.array([0.0, 0.0, 1.0, 1.0]))

        self.sim.set_base_pose("steak", steak_position, np.array([0.0, 0.0, 0.0, 1.0]))

        # create constraints between handles and table
        self.sim.create_fixed_constraint("pan", 
                                        "pan_handle_left", 
                                        self.handle_offsets[0], 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.array([0., -1., 1., 0.]))

        self.sim.create_fixed_constraint("pan", 
                                        "pan_handle_right", 
                                        self.handle_offsets[1], 
                                        np.zeros(3), 
                                        np.zeros(3), 
                                        np.array([0., -1., 1., 0.]))

        return

    def _sample_goal(self) -> np.ndarray:
        return np.zeros(1)

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        pan_position = np.array([-0.1, -0.1, 0.])
        oven_position = np.array([-0.15, 0.6, -0.4])
        steak_position = np.array([-0.35, -0.1, 0.05])
        return pan_position, oven_position, steak_position# np.array([-0.1, -0.1, 0.])

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