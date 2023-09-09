from fileinput import filename
from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

#
class MoveTable(Task):
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

        self.table_offset = [np.array([-0.225, 0.0, 0.18]), np.array([0.225, 0.0, 0.18])]

        # sphere for visualization
        radius = 0.015
        self.sim.create_sphere(
          body_name="table_left",
          radius=radius,
          mass=0.,
          position=self._sample_objects()+self.table_offset[0],
          rgba_color=np.array([0.1, 0.9, 0.1, 0.6]),
          ghost=True
        )

        self.sim.create_sphere(
          body_name="table_right",
          radius=radius,
          mass=0.,
          position=self._sample_objects()+self.table_offset[1],
          rgba_color=np.array([0.1, 0.9, 0.1, 0.6]),
          ghost=True
        )

        # load table URDF
        self.sim.loadURDF(body_name="movable_table", 
                        mass=2.0, 
                        fileName="panda_gym/assets/table/table.urdf", 
                        basePosition=self._sample_objects(), 
                        globalScaling=0.3)
        


    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        table_position = np.array(self.sim.get_base_position("movable_table"))
        obs = {
            "table":  table_position,
            "table_left": table_position+self.table_offset[0],
            "table_right": table_position+self.table_offset[1]
        }

        self._update_visualizations(table_position)
        return obs
    
    def _update_visualizations(self, ref_position:np.ndarray):
        self.sim.set_base_pose("table_left", ref_position+self.table_offset[0], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("table_right", ref_position+self.table_offset[1], np.array([0.0, 0.0, 0.0, 1.0]))
        return

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:        
        # NOTE: for some reason set_base_pose of the table causes it to disappear. You can't use reset() but have to run again the environment
        #self.sim.set_base_pose("movable_table",   np.array([0.6, 0.1, 0.1]), np.array([0.0, 0.0, 0.0, 0.0]))
        
        self.sim.set_base_pose("table_left", self._sample_objects()+self.table_offset[0], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("table_right", self._sample_objects()+self.table_offset[1], np.array([0.0, 0.0, 0.0, 1.0]))

        return

    def _sample_goal(self) -> np.ndarray:
        return np.zeros(1)

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0.4, 0.1, 0.1])

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