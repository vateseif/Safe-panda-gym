from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet


class  Cubes(Task):
    def __init__(
        self,
        sim : PyBullet,
        debug: bool = False,
        obj_xy_range=0.2,
    ) -> None:
        super().__init__(sim)
        self.debug = debug
        self.object_size = 0.04
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=1.5, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="blue_cube",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )

        self.sim.create_box(
            body_name="green_cube",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="orange_cube",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([255/255, 165/255, 0., 1.0]),
        )


        self.sim.create_box(
            body_name="red_cube",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )
        
        if self.debug:
            self._create_visuals()

    def _create_visuals(self) -> None:
        radius = 0.03

        self.sim.create_sphere(
            body_name="blue_cube_sphere",
            radius=radius,
            mass=0.,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
            ghost=True
        )

        self.sim.create_sphere(
            body_name="green_cube_sphere",
            radius=radius,
            mass=0.,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
            ghost=True
        )

        self.sim.create_sphere(
            body_name="orange_cube_sphere",
            radius=radius,
            mass=0.,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
            ghost=True
        )

        self.sim.create_sphere(
            body_name="red_cube_sphere",
            radius=radius,
            mass=0.,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
            ghost=True
        )
        

    def get_obs(self) -> Dict[str, np.ndarray]:
        # position of objects
        obs = {
            "blue_cube": self.sim.get_base_position("blue_cube"),
            "green_cube": self.sim.get_base_position("green_cube"),
            "orange_cube": self.sim.get_base_position("orange_cube"),
            "red_cube": self.sim.get_base_position("red_cube"),
            "blue_cube_orientation": self.sim.get_base_orientation("blue_cube"),
            "green_cube_orientation": self.sim.get_base_orientation("green_cube"),
            "orange_cube_orientation": self.sim.get_base_orientation("orange_cube"),
            "red_cube_orientation": self.sim.get_base_orientation("red_cube")
        }

        if self.debug:
            self._update_visuals()

        return obs
    
    def _update_visuals(self) -> None:
        try:
            self.sim.set_base_pose("blue_cube_sphere", self.sim.get_base_position("blue_cube"), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("green_cube_sphere", self.sim.get_base_position("green_cube"), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("orange_cube_sphere", self.sim.get_base_position("orange_cube"), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("red_cube_sphere", self.sim.get_base_position("red_cube"), np.array([0.0, 0.0, 0.0, 1.0]))
        except:
            pass

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(1)

    def reset(self) -> None:
        object1_position, object2_position, object3_position, object4_position = self._sample_objects()
        
        self.sim.set_base_pose("blue_cube",  object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("green_cube",  object2_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("orange_cube",  object3_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("red_cube",  object4_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        # harcoded
        return np.zeros(1)

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([-0.05, 0.3, self.object_size / 2])
        object2_position = np.array([-0.05, 0.1, self.object_size / 2])
        object3_position = np.array([-0.05, -0.1, self.object_size / 2])
        object4_position = np.array([-0.05, -0.3, self.object_size / 2])
        
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise3 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise4 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position+noise1, object2_position+noise2, object3_position+noise3, object4_position+noise4

    def _get_object_orietation(self):
        object1_rotation = np.array(self.sim.get_base_rotation("blue_cube", "quaternion"))
        object2_rotation = np.array(self.sim.get_base_rotation("green_cube", "quaternion"))
        object3_rotation = np.array(self.sim.get_base_rotation("orange_cube", "quaternion"))
        object4_rotation = np.array(self.sim.get_base_rotation("red_cube", "quaternion"))
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
