import os

from gym.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaCubes{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaCubes",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaRope{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaRope",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaCleanPlate{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaCleanPlate",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaSponge{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaSponge",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaMoveTable{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaMoveTable",
            kwargs=kwargs,
            max_episode_steps=100000,
        )
