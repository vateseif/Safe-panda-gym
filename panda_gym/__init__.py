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
            id="PandaReach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaReachSafe{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaReachSafeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
      

        register(
            id="PandaPush{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPushSafe{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPushSafeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlideSafe{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaSlideSafeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlaceSafe{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPickAndPlaceSafeEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlacePlatform{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaPickAndPlacePlatformEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
        

        register(
            id="PandaStack{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaStackEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )
          
        register(
            id="PandaStackSafe{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaStackSafeEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )


        register(
            id="PandaStack3{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaStack3Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="PandaStackPyramid{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaStackPyramidEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="PandaBuildL{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaBuildLEnv",
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

        register(
            id="PandaFlip{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaFlipEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
